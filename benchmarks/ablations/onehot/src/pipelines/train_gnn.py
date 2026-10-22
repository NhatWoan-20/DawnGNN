#!/usr/bin/env python3
"""
Training script for GAT-based malware detection model with One-Hot encoding.
Includes early stopping, learning rate scheduling, checkpointing, and comprehensive metrics.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import shutil

# Import utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import get_project_root, get_model_path, ensure_dir, get_data_path, PROJECT_ROOT
from utils.logger import setup_logger

# Get project roots
DAWNGNN_ONEHOT_ROOT = get_project_root()

# Setup logger
logger = setup_logger("train_gnn")

CONFIG_FILE_PATH = DAWNGNN_ONEHOT_ROOT.joinpath('configs', 'onehot.yaml')


def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        
        # Auto-detect input_dim from vocabulary if not set
        if config['model']['input_dim'] is None:
            vocab_file = PROJECT_ROOT / config['paths']['vocabulary_file']
            if vocab_file.exists():
                with open(vocab_file, 'r') as f:
                    vocabulary = json.load(f)
                vocab_size = len(vocabulary)
                config['model']['input_dim'] = vocab_size
                logger.info(f"Auto-detected input_dim={vocab_size} from vocabulary")
            else:
                raise ValueError("input_dim is None and vocabulary file not found. Please run generate_onehot_embeddings.py first.")
        
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


# ==================== A. DATASET LOADER ====================
class MalwareGraphDataset(Dataset):
    """Dataset for loading malware graphs."""
    
    def __init__(self, root, config, split_type: str):
        self.config = config
        self.root = root
        self.split_type = split_type  # 'train', 'val', or 'test'
        self.graphs_dir = PROJECT_ROOT.joinpath(config['paths']['output_augmented_dir'])
        self.labels_path = PROJECT_ROOT.joinpath(config['paths']['labels_file'])
        self.splits_dir = PROJECT_ROOT.joinpath(config['paths']['splits_dir'])
        
        # Load labels and sample IDs
        try:
            with open(self.labels_path, 'r') as f:
                self.labels = json.load(f)
        except FileNotFoundError:
            logger.error(f"Labels file not found: {self.labels_path}")
            raise
        
        split_file = self.splits_dir.joinpath(f'{self.split_type}.json')
        try:
            with open(split_file, 'r') as f:
                self.sample_ids = json.load(f)
        except FileNotFoundError:
            logger.error(f"Split file not found: {split_file}")
            raise

        super().__init__(root)
        logger.info(f"Loaded {split_type} dataset with {len(self.sample_ids)} samples")

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{sha}.gpickle' for sha in self.sample_ids]

    @property
    def processed_file_names(self) -> List[str]:
        return [] 

    def len(self):
        return len(self.sample_ids)

    def get(self, idx):
        sha = self.sample_ids[idx]
        graph_path = self.graphs_dir.joinpath(f'{sha}.gpickle')
        
        # Load NetworkX graph
        try:
            with open(graph_path, 'rb') as f:
                G_nx = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load graph {graph_path}: {e}")
            return None

        # Convert to PyG Data
        node_features = []
        mapping = {}
        for i, (node_id, data) in enumerate(G_nx.nodes(data=True)):
            mapping[node_id] = i
            if 'x' not in data:
                logger.warning(f"Node {node_id} missing 'x' feature, skipping graph")
                return None
            node_features.append(data['x'])

        if not node_features:
            return None 
             
        x = torch.tensor(np.stack(node_features), dtype=torch.float)

        # Edge Index
        edge_index_list = []
        for u, v, data in G_nx.edges(data=True):
            edge_index_list.append([mapping[u], mapping[v]])
            
        if len(edge_index_list) == 0:
            # Empty graph, create dummy edge
            edge_index = torch.zeros((2, 1), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        y = torch.tensor([self.labels[sha]], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)


# ==================== B. MODEL ARCHITECTURE ====================
class GATClassifier(torch.nn.Module):
    """GAT-based classifier for malware detection."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, dropout, pooling_type="mean"):
        super(GATClassifier, self).__init__()
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.convs = torch.nn.ModuleList()
        
        # First GAT layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
            
        # Final layer (single head)
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=dropout))
        
        # Classifier
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GAT layers
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Final layer
        x = self.convs[-1](x, edge_index)

        # Global pooling (support different pooling types)
        if self.pooling_type == "max":
            x = global_max_pool(x, batch)
        elif self.pooling_type == "sum":
            x = global_add_pool(x, batch)
        else:  # default: "mean"
            x = global_mean_pool(x, batch)

        # Classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x


# ==================== C. TRAINING FUNCTIONS ====================
def train(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for data in loader:
        if data is None:
            continue
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
        # Collect predictions for metrics
        pred = out.argmax(dim=-1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    return avg_loss, acc, precision, recall, f1


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, float, float, float, np.ndarray]:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    for data in loader:
        if data is None:
            continue
        data = data.to(device)
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        total_loss += loss.item() * data.num_graphs
        
        pred = out.argmax(dim=-1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, acc, precision, recall, f1, cm


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir: Path, is_best: bool = False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        shutil.copy(checkpoint_path, best_path)
        logger.info(f"Saved best model to {best_path}")
    
    # Keep only last N checkpoints
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    if len(checkpoints) > 3:  # Keep last 3
        for old_checkpoint in checkpoints[:-3]:
            old_checkpoint.unlink()


def main():
    # Load config
    config = load_config(CONFIG_FILE_PATH)
    
    MODEL_CONFIG = config['model']
    TRAINING_CONFIG = config['training']
    DEVICE = torch.device(TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")

    # Ensure model directory exists
    model_dir = get_model_path('gnn')
    ensure_dir(model_dir)
    checkpoint_dir = model_dir
    logger.info(f"Model directory: {model_dir}")

    # Load data
    logger.info("Loading datasets...")
    root_dir = PROJECT_ROOT.joinpath('data')
    
    train_dataset = MalwareGraphDataset(root=root_dir, config=config, split_type='train')
    val_dataset = MalwareGraphDataset(root=root_dir, config=config, split_type='val')
    
    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'])

    # Initialize model
    model = GATClassifier(
        in_channels=MODEL_CONFIG['input_dim'],
        hidden_channels=MODEL_CONFIG['hidden_dim'],
        out_channels=2,
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout=MODEL_CONFIG['dropout'],
        pooling_type=MODEL_CONFIG.get('pooling_type', 'mean')
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'], 
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    lr_scheduler = None
    if TRAINING_CONFIG.get('lr_scheduler', {}).get('enabled', False):
        lr_config = TRAINING_CONFIG['lr_scheduler']
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=lr_config.get('mode', 'min'),
            factor=lr_config.get('factor', 0.5),
            patience=lr_config.get('patience', 5),
            min_lr=lr_config.get('min_lr', 1e-5),
        )
        logger.info("Learning rate scheduler enabled")

    # Early stopping
    early_stopping = None
    if TRAINING_CONFIG.get('early_stopping', {}).get('enabled', False):
        es_config = TRAINING_CONFIG['early_stopping']
        early_stopping = {
            'patience': es_config.get('patience', 10),
            'min_delta': es_config.get('min_delta', 0.0001),
            'counter': 0,
            'best_score': None,
            'best_epoch': 0
        }
        logger.info(f"Early stopping enabled with patience={early_stopping['patience']}")

    # Training loop
    logger.info(f"Starting training on {len(train_dataset)} samples...")
    best_val_acc = 0
    best_val_loss = float('inf')
    training_history = []

    for epoch in range(1, TRAINING_CONFIG['epochs'] + 1):
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1 = train(model, train_loader, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = evaluate(model, val_loader, DEVICE)
        
        # Update learning rate
        if lr_scheduler:
            lr_scheduler.step(val_loss)
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1,
            'lr': optimizer.param_groups[0]['lr']
        }
        training_history.append(metrics)
        
        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Save best model
        is_best = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            is_best = True
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
        
        # Checkpointing
        checkpoint_enabled = TRAINING_CONFIG.get('checkpoint', {}).get('enabled', False)
        save_every = TRAINING_CONFIG.get('checkpoint', {}).get('save_every', 10)
        
        if checkpoint_enabled and (epoch % save_every == 0 or is_best):
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best)
        
        # Early stopping
        if early_stopping:
            if early_stopping['best_score'] is None:
                early_stopping['best_score'] = val_acc
                early_stopping['best_epoch'] = epoch
            elif val_acc > early_stopping['best_score'] + early_stopping['min_delta']:
                early_stopping['best_score'] = val_acc
                early_stopping['best_epoch'] = epoch
                early_stopping['counter'] = 0
            else:
                early_stopping['counter'] += 1
                if early_stopping['counter'] >= early_stopping['patience']:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best validation accuracy: {early_stopping['best_score']:.4f} at epoch {early_stopping['best_epoch']}"
                    )
                    break
    
    # Save training history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

