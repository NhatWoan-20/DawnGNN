#!/usr/bin/env python3
"""
train_gnn.py — Training pipeline for DawnGNN Graph Attention Network (GAT) classifier.

Methodology (DawnGNN Phase 4 - Graph Representation Learning & Classification):
--------------------------------------------------------------------------------
1. Graph Attention Network (GAT) Architecture (Veličković et al.):
   Unlike standard GCNs that assign fixed weights based solely on node degrees, GAT computes
   dynamic attention coefficients between adjacent API call nodes:
       alpha_{ij} = softmax_j ( LeakyReLU( a^T [ W h_i || W h_j ] ) )
   This allows the model to selectively prioritize critical transitions (e.g., VirtualAllocEx -> WriteProcessMemory)
   over ubiquitous standard routines (e.g., NtClose).

2. Multi-Head Attention Mechanism:
   - Layer 1 (Input -> Hidden): Multi-head attention (heads=K) with output concatenation:
         h_i^(1) = ||_{k=1}^K sigma( sum_{j in N(i)} alpha_{ij}^k W^k x_j )  in R^{K * hidden_dim}
   - Intermediate Hidden Layers: Multi-head attention with concatenation.
   - Final Graph Convolution Layer: Single-head attention (heads=1, concat=False) projecting to hidden_dim.

3. Global Readout Pooling & Classification:
   - Aggregates variable-sized node sets V into a fixed-dimensional graph representation h_G:
         h_G = Readout({ h_v^(L) | v in V })  [Default: Global Mean Pooling]
   - Fully-connected linear layer projects h_G to 2 output logits (Benign vs. Malware).

4. Optimization & Safeguards:
   - Loss: Binary Cross-Entropy (torch.nn.functional.cross_entropy).
   - Optimizer: Adam with weight decay L2 regularization.
   - Scheduling: ReduceLROnPlateau dynamically decays learning rate when validation loss plateaus.
   - Regularization: Dropout (p=0.4) applied between convolutional layers and before the classifier.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool, global_mean_pool
import yaml

# -----------------------------------------------------------------------------
# Path and Utility Bootstrap (Supports both direct script and module execution)
# -----------------------------------------------------------------------------
try:
    from dawngnn.utils.logger import setup_logger
    from dawngnn.utils.paths import ensure_dir, get_config_path, get_data_path, get_model_path, get_project_root
except ImportError:
    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from utils.logger import setup_logger
    from utils.paths import ensure_dir, get_config_path, get_data_path, get_model_path, get_project_root

# Root directories (DawnGNN project root)
PROJECT_ROOT = get_project_root()
DAWNGNN_ROOT = PROJECT_ROOT

# Setup logger
logger = setup_logger("train_gnn")

# Path to model configuration
CONFIG_FILE_PATH = get_config_path("gnn.yaml")


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load hyperparameters, training options, and paths from YAML configuration file.

    Args:
        path: Absolute Path to the gnn.yaml configuration file.

    Returns:
        Dict containing 'model', 'training', 'paths', and 'embedding' sections.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


# =============================================================================
# A. DATASET LOADER (Exact original logic from gnn-malware-detection)
# =============================================================================
class MalwareGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset loader for semantically-augmented malware call graphs.

    Reads serialized NetworkX .gpickle files, extracts pre-assigned node features ('x' in R^768),
    maps string API nodes to contiguous integer indices [0, |V|-1], constructs the directed
    edge_index tensor, and associates the binary ground-truth label y in {0, 1}.
    """

    def __init__(self, root: str | Path, config: Dict[str, Any], split_type: str):
        """
        Initialize dataset for a specific split partition ('train', 'val', or 'test').

        Args:
            root: Root data directory path.
            config: Loaded gnn.yaml configuration dictionary.
            split_type: Partition name corresponding to data/splits/{split_type}.json.
        """
        self.config = config
        self.root = root
        self.split_type = split_type  # 'train', 'val', or 'test'
        self.graphs_dir = PROJECT_ROOT.joinpath(config["paths"]["output_augmented_dir"])
        self.labels_path = PROJECT_ROOT.joinpath(config["paths"]["labels_file"])
        self.splits_dir = PROJECT_ROOT.joinpath(config["paths"]["splits_dir"])

        # Load labels mapping (sha256 -> int label: 0 for Benign, 1 for Malware)
        try:
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
        except FileNotFoundError:
            logger.error(f"Labels file not found: {self.labels_path}")
            raise

        # Load split sample IDs from JSON partition
        split_file = self.splits_dir.joinpath(f"{self.split_type}.json")
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                self.sample_ids = json.load(f)
        except FileNotFoundError:
            logger.error(f"Split file not found: {split_file}")
            raise

        super().__init__(str(root))
        logger.info(f"Loaded {split_type} dataset with {len(self.sample_ids)} samples")

    @property
    def raw_file_names(self) -> List[str]:
        return [f"{sha}.gpickle" for sha in self.sample_ids]

    @property
    def processed_file_names(self) -> List[str]:
        return []

    def len(self) -> int:
        return len(self.sample_ids)

    def get(self, idx: int) -> Data | None:
        """
        Load and convert a single sample into a PyTorch Geometric Data object.

        Args:
            idx: Integer index into self.sample_ids.

        Returns:
            torch_geometric.data.Data with attributes:
                - x: Tensor of shape (|V|, 768), float32 node embedding features.
                - edge_index: Tensor of shape (2, |E|), long integer directed edges.
                - y: Tensor of shape (1,), long integer label (0 or 1).
            Returns None if the graph file is unreadable or missing required features.
        """
        sha = self.sample_ids[idx]
        graph_path = self.graphs_dir.joinpath(f"{sha}.gpickle")

        # Load NetworkX graph
        try:
            with open(graph_path, "rb") as f:
                G_nx = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load graph {graph_path}: {e}")
            return None

        # Convert to PyG Data: Node Features
        node_features = []
        mapping = {}
        for i, (node_id, data) in enumerate(G_nx.nodes(data=True)):
            mapping[node_id] = i
            if "x" not in data:
                logger.warning(f"Node {node_id} missing 'x' feature, skipping graph")
                return None
            node_features.append(data["x"])

        if not node_features:
            return None

        x = torch.tensor(np.stack(node_features), dtype=torch.float)

        # Convert to PyG Data: Edge Index
        edge_index_list = []
        for u, v, data in G_nx.edges(data=True):
            edge_index_list.append([mapping[u], mapping[v]])

        if len(edge_index_list) == 0:
            # Self-loop / isolated node fallback for graphs without transitions
            edge_index = torch.zeros((2, 1), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        y = torch.tensor([self.labels[sha]], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)


# =============================================================================
# B. MODEL ARCHITECTURE (Exact original logic from gnn-malware-detection)
# =============================================================================
class GATClassifier(torch.nn.Module):
    """
    Graph Attention Network (GAT) Classifier for Windows Malware Detection.

    Architecture:
    -------------
    1. Input GAT Layer:
       in_channels (768) -> hidden_channels (128) with num_heads attention heads.
       Multi-head outputs are concatenated -> dim: hidden_channels * num_heads.
    2. Hidden GAT Layers (num_layers - 2):
       (hidden_channels * num_heads) -> hidden_channels with num_heads attention heads.
       Concatenated output -> dim: hidden_channels * num_heads.
    3. Final GAT Layer:
       (hidden_channels * num_heads) -> hidden_channels with 1 head (concat=False).
    4. Global Readout Pooling:
       Aggregates node embeddings across the graph (mean, sum, or max pooling).
    5. Classification Head:
       Dropout -> Linear(hidden_channels, out_channels=2).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        pooling_type: str = "mean",
    ):
        super(GATClassifier, self).__init__()
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.convs = torch.nn.ModuleList()

        # First GAT layer (concat=True by default)
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))

        # Hidden GAT layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
            )

        # Final GAT layer (single head, concat=False)
        self.convs.append(
            GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        )

        # Linear classifier head
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        """
        Forward propagation pass.

        Args:
            data: PyTorch Geometric Data or Batch instance containing x, edge_index, and batch.

        Returns:
            Tensor of unnormalized logits of shape (batch_size, out_channels).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GAT convolutional layers with ReLU activation and Dropout regularization
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final convolution (projects node representations to hidden_channels)
        x = self.convs[-1](x, edge_index)

        # Global readout pooling (graph-level aggregation)
        if self.pooling_type == "max":
            x = global_max_pool(x, batch)
        elif self.pooling_type == "sum":
            x = global_add_pool(x, batch)
        else:  # default: "mean"
            x = global_mean_pool(x, batch)

        # Classifier head
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x


# =============================================================================
# C. TRAINING AND EVALUATION FUNCTIONS
# =============================================================================
def train(model, loader, optimizer, device):
    """
    Execute one full training epoch over the training DataLoader.

    Computes cross-entropy loss against ground-truth labels, performs backpropagation,
    updates model weights via Adam optimizer, and tracks classification metrics.

    Args:
        model: GATClassifier instance.
        loader: PyG DataLoader yielding batches of Data graphs.
        optimizer: PyTorch optimizer (Adam).
        device: Target execution device (torch.device('cuda') or torch.device('cpu')).

    Returns:
        Tuple of (average_loss, accuracy, macro_precision, macro_recall, macro_f1).
    """
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

        # Collect predictions for epoch-level metrics
        pred = out.argmax(dim=-1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    return avg_loss, acc, precision, recall, f1


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    Evaluate the model on a validation or test DataLoader without gradient tracking.

    Args:
        model: GATClassifier instance.
        loader: PyG DataLoader yielding batches of Data graphs.
        device: Target execution device.

    Returns:
        Tuple of (average_loss, accuracy, macro_precision, macro_recall, macro_f1, confusion_matrix).
    """
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, precision, recall, f1, cm


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir: Path, is_best: bool = False):
    """
    Persist training checkpoint to disk and maintain rolling window of recent checkpoints.

    Args:
        model: GATClassifier instance.
        optimizer: PyTorch optimizer instance.
        epoch: Current integer epoch index.
        metrics: Dictionary of training and validation metrics.
        checkpoint_dir: Target directory path for saving checkpoints.
        is_best: If True, copies the checkpoint to 'best_model.pt'.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        shutil.copy(checkpoint_path, best_path)
        logger.info(f"Saved best model to {best_path}")

    # Retention policy: keep only the last 3 rolling checkpoints to save disk space
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) > 3:
        for old_checkpoint in checkpoints[:-3]:
            old_checkpoint.unlink()


# =============================================================================
# D. MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    # Load config
    config = load_config(CONFIG_FILE_PATH)

    MODEL_CONFIG = config["model"]
    TRAINING_CONFIG = config["training"]
    DEVICE = torch.device(TRAINING_CONFIG["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # Ensure model directory exists
    model_dir = get_model_path("gnn")
    ensure_dir(model_dir)
    checkpoint_dir = model_dir
    logger.info(f"Model directory: {model_dir}")

    # Load data
    logger.info("Loading datasets...")
    root_dir = PROJECT_ROOT.joinpath("data")

    train_dataset = MalwareGraphDataset(root=root_dir, config=config, split_type="train")
    val_dataset = MalwareGraphDataset(root=root_dir, config=config, split_type="val")

    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG["batch_size"])

    # Initialize model
    model = GATClassifier(
        in_channels=MODEL_CONFIG["input_dim"],
        hidden_channels=MODEL_CONFIG["hidden_dim"],
        out_channels=2,
        num_layers=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        dropout=MODEL_CONFIG["dropout"],
        pooling_type=MODEL_CONFIG.get("pooling_type", "mean"),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )

    # Learning rate scheduler
    lr_scheduler = None
    if TRAINING_CONFIG.get("lr_scheduler", {}).get("enabled", False):
        lr_config = TRAINING_CONFIG["lr_scheduler"]
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=lr_config.get("mode", "min"),
            factor=lr_config.get("factor", 0.5),
            patience=lr_config.get("patience", 5),
            min_lr=lr_config.get("min_lr", 1e-5),
        )
        logger.info("Learning rate scheduler enabled")

    # Early stopping
    early_stopping = None
    if TRAINING_CONFIG.get("early_stopping", {}).get("enabled", False):
        es_config = TRAINING_CONFIG["early_stopping"]
        early_stopping = {
            "patience": es_config.get("patience", 10),
            "min_delta": es_config.get("min_delta", 0.0001),
            "counter": 0,
            "best_score": None,
            "best_epoch": 0,
        }
        logger.info(f"Early stopping enabled with patience={early_stopping['patience']}")

    # Training loop
    logger.info(f"Starting training on {len(train_dataset)} samples...")
    best_val_acc = 0.0
    best_val_loss = float("inf")
    training_history = []

    for epoch in range(1, TRAINING_CONFIG["epochs"] + 1):
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1 = train(
            model, train_loader, optimizer, DEVICE
        )

        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = evaluate(model, val_loader, DEVICE)

        # Update learning rate
        if lr_scheduler:
            lr_scheduler.step(val_loss)

        # Log metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
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
        checkpoint_enabled = TRAINING_CONFIG.get("checkpoint", {}).get("enabled", False)
        save_every = TRAINING_CONFIG.get("checkpoint", {}).get("save_every", 10)

        if checkpoint_enabled and (epoch % save_every == 0 or is_best):
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best)

        # Early stopping check
        if early_stopping:
            if early_stopping["best_score"] is None:
                early_stopping["best_score"] = val_acc
                early_stopping["best_epoch"] = epoch
            elif val_acc > early_stopping["best_score"] + early_stopping["min_delta"]:
                early_stopping["best_score"] = val_acc
                early_stopping["best_epoch"] = epoch
                early_stopping["counter"] = 0
            else:
                early_stopping["counter"] += 1
                if early_stopping["counter"] >= early_stopping["patience"]:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best validation accuracy: {early_stopping['best_score']:.4f} at epoch {early_stopping['best_epoch']}"
                    )
                    break

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
