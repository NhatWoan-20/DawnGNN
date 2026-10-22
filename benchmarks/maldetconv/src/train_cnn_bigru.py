#!/usr/bin/env python3
"""
Train CNN+BiGRU model for malware detection.
Training setup similar to DawnGNN (split data, optimizer, epochs, early stopping).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
import sys
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

MALDETCONV_SRC = Path(__file__).resolve().parent
MALDETCONV_ROOT = MALDETCONV_SRC.parent

# Import local utils directly from file to avoid conflict with project root utils package
import importlib.util
local_utils_spec = importlib.util.spec_from_file_location("maldetconv_utils", MALDETCONV_SRC / "utils.py")
local_utils = importlib.util.module_from_spec(local_utils_spec)
local_utils_spec.loader.exec_module(local_utils)

PROJECT_ROOT = local_utils.PROJECT_ROOT

# Import logger from project root utils
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils.logger import setup_logger

# Use local utils functions
get_project_root = local_utils.get_project_root
get_model_path = local_utils.get_model_path
get_data_path = local_utils.get_data_path
get_config_path = local_utils.get_config_path
get_local_data_path = local_utils.get_local_data_path
from model import CNNBiGRUClassifier
from dataset import APISequenceDataset

logger = setup_logger("maldetconv_train")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def collate_fn(batch):
    """Custom collate function for batching sequences."""
    sequences = [item['sequence'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    shas = [item['sha'] for item in batch]
    
    # Pad sequences to same length
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    
    return {
        'sequence': sequences,
        'label': labels,
        'sha': shas
    }


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Training"):
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = outputs.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return avg_loss, acc, precision, recall, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        preds = outputs.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, acc, precision, recall, f1, cm


def main():
    """Main training function."""
    config_path = get_config_path("cnn_bigru.yaml")
    config = load_config(config_path)
    
    # Paths - use utils functions to ensure paths are in maldetconv directory
    paths = config['paths']
    processed_dir = get_data_path("maldetconv", "processed")
    splits_dir = get_data_path("splits")
    model_dir = get_model_path("maldetconv")  # maldetconv/models/maldetconv
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Word2Vec embeddings from maldetconv/data/ directory
    word_to_idx_path = get_local_data_path("word_to_idx.json")
    embedding_matrix_path = get_local_data_path("word2vec_vectors.npy")
    
    with open(word_to_idx_path, 'r', encoding='utf-8') as f:
        word_to_idx = json.load(f)
    
    # vocab_size = len(word_to_idx) + 2  # +1 for padding (idx 0), +1 for unknown
    # Padding token is at index 0, so vocab_size = max_idx + 1
    max_idx = max(word_to_idx.values()) if word_to_idx else 0
    vocab_size = max_idx + 2  # +1 for padding (idx 0), +1 for unknown (max_idx+1)
    
    embedding_matrix = np.load(embedding_matrix_path)
    
    # Embedding matrix already has padding token at index 0
    # Add zero vector for unknown token
    unk_vector = np.zeros((1, embedding_matrix.shape[1]), dtype=np.float32)
    embedding_matrix = np.vstack([embedding_matrix, unk_vector])
    
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")
    
    # Device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Datasets
    sequences_path = processed_dir / "api_sequences.json"
    labels_path = processed_dir / "labels.json"
    
    train_dataset = APISequenceDataset(
        sequences_path, labels_path, word_to_idx_path,
        split_path=splits_dir / "train.json"
    )
    val_dataset = APISequenceDataset(
        sequences_path, labels_path, word_to_idx_path,
        split_path=splits_dir / "val.json"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Model
    model_config = config['model']
    model = CNNBiGRUClassifier(
        vocab_size=vocab_size,
        embedding_dim=model_config['embedding_dim'],
        num_filters=model_config['num_filters'],
        filter_sizes=model_config['filter_sizes'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        num_classes=model_config['num_classes'],
        embedding_matrix=embedding_matrix
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    training_config = config['training']
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = None
    if training_config['lr_scheduler']['enabled']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=training_config['lr_scheduler']['mode'],
            factor=training_config['lr_scheduler']['factor'],
            patience=training_config['lr_scheduler']['patience'],
            min_lr=training_config['lr_scheduler']['min_lr']
        )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    epochs = training_config['epochs']
    early_stopping = training_config['early_stopping']
    
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step(val_loss)
        
        # Log
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        logger.info(f"Confusion Matrix:\n{val_cm}")
        
        # History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config
            }
            torch.save(checkpoint, model_dir / "best_model.pt")
            logger.info("[+] Saved best model")
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stopping['enabled'] and patience_counter >= early_stopping['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save training history
    history_path = model_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to: {history_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Completed!")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best Val Acc: {best_val_acc:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

