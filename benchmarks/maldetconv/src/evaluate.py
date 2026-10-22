#!/usr/bin/env python3
"""
Evaluate CNN+BiGRU model on test set.
"""

import torch
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
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report
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
from train_cnn_bigru import collate_fn

logger = setup_logger("maldetconv_evaluate")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_test_set(model, loader, device) -> Dict[str, Any]:
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(sequences)
        probs = F.softmax(outputs, dim=-1)
        preds = outputs.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (malware)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    except ValueError as e:
        logger.warning(f"Could not calculate AUC-ROC: {e}")
        auc_roc = None
        fpr = tpr = thresholds = None
    
    # Classification report
    class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': {
            'class_0': float(precision[0]) if len(precision) > 0 else 0.0,
            'class_1': float(precision[1]) if len(precision) > 1 else 0.0,
            'macro': float(precision_macro),
            'weighted': float(precision_weighted)
        },
        'recall': {
            'class_0': float(recall[0]) if len(recall) > 0 else 0.0,
            'class_1': float(recall[1]) if len(recall) > 1 else 0.0,
            'macro': float(recall_macro),
            'weighted': float(recall_weighted)
        },
        'f1_score': {
            'class_0': float(f1[0]) if len(f1) > 0 else 0.0,
            'class_1': float(f1[1]) if len(f1) > 1 else 0.0,
            'macro': float(f1_macro),
            'weighted': float(f1_weighted)
        },
        'support': {
            'class_0': int(support[0]) if len(support) > 0 else 0,
            'class_1': int(support[1]) if len(support) > 1 else 0,
            'total': int(len(all_labels))
        },
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'matrix': cm.tolist()
        },
        'auc_roc': float(auc_roc) if auc_roc is not None else None,
        'classification_report': class_report
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """Print evaluation metrics."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    auc_roc = metrics.get('auc_roc')
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.4f}")
    else:
        print("AUC-ROC: N/A")
    
    print("\nPer-Class Metrics:")
    print(f"  Class 0 (Benign):")
    print(f"    Precision: {metrics['precision']['class_0']:.4f}")
    print(f"    Recall: {metrics['recall']['class_0']:.4f}")
    print(f"    F1-Score: {metrics['f1_score']['class_0']:.4f}")
    print(f"    Support: {metrics['support']['class_0']}")
    
    print(f"  Class 1 (Malware):")
    print(f"    Precision: {metrics['precision']['class_1']:.4f}")
    print(f"    Recall: {metrics['recall']['class_1']:.4f}")
    print(f"    F1-Score: {metrics['f1_score']['class_1']:.4f}")
    print(f"    Support: {metrics['support']['class_1']}")
    
    print("\nMacro Averages:")
    print(f"  Precision: {metrics['precision']['macro']:.4f}")
    print(f"  Recall: {metrics['recall']['macro']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']['macro']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm['tn']}, FP: {cm['fp']}")
    print(f"  FN: {cm['fn']}, TP: {cm['tp']}")
    print(f"\n  [[{cm['tn']:4d} {cm['fp']:4d}]")
    print(f"   [{cm['fn']:4d} {cm['tp']:4d}]]")
    
    print("\n" + "=" * 60)


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CNN+BiGRU model on test set")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    
    args = parser.parse_args()
    
    config_path = get_config_path("cnn_bigru.yaml")
    config = load_config(config_path)
    
    # Paths - use utils functions to ensure paths are in maldetconv directory
    paths = config['paths']
    processed_dir = get_data_path("maldetconv", "processed")
    splits_dir = get_data_path("splits")
    model_dir = get_model_path("maldetconv")  # maldetconv/models/maldetconv
    
    # Model path
    model_path = Path(args.model) if args.model else model_dir / "best_model.pt"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['config']['model']
    
    # Load Word2Vec embeddings from maldetconv/data/ directory
    word_to_idx_path = get_local_data_path("word_to_idx.json")
    embedding_matrix_path = get_local_data_path("word2vec_vectors.npy")
    
    with open(word_to_idx_path, 'r', encoding='utf-8') as f:
        word_to_idx = json.load(f)
    
    # vocab_size = len(word_to_idx) + 2  # +1 for padding (idx 0), +1 for unknown
    max_idx = max(word_to_idx.values()) if word_to_idx else 0
    vocab_size = max_idx + 2  # +1 for padding (idx 0), +1 for unknown (max_idx+1)
    
    embedding_matrix = np.load(embedding_matrix_path)
    unk_vector = np.zeros((1, embedding_matrix.shape[1]), dtype=np.float32)
    embedding_matrix = np.vstack([embedding_matrix, unk_vector])
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Test dataset
    sequences_path = processed_dir / "api_sequences.json"
    labels_path = processed_dir / "labels.json"
    
    test_dataset = APISequenceDataset(
        sequences_path, labels_path, word_to_idx_path,
        split_path=splits_dir / "test.json"
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    metrics = evaluate_test_set(model, test_loader, device)
    
    # Print
    print_metrics(metrics)
    
    # Save results
    output_path = Path(args.output) if args.output else model_dir / "test_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add method info to metrics
    metrics['method'] = 'MalDetConv'
    metrics['model_name'] = 'CNN+BiGRU Classifier'
    metrics['description'] = 'CNN + BiGRU + Word2Vec Embeddings (Sequence-based)'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

