#!/usr/bin/env python3
"""
Evaluation script for test set evaluation with One-Hot encoding.
Computes comprehensive metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import json
import yaml
from pathlib import Path
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import sys

# Import utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.paths import get_project_root, get_model_path, get_data_path, PROJECT_ROOT
from utils.logger import setup_logger

# Get project roots
DAWNGNN_ONEHOT_ROOT = get_project_root()
from pipelines.train_gnn import GATClassifier, MalwareGraphDataset

logger = setup_logger("evaluate")
CONFIG_FILE_PATH = DAWNGNN_ONEHOT_ROOT.joinpath('configs', 'onehot.yaml')


def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(path, "r", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # Auto-detect input_dim from vocabulary if not set
        if config['model']['input_dim'] is None:
            vocab_file = PROJECT_ROOT / config['paths']['vocabulary_file']
            if vocab_file.exists():
                with open(vocab_file, 'r') as f:
                    vocabulary = json.load(f)
                vocab_size = len(vocabulary)
                config['model']['input_dim'] = vocab_size
                logger.info(f"Auto-detected input_dim={vocab_size} from vocabulary")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def load_model(model_path: Path, config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    MODEL_CONFIG = config['model']
    
    model = GATClassifier(
        in_channels=MODEL_CONFIG['input_dim'],
        hidden_channels=MODEL_CONFIG['hidden_dim'],
        out_channels=2,
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        dropout=MODEL_CONFIG['dropout'],
        pooling_type=MODEL_CONFIG.get('pooling_type', 'mean')
    ).to(device)
    
    # Try to load from checkpoint (if it's a checkpoint file)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # Assume it's just state_dict
        model.load_state_dict(checkpoint)
        logger.info("Loaded model state dict")
    
    model.eval()
    return model


@torch.no_grad()
def evaluate_test_set(model, loader, device) -> Dict[str, Any]:
    """Evaluate model on test set and return comprehensive metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    logger.info("Evaluating on test set...")
    for data in loader:
        if data is None:
            continue
        data = data.to(device)
        out = model(data)
        probs = F.softmax(out, dim=-1)
        pred = out.argmax(dim=-1)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
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
    """Print evaluation metrics in a readable format."""
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    
    print("\n--- Per-Class Metrics ---")
    print(f"Class 0 (Benign):")
    print(f"  Precision: {metrics['precision']['class_0']:.4f}")
    print(f"  Recall:    {metrics['recall']['class_0']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']['class_0']:.4f}")
    print(f"  Support:   {metrics['support']['class_0']}")
    
    print(f"\nClass 1 (Malware):")
    print(f"  Precision: {metrics['precision']['class_1']:.4f}")
    print(f"  Recall:    {metrics['recall']['class_1']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']['class_1']:.4f}")
    print(f"  Support:   {metrics['support']['class_1']}")
    
    print("\n--- Average Metrics ---")
    print(f"Macro Average:")
    print(f"  Precision: {metrics['precision']['macro']:.4f}")
    print(f"  Recall:    {metrics['recall']['macro']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']['macro']:.4f}")
    
    print(f"\nWeighted Average:")
    print(f"  Precision: {metrics['precision']['weighted']:.4f}")
    print(f"  Recall:    {metrics['recall']['weighted']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']['weighted']:.4f}")
    
    if metrics['auc_roc'] is not None:
        print(f"\nAUC-ROC: {metrics['auc_roc']:.4f}")
    
    print("\n--- Confusion Matrix ---")
    cm = metrics['confusion_matrix']
    print(f"                Predicted")
    print(f"              Benign  Malware")
    print(f"Actual Benign   {cm['tn']:4d}    {cm['fp']:4d}")
    print(f"      Malware   {cm['fn']:4d}    {cm['tp']:4d}")
    
    print("\n" + "="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to model checkpoint (default: models/gnn/best_model.pt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save evaluation results JSON (default: models/gnn/test_evaluation.json)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(CONFIG_FILE_PATH)
    TRAINING_CONFIG = config['training']
    DEVICE = torch.device(TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")
    
    # Model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = get_model_path('gnn', 'best_model.pt')
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path, config, DEVICE)
    
    # Load test dataset
    logger.info("Loading test dataset...")
    root_dir = PROJECT_ROOT.joinpath('data')
    test_dataset = MalwareGraphDataset(root=root_dir, config=config, split_type='test')
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    
    logger.info(f"Test set size: {len(test_dataset)}")
    
    # Evaluate
    metrics = evaluate_test_set(model, test_loader, DEVICE)
    
    # Print results
    print_metrics(metrics)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = get_model_path('gnn', 'test_evaluation.json')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add method info to metrics
    metrics['method'] = 'DawnGNN-OneHot'
    metrics['model_name'] = 'GAT Classifier'
    metrics['description'] = 'GAT + One-Hot Encoding'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation results saved to {output_path}")
    
    return metrics


if __name__ == "__main__":
    main()

