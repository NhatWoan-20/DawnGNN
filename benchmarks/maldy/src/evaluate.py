#!/usr/bin/env python3
"""
Evaluate trained ensemble model on test set.
Calculates and prints: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
"""

import json
import yaml
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any
import sys

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
    classification_report
)

MALDY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MALDY_ROOT / "src"))

from utils import get_model_path, ensure_dir, PROJECT_ROOT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maldy_evaluate")


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """
    Evaluate a model and return comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name for logging
    
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # AUC-ROC
    auc_roc = None
    if y_proba is not None:
        try:
            auc_roc = roc_auc_score(y_test, y_proba)
        except:
            pass
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
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
            'total': int(len(y_test))
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
    
    # Print metrics
    logger.info("\n" + "=" * 60)
    logger.info(f"EVALUATION RESULTS: {model_name}")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision_macro:.4f} (macro)")
    logger.info(f"Recall:    {recall_macro:.4f} (macro)")
    logger.info(f"F1-Score:  {f1_macro:.4f} (macro)")
    if auc_roc is not None:
        logger.info(f"AUC-ROC:   {auc_roc:.4f}")
    else:
        logger.info("AUC-ROC:   N/A")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"                Predicted")
    logger.info(f"              Benign  Malware")
    logger.info(f"Actual Benign    {tn:4d}    {fp:4d}")
    logger.info(f"      Malware    {fn:4d}    {tp:4d}")
    
    logger.info("\nPer-Class Metrics:")
    logger.info(f"Class 0 (Benign):")
    logger.info(f"  Precision: {precision[0]:.4f}")
    logger.info(f"  Recall:    {recall[0]:.4f}")
    logger.info(f"  F1-Score:  {f1[0]:.4f}")
    logger.info(f"  Support:   {support[0]}")
    
    logger.info(f"Class 1 (Malware):")
    logger.info(f"  Precision: {precision[1]:.4f}")
    logger.info(f"  Recall:    {recall[1]:.4f}")
    logger.info(f"  F1-Score:  {f1[1]:.4f}")
    logger.info(f"  Support:   {support[1]}")
    logger.info("=" * 60)
    
    return metrics


def main():
    """Main evaluation function."""
    config_path = MALDY_ROOT / "configs" / "maldy.yaml"
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Paths
    features_dir = PROJECT_ROOT / config['paths']['features_dir']
    model_dir = get_model_path()
    ensure_dir(model_dir)
    
    # Load test features
    logger.info("Loading test features...")
    X_test = np.load(features_dir / 'test_features.npy')
    
    # Load test labels
    processed_dir = PROJECT_ROOT / config['paths']['processed_data_dir']
    with open(processed_dir / "labels.json", 'r') as f:
        all_labels = json.load(f)
    
    splits_dir = PROJECT_ROOT / config['paths']['splits_dir']
    with open(splits_dir / "test.json", 'r') as f:
        test_shas = json.load(f)
    
    # Filter: only keep samples with valid labels (0 or 1) and align with features
    test_valid = [(sha, all_labels[sha]) for sha in test_shas 
                  if sha in all_labels and all_labels[sha] in [0, 1]]
    
    # Ensure features and labels are aligned
    if len(test_valid) != X_test.shape[0]:
        logger.warning(f"Test features ({X_test.shape[0]}) and labels ({len(test_valid)}) don't match!")
        logger.warning("Filtering to match feature count...")
        test_valid = test_valid[:X_test.shape[0]]
    
    # Extract labels
    y_test = np.array([label for _, label in test_valid])
    
    # Update features to match filtered labels
    X_test = X_test[:len(y_test)]
    
    # Log statistics
    test_label_0 = (y_test == 0).sum()
    test_label_1 = (y_test == 1).sum()
    logger.info(f"Test: {X_test.shape}, Labels: {len(y_test)} (0: {test_label_0}, 1: {test_label_1})")
    
    # Load ensemble model
    ensemble_path = model_dir / 'ensemble_model.pkl'
    if not ensemble_path.exists():
        logger.error(f"Ensemble model not found at {ensemble_path}. Please run train_models.py first.")
        return
    
    logger.info(f"Loading ensemble model from: {ensemble_path}")
    ensemble = joblib.load(ensemble_path)
    
    # Get ensemble info from training results
    training_results_path = model_dir / 'training_results.json'
    ensemble_info = {}
    if training_results_path.exists():
        with open(training_results_path, 'r') as f:
            training_results = json.load(f)
        ensemble_info = training_results.get('ensemble', {})
    
    # Determine ensemble type and name
    ensemble_type = ensemble_info.get('ensemble_type', 'voting')
    base_models = ensemble_info.get('base_models', [])
    
    if ensemble_type == 'random_forest':
        ensemble_name = "RandomForestClassifier"
    else:
        voting_method = config.get('ensemble', {}).get('voting', 'soft')
        ensemble_name = f"VotingClassifier ({voting_method}, {', '.join(base_models)})"
    
    # Evaluate ensemble
    metrics = evaluate_model(ensemble, X_test, y_test, ensemble_name)
    
    # Add metadata
    metrics['method'] = 'MalDy'
    metrics['model_name'] = ensemble_name
    metrics['ensemble_type'] = ensemble_type
    metrics['base_models'] = base_models
    
    # Save results
    results_path = model_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump({'ensemble': metrics}, f, indent=2)
    logger.info(f"\nSaved evaluation results to: {results_path}")
    
    # Save standardized results
    best_model_path = model_dir / 'test_evaluation.json'
    with open(best_model_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved standardized results to: {best_model_path}")
    
    logger.info("\n[+] Evaluation completed!")


if __name__ == "__main__":
    main()
