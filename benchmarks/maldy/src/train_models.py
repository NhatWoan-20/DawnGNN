#!/usr/bin/env python3
"""
Train Ensemble Learning model for MalDy malware detection.
Uses VotingClassifier or RandomForestClassifier as specified in config.
"""

import json
import yaml
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any
import sys

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MALDY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MALDY_ROOT / "src"))

from utils import get_model_path, ensure_dir, PROJECT_ROOT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maldy_train")

if not XGBOOST_AVAILABLE:
    logger.warning("XGBoost not available. Install with: pip install xgboost")


def create_base_models(config: Dict) -> list:
    """
    Create base models for ensemble.
    
    Returns:
        List of (name, model) tuples
    """
    estimators = []
    
    # Random Forest
    if config['models']['random_forest']['enabled']:
        rf_config = config['models']['random_forest']
        rf_kwargs = {
            'n_estimators': rf_config['n_estimators'],
            'max_depth': rf_config.get('max_depth', None),
            'min_samples_split': rf_config.get('min_samples_split', 2),
            'min_samples_leaf': rf_config.get('min_samples_leaf', 1),
            'random_state': rf_config['random_state'],
            'n_jobs': rf_config.get('n_jobs', -1)
        }
        # Add class_weight if specified
        if 'class_weight' in rf_config:
            rf_kwargs['class_weight'] = rf_config['class_weight']
        rf = RandomForestClassifier(**rf_kwargs)
        estimators.append(('rf', rf))
        logger.info("Added Random Forest to ensemble")
    
    # XGBoost (optional)
    if config['models'].get('xgboost', {}).get('enabled', False) and XGBOOST_AVAILABLE:
        xgb_config = config['models']['xgboost']
        xgb_kwargs = {
            'n_estimators': xgb_config['n_estimators'],
            'max_depth': xgb_config['max_depth'],
            'learning_rate': xgb_config['learning_rate'],
            'subsample': xgb_config.get('subsample', 0.8),
            'colsample_bytree': xgb_config.get('colsample_bytree', 0.8),
            'random_state': xgb_config['random_state'],
            'n_jobs': xgb_config.get('n_jobs', -1),
            'eval_metric': 'logloss'
        }
        # Add scale_pos_weight if specified
        if 'scale_pos_weight' in xgb_config:
            xgb_kwargs['scale_pos_weight'] = xgb_config['scale_pos_weight']
        xgb_model = xgb.XGBClassifier(**xgb_kwargs)
        estimators.append(('xgboost', xgb_model))
        logger.info("Added XGBoost to ensemble")
    elif config['models'].get('xgboost', {}).get('enabled', False) and not XGBOOST_AVAILABLE:
        logger.warning("XGBoost enabled but not available, skipping...")
    
    # SVM (optional)
    if config['models'].get('svm', {}).get('enabled', False):
        svm_config = config['models']['svm']
        svm_kwargs = {
            'kernel': svm_config['kernel'],
            'C': svm_config['C'],
            'gamma': svm_config['gamma'],
            'probability': svm_config['probability'],
            'random_state': svm_config['random_state']
        }
        # Add class_weight if specified
        if 'class_weight' in svm_config:
            svm_kwargs['class_weight'] = svm_config['class_weight']
        svm = SVC(**svm_kwargs)
        estimators.append(('svm', svm))
        logger.info("Added SVM to ensemble")
    
    return estimators


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict
) -> Dict[str, Any]:
    """
    Train ensemble model (VotingClassifier or RandomForestClassifier).
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary
    
    Returns:
        Dictionary with model and metrics
    """
    ensemble_config = config.get('ensemble', {})
    ensemble_type = ensemble_config.get('type', 'voting')  # 'voting' or 'random_forest'
    voting_method = ensemble_config.get('voting', 'soft')  # 'hard' or 'soft'
    
    # Create base models
    estimators = create_base_models(config)
    
    if len(estimators) == 0:
        raise ValueError("No base models enabled. Please enable at least one model.")
    
    # Create ensemble
    if ensemble_type == 'random_forest':
        # Use RandomForestClassifier directly (it's already an ensemble)
        if len(estimators) > 1:
            logger.warning("Multiple models specified but using RandomForest only. Consider using 'voting' type.")
        
        rf_config = config['models']['random_forest']
        rf_kwargs = {
            'n_estimators': rf_config['n_estimators'],
            'max_depth': rf_config.get('max_depth', None),
            'min_samples_split': rf_config.get('min_samples_split', 2),
            'min_samples_leaf': rf_config.get('min_samples_leaf', 1),
            'random_state': rf_config['random_state'],
            'n_jobs': rf_config.get('n_jobs', -1)
        }
        if 'class_weight' in rf_config:
            rf_kwargs['class_weight'] = rf_config['class_weight']
        
        ensemble = RandomForestClassifier(**rf_kwargs)
        logger.info(f"\nTraining RandomForestClassifier with {rf_kwargs['n_estimators']} estimators...")
        
    else:
        # VotingClassifier
        if len(estimators) < 2:
            logger.error(f"VotingClassifier needs at least 2 models, but only {len(estimators)} enabled.")
            logger.error("Please enable at least 2 models in config (e.g., random_forest + xgboost)")
            raise ValueError(f"Need at least 2 models for VotingClassifier. Found: {len(estimators)}")
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting_method,
            n_jobs=-1
        )
        logger.info(f"\nTraining VotingClassifier with {len(estimators)} base models...")
        logger.info(f"Base models: {[name for name, _ in estimators]}")
        logger.info(f"Voting method: {voting_method}")
    
    # Train ensemble
    logger.info("Training ensemble model...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate on validation
    logger.info("Evaluating on validation set...")
    y_pred = ensemble.predict(X_val)
    y_proba = ensemble.predict_proba(X_val)[:, 1] if hasattr(ensemble, 'predict_proba') else None
    
    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)
    
    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_val, y_proba)
        except:
            auc = None
    
    if auc is not None:
        logger.info(f"Ensemble - Val Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    else:
        logger.info(f"Ensemble - Val Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: N/A")
    
    return {
        'model': ensemble,
        'val_accuracy': acc,
        'val_f1': f1,
        'val_auc': auc if auc else 0.0,
        'ensemble_type': ensemble_type,
        'base_models': [name for name, _ in estimators] if ensemble_type == 'voting' else ['random_forest']
    }


def main():
    """Main training function."""
    config_path = MALDY_ROOT / "configs" / "maldy.yaml"
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Paths
    features_dir = PROJECT_ROOT / config['paths']['features_dir']
    model_dir = get_model_path()
    ensure_dir(model_dir)
    
    # Load features
    logger.info("Loading features...")
    X_train = np.load(features_dir / 'train_features.npy')
    X_val = np.load(features_dir / 'val_features.npy')
    
    # Load labels
    processed_dir = PROJECT_ROOT / config['paths']['processed_data_dir']
    with open(processed_dir / "labels.json", 'r') as f:
        all_labels = json.load(f)
    
    splits_dir = PROJECT_ROOT / config['paths']['splits_dir']
    with open(splits_dir / "train.json", 'r') as f:
        train_shas = json.load(f)
    with open(splits_dir / "val.json", 'r') as f:
        val_shas = json.load(f)
    
    # Filter: only keep samples with valid labels (0 or 1) and align with features
    train_valid = [(sha, all_labels[sha]) for sha in train_shas 
                   if sha in all_labels and all_labels[sha] in [0, 1]]
    val_valid = [(sha, all_labels[sha]) for sha in val_shas 
                 if sha in all_labels and all_labels[sha] in [0, 1]]
    
    # Ensure features and labels are aligned
    if len(train_valid) != X_train.shape[0]:
        logger.warning(f"Train features ({X_train.shape[0]}) and labels ({len(train_valid)}) don't match!")
        logger.warning("Filtering to match feature count...")
        train_valid = train_valid[:X_train.shape[0]]
    
    if len(val_valid) != X_val.shape[0]:
        logger.warning(f"Val features ({X_val.shape[0]}) and labels ({len(val_valid)}) don't match!")
        logger.warning("Filtering to match feature count...")
        val_valid = val_valid[:X_val.shape[0]]
    
    # Extract labels
    y_train = np.array([label for _, label in train_valid])
    y_val = np.array([label for _, label in val_valid])
    
    # Update features to match filtered labels
    X_train = X_train[:len(y_train)]
    X_val = X_val[:len(y_val)]
    
    # Log statistics
    train_label_0 = (y_train == 0).sum()
    train_label_1 = (y_train == 1).sum()
    val_label_0 = (y_val == 0).sum()
    val_label_1 = (y_val == 1).sum()
    
    logger.info(f"Train: {X_train.shape}, Labels: {len(y_train)} (0: {train_label_0}, 1: {train_label_1})")
    logger.info(f"Val: {X_val.shape}, Labels: {len(y_val)} (0: {val_label_0}, 1: {val_label_1})")
    
    # Check for class imbalance
    if train_label_0 == 0 or train_label_1 == 0:
        logger.error("ERROR: Only one class in training data!")
        raise ValueError("Training data must contain both classes (0 and 1)")
    
    train_imbalance = abs(train_label_0 - train_label_1) / len(y_train)
    if train_imbalance > 0.3:
        logger.warning(f"Class imbalance detected: {train_imbalance:.2%} difference")
    
    # Train ensemble
    try:
        result = train_ensemble(X_train, y_train, X_val, y_val, config)
        
        # Save ensemble model
        ensemble_path = model_dir / 'ensemble_model.pkl'
        joblib.dump(result['model'], ensemble_path)
        logger.info(f"Saved ensemble model to: {ensemble_path}")
        
        # Save training results
        results_path = model_dir / 'training_results.json'
        json_results = {
            'ensemble': {
                'val_accuracy': float(result['val_accuracy']),
                'val_f1': float(result['val_f1']),
                'val_f1_macro': float(result['val_f1']),
                'val_auc': float(result['val_auc']),
                'ensemble_type': result['ensemble_type'],
                'base_models': result['base_models']
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Saved training results to: {results_path}")
        logger.info("\n[+] Training completed!")
        
    except Exception as e:
        logger.error(f"Error training ensemble: {e}")
        raise


if __name__ == "__main__":
    main()
