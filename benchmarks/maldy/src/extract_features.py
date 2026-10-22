#!/usr/bin/env python3
"""
Extract features from API sequences using N-grams, Feature Hashing, and TF-IDF.
This follows the MalDy methodology: N-grams -> HashingVectorizer -> TfidfTransformer
"""

import json
import yaml
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List
import sys

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

MALDY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MALDY_ROOT / "src"))

from utils import get_data_path, PROJECT_ROOT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maldy_features")


def sequences_to_text(sequences: Dict[str, List[str]]) -> List[str]:
    """Convert API sequences to text format (space-separated)."""
    texts = []
    for sha, seq in sequences.items():
        texts.append(' '.join(seq))
    return texts


def create_feature_pipeline(config: Dict):
    """
    Create sklearn Pipeline: HashingVectorizer -> TfidfTransformer
    
    Args:
        config: Configuration dictionary
    
    Returns:
        sklearn Pipeline
    """
    feat_config = config['features']
    
    # Step 1: Feature Hashing (HashingVectorizer)
    # This reduces dimensionality and handles large vocabularies
    hashing_vectorizer = HashingVectorizer(
        n_features=feat_config['n_features'],
        ngram_range=tuple(feat_config['ngram_range']),
        lowercase=False,
        alternate_sign=False  # Use positive values only
    )
    
    # Step 2: TF-IDF Transformation
    # This weights the hashed features by TF-IDF
    tfidf_transformer = TfidfTransformer(
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False
    )
    
    # Create pipeline: Hashing -> TF-IDF
    pipeline = Pipeline([
        ('hashing', hashing_vectorizer),
        ('tfidf', tfidf_transformer)
    ])
    
    return pipeline


def extract_features(sequences: Dict[str, List[str]], config: Dict, split_name: str = 'train'):
    """
    Extract features using N-grams + Hashing + TF-IDF pipeline.
    
    Args:
        sequences: Dictionary of {sha: api_sequence}
        config: Configuration dictionary
        split_name: 'train', 'val', or 'test'
    
    Returns:
        Feature matrix (sparse or dense)
    """
    logger.info(f"Extracting features for {split_name}...")
    
    # Convert sequences to text
    texts = sequences_to_text(sequences)
    logger.info(f"  {split_name}: {len(texts)} samples")
    
    # Create or load pipeline
    features_dir = PROJECT_ROOT / config['paths']['features_dir']
    pipeline_path = features_dir / 'feature_pipeline.pkl'
    
    if split_name == 'train':
        # Create and fit pipeline
        pipeline = create_feature_pipeline(config)
        features = pipeline.fit_transform(texts)
        
        # Save pipeline for later use
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, pipeline_path)
        logger.info(f"  Saved feature pipeline to: {pipeline_path}")
    else:
        # Load pipeline and transform
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}. Please run on train split first.")
        pipeline = joblib.load(pipeline_path)
        features = pipeline.transform(texts)
    
    logger.info(f"  Features shape: {features.shape}")
    logger.info(f"  Feature type: {type(features).__name__}")
    
    return features, pipeline


def main():
    """Main feature extraction function."""
    config_path = MALDY_ROOT / "configs" / "maldy.yaml"
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Paths
    processed_dir = PROJECT_ROOT / config['paths']['processed_data_dir']
    features_dir = PROJECT_ROOT / config['paths']['features_dir']
    splits_dir = PROJECT_ROOT / config['paths']['splits_dir']
    
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sequences
    sequences_path = processed_dir / "api_sequences.json"
    with open(sequences_path, 'r', encoding='utf-8') as f:
        all_sequences = json.load(f)
    
    # Load labels to filter valid samples (only 0 or 1)
    labels_path = processed_dir / "labels.json"
    with open(labels_path, 'r') as f:
        all_labels = json.load(f)
    
    # Load splits
    with open(splits_dir / "train.json", 'r') as f:
        train_shas = set(json.load(f))
    with open(splits_dir / "val.json", 'r') as f:
        val_shas = set(json.load(f))
    with open(splits_dir / "test.json", 'r') as f:
        test_shas = set(json.load(f))
    
    # Split sequences - only keep samples with valid labels (0 or 1)
    train_sequences = {sha: all_sequences[sha] for sha in train_shas 
                       if sha in all_sequences and sha in all_labels 
                       and all_labels[sha] in [0, 1]}
    val_sequences = {sha: all_sequences[sha] for sha in val_shas 
                     if sha in all_sequences and sha in all_labels 
                     and all_labels[sha] in [0, 1]}
    test_sequences = {sha: all_sequences[sha] for sha in test_shas 
                     if sha in all_sequences and sha in all_labels 
                     and all_labels[sha] in [0, 1]}
    
    logger.info(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
    
    # Log label distribution
    train_labels = [all_labels[sha] for sha in train_sequences.keys()]
    val_labels = [all_labels[sha] for sha in val_sequences.keys()]
    test_labels = [all_labels[sha] for sha in test_sequences.keys()]
    
    logger.info(f"Train labels - 0: {train_labels.count(0)}, 1: {train_labels.count(1)}")
    logger.info(f"Val labels - 0: {val_labels.count(0)}, 1: {val_labels.count(1)}")
    logger.info(f"Test labels - 0: {test_labels.count(0)}, 1: {test_labels.count(1)}")
    
    # Extract features
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE EXTRACTION PIPELINE")
    logger.info("=" * 60)
    logger.info("Step 1: N-grams (from sequences)")
    logger.info("Step 2: Feature Hashing (HashingVectorizer)")
    logger.info("Step 3: TF-IDF Transformation (TfidfTransformer)")
    logger.info("=" * 60 + "\n")
    
    # Train
    train_features, pipeline = extract_features(train_sequences, config, 'train')
    
    # Val
    val_features, _ = extract_features(val_sequences, config, 'val')
    
    # Test
    test_features, _ = extract_features(test_sequences, config, 'test')
    
    # Save features (convert sparse to dense for compatibility)
    np.save(features_dir / 'train_features.npy', train_features.toarray() if hasattr(train_features, 'toarray') else train_features)
    np.save(features_dir / 'val_features.npy', val_features.toarray() if hasattr(val_features, 'toarray') else val_features)
    np.save(features_dir / 'test_features.npy', test_features.toarray() if hasattr(test_features, 'toarray') else test_features)
    
    logger.info(f"\nSaved features to: {features_dir}")
    logger.info("[+] Feature extraction completed!")


if __name__ == "__main__":
    main()
