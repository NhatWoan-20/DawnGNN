#!/usr/bin/env python3
"""
Preprocess MalBehavD-V1 dataset: Convert CSV to API sequences with padding/truncation.
"""

import argparse
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys

# Add project root to path
MALDETCONV_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MALDETCONV_ROOT / "src"))

from utils import get_project_root, get_data_path, get_model_path, PROJECT_ROOT
import yaml
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maldetconv_preprocess")


def normalize_api(name: str) -> str:
    """Normalize API names."""
    if not isinstance(name, str):
        return ""
    return name.strip()


def extract_api_sequence(row: pd.Series, api_cols: List[str]) -> List[str]:
    """Extract API sequence from a row."""
    api_calls = []
    for col in api_cols:
        value = row[col]
        if pd.isna(value):
            continue
        api = normalize_api(str(value))
        if api and api.lower() not in ("nan", "none", ""):
            api_calls.append(api)
    return api_calls


def pad_or_truncate(sequence: List[str], max_length: int, pad_token: str = "<PAD>") -> List[str]:
    """Pad or truncate sequence to fixed length."""
    if len(sequence) > max_length:
        return sequence[:max_length]
    elif len(sequence) < max_length:
        return sequence + [pad_token] * (max_length - len(sequence))
    return sequence


def create_splits(samples: List[str], train_frac: float, val_frac: float, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Create train/val/test splits."""
    random.seed(seed)
    samples = list(samples)
    random.shuffle(samples)
    
    n = len(samples)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    
    return train, val, test


def main(args):
    """Main preprocessing function."""
    config_path = MALDETCONV_ROOT / "configs" / "cnn_bigru.yaml"
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Paths
    csv_path = Path(args.csv) if args.csv else PROJECT_ROOT / config['paths']['raw_data']
    output_dir = PROJECT_ROOT / config['paths']['processed_data_dir']
    splits_dir = PROJECT_ROOT / config['paths']['splits_dir']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Data config
    data_config = config['data']
    sha_col = data_config['sha_col']
    label_col = data_config['label_col']
    api_start_col = data_config['api_start_col']
    max_length = data_config['max_sequence_length']
    
    logger.info(f"Loading CSV from: {csv_path}")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise
    
    df.fillna('', inplace=True)
    
    if sha_col not in df.columns:
        raise ValueError(f"Column '{sha_col}' not found. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. Available: {list(df.columns)}")
    
    # Extract API columns
    api_cols = list(df.columns)[api_start_col:]
    
    logger.info(f"Processing {len(df)} samples...")
    logger.info(f"API columns: {len(api_cols)}")
    logger.info(f"Max sequence length: {max_length}")
    
    # Process samples
    sequences = {}
    labels = {}
    sequence_lengths = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sha = str(row[sha_col]).strip()
        if not sha:
            continue
        
        # Parse label
        try:
            label = int(row[label_col])
        except (ValueError, TypeError):
            label = -1
        
        # Extract API sequence
        api_sequence = extract_api_sequence(row, api_cols)
        
        if len(api_sequence) == 0:
            continue
        
        # Pad/truncate
        padded_sequence = pad_or_truncate(api_sequence, max_length)
        
        sequences[sha] = padded_sequence
        labels[sha] = label
        sequence_lengths.append(len(api_sequence))  # Original length before padding
    
    logger.info(f"Processed {len(sequences)} samples")
    logger.info(f"Sequence length stats: min={min(sequence_lengths)}, max={max(sequence_lengths)}, mean={np.mean(sequence_lengths):.1f}")
    
    # Save sequences and labels
    sequences_path = output_dir / "api_sequences.json"
    with open(sequences_path, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved sequences to: {sequences_path}")
    
    labels_path = output_dir / "labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    logger.info(f"Saved labels to: {labels_path}")
    
    # Create splits
    if args.make_splits:
        train_shas, val_shas, test_shas = create_splits(
            list(sequences.keys()),
            data_config['train_frac'],
            data_config['val_frac'],
            data_config['seed']
        )
        
        with open(splits_dir / "train.json", 'w') as f:
            json.dump(train_shas, f, indent=2)
        with open(splits_dir / "val.json", 'w') as f:
            json.dump(val_shas, f, indent=2)
        with open(splits_dir / "test.json", 'w') as f:
            json.dump(test_shas, f, indent=2)
        
        logger.info(f"Created splits: train={len(train_shas)}, val={len(val_shas)}, test={len(test_shas)}")
    
    # Statistics
    label_counts = {}
    for label in labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info("Label distribution:")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  Label {label}: {count} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MalBehavD-V1 dataset")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    parser.add_argument("--make-splits", action="store_true", help="Create train/val/test splits")
    
    args = parser.parse_args()
    main(args)

