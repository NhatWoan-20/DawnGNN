#!/usr/bin/env python3
"""
Preprocess MalBehavD-V1 dataset: Convert CSV to API sequences.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import sys

MALDY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MALDY_ROOT / "src"))

from utils import get_project_root, get_data_path, get_model_path, PROJECT_ROOT
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maldy_preprocess")


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


def main(args):
    """Main preprocessing function."""
    config_path = MALDY_ROOT / "configs" / "maldy.yaml"
    
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
    
    # Process samples
    sequences = {}
    labels = {}
    
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
        
        sequences[sha] = api_sequence
        labels[sha] = label
    
    logger.info(f"Processed {len(sequences)} samples")
    
    # Save sequences and labels
    sequences_path = output_dir / "api_sequences.json"
    with open(sequences_path, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved sequences to: {sequences_path}")
    
    labels_path = output_dir / "labels.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    logger.info(f"Saved labels to: {labels_path}")
    
    # Create splits (if not exists)
    if args.make_splits or not (splits_dir / "train.json").exists():
        import random
        random.seed(data_config['seed'])
        
        shas = list(sequences.keys())
        random.shuffle(shas)
        
        n = len(shas)
        n_train = int(n * data_config['train_frac'])
        n_val = int(n * data_config['val_frac'])
        
        train_shas = shas[:n_train]
        val_shas = shas[n_train:n_train + n_val]
        test_shas = shas[n_train + n_val:]
        
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

