#!/usr/bin/env python3
"""
Generate Word2Vec Embeddings for API calls.

Reads API sequences directly from CSV file (preserving temporal order),
trains a Word2Vec model to learn co-occurrence patterns,
and generates embedding vectors for each API.

Word2Vec treats each API as a "word" and each sequence as a "sentence",
learning contextual relationships based on co-occurrence in sequences.
"""

import pickle
import json
import numpy as np
import networkx as nx
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Set
import sys
from gensim.models import Word2Vec

# Add utils to path
DAWNGNN_WORD2VEC_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(DAWNGNN_WORD2VEC_ROOT / "src"))
from utils.paths import PROJECT_ROOT, get_config_path
from utils.logger import setup_logger

logger = setup_logger("word2vec_embed")

CONFIG_FILE_PATH = get_config_path('word2vec.yaml')


def load_config(path: Path) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {path}. Using defaults.")
        return {
            'paths': {
                'csv_file': 'data/raw/MalBehavD-V1-dataset.csv',
                'embeddings_dir': 'data/embeddings_word2vec',
                'model_file': 'data/embeddings_word2vec/word2vec.model'
            },
            'data': {
                'sha_col': 'sha256',
                'api_start_col': 2
            },
            'word2vec': {
                'vector_size': 128,
                'window': 5,
                'min_count': 1,
                'workers': 4,
                'sg': 0,  # 0 for CBOW, 1 for skip-gram
                'epochs': 10
            }
        }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def collect_api_sequences_from_csv(csv_path: Path, sha_col: str = 'sha256', api_start_col: int = 2) -> List[List[str]]:
    """
    Collect API sequences directly from CSV file to preserve temporal order.
    This is crucial for Word2Vec to learn co-occurrence patterns correctly.
    """
    import pandas as pd
    
    all_sequences = []
    
    logger.info(f"Reading API sequences from CSV: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return all_sequences
    
    df.fillna('', inplace=True)
    
    if sha_col not in df.columns:
        logger.error(f"Column '{sha_col}' not found in CSV")
        return all_sequences
    
    api_cols = list(df.columns)[api_start_col:]
    
    logger.info(f"Collecting sequences from {len(df)} rows...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting sequences"):
        # Extract API sequence preserving temporal order
        api_sequence = []
        for c in api_cols:
            v = row[c]
            if v is None:
                continue
            s = str(v).strip()
            if s == "" or s.lower() in ("nan", "none"):
                continue
            api_sequence.append(s)
        
        if len(api_sequence) > 0:
            all_sequences.append(api_sequence)
    
    logger.info(f"Collected {len(all_sequences)} sequences with temporal order preserved")
    return all_sequences


def train_word2vec(sequences: List[List[str]], config: Dict) -> Word2Vec:
    """Train Word2Vec model on API sequences."""
    word2vec_config = config.get('word2vec', {})
    
    logger.info("Training Word2Vec model...")
    logger.info(f"  Vector size: {word2vec_config.get('vector_size', 128)}")
    logger.info(f"  Window: {word2vec_config.get('window', 5)}")
    logger.info(f"  Min count: {word2vec_config.get('min_count', 1)}")
    logger.info(f"  Epochs: {word2vec_config.get('epochs', 10)}")
    
    model = Word2Vec(
        sentences=sequences,
        vector_size=word2vec_config.get('vector_size', 128),
        window=word2vec_config.get('window', 5),
        min_count=word2vec_config.get('min_count', 1),
        workers=word2vec_config.get('workers', 4),
        sg=word2vec_config.get('sg', 0),  # 0=CBOW, 1=skip-gram
        epochs=word2vec_config.get('epochs', 10)
    )
    
    logger.info(f"[+] Trained Word2Vec model with vocabulary size: {len(model.wv)}")
    return model


def save_embeddings(model: Word2Vec, output_dir: Path, unknown_strategy: str = 'zeros'):
    """Save Word2Vec embeddings as .npy files for each API."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_size = model.wv.vector_size
    
    logger.info(f"Saving embeddings to {output_dir}...")
    
    # Save embeddings for each API in vocabulary
    for api_name in tqdm(model.wv.index_to_key, desc="Saving vectors"):
        vector = model.wv[api_name]
        output_path = output_dir / f"{api_name}.npy"
        np.save(output_path, vector.astype(np.float32))
    
    # Create UNKNOWN vector
    if unknown_strategy == 'zeros':
        unknown_vector = np.zeros(vector_size, dtype=np.float32)
    elif unknown_strategy == 'random':
        # Random vector with small values
        unknown_vector = np.random.normal(0, 0.01, vector_size).astype(np.float32)
    else:
        unknown_vector = np.zeros(vector_size, dtype=np.float32)
    
    unknown_path = output_dir / "__UNKNOWN_API__.npy"
    np.save(unknown_path, unknown_vector)
    logger.info(f"Created UNKNOWN vector using strategy: {unknown_strategy}")
    
    logger.info(f"[+] Saved {len(model.wv)} + 1 embeddings to {output_dir}")


def main():
    # Load config
    config = load_config(CONFIG_FILE_PATH)
    
    # Get paths
    csv_path = PROJECT_ROOT / config['paths'].get('csv_file', 'data/raw/MalBehavD-V1-dataset.csv')
    embeddings_dir = PROJECT_ROOT / config['paths']['embeddings_dir']
    model_file = PROJECT_ROOT / config['paths']['model_file']
    unknown_strategy = config.get('embedding', {}).get('unknown_api_strategy', 'zeros')
    sha_col = config.get('data', {}).get('sha_col', 'sha256')
    api_start_col = config.get('data', {}).get('api_start_col', 2)
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        logger.info("Please specify csv_file path in config or ensure CSV exists at default location")
        return
    
    # Step 1: Collect API sequences from CSV (preserving temporal order)
    sequences = collect_api_sequences_from_csv(csv_path, sha_col, api_start_col)
    
    if len(sequences) == 0:
        logger.error("No sequences found in graphs!")
        return
    
    # Step 2: Train Word2Vec model
    model = train_word2vec(sequences, config)
    
    # Step 3: Save Word2Vec model
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_file))
    logger.info(f"[+] Saved Word2Vec model to {model_file}")
    
    # Step 4: Save embeddings as .npy files
    save_embeddings(model, embeddings_dir, unknown_strategy)
    
    logger.info(f"\n[+] Word2Vec embedding generation complete!")
    logger.info(f"   Vocabulary size: {len(model.wv)}")
    logger.info(f"   Embedding dimension: {model.wv.vector_size}")
    logger.info(f"   Embeddings saved to: {embeddings_dir}")


if __name__ == "__main__":
    main()

