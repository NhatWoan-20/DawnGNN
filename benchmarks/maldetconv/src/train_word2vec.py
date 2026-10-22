#!/usr/bin/env python3
"""
Train Word2Vec embeddings on API sequences.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict
from gensim.models import Word2Vec
import numpy as np
import sys
from tqdm import tqdm

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

logger = setup_logger("word2vec_train")


def load_sequences(sequences_path: Path) -> Dict[str, List[str]]:
    """Load API sequences."""
    with open(sequences_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def train_word2vec(sequences: Dict[str, List[str]], config: Dict) -> Word2Vec:
    """Train Word2Vec model on API sequences."""
    # Extract all sequences (remove padding tokens for training)
    all_sequences = []
    for seq in sequences.values():
        # Filter out padding tokens
        filtered_seq = [api for api in seq if api != "<PAD>"]
        if len(filtered_seq) > 0:
            all_sequences.append(filtered_seq)
    
    logger.info(f"Training Word2Vec on {len(all_sequences)} sequences")
    logger.info(f"Total unique APIs: {len(set(api for seq in all_sequences for api in seq))}")
    
    # Word2Vec config
    w2v_config = config['word2vec']
    
    # Train model
    model = Word2Vec(
        sentences=all_sequences,
        vector_size=w2v_config['vector_size'],
        window=w2v_config['window'],
        min_count=w2v_config['min_count'],
        workers=w2v_config['workers'],
        sg=w2v_config['sg'],  # 0=CBOW, 1=Skip-gram
        epochs=w2v_config['epochs']
    )
    
    logger.info(f"Word2Vec model trained. Vocabulary size: {len(model.wv)}")
    
    return model


def save_embeddings(model: Word2Vec, output_path: Path, vector_size: int):
    """Save Word2Vec embeddings as numpy array."""
    vocab_size = len(model.wv)
    
    # Create embedding matrix: [vocab_size, vector_size]
    # Index 0 reserved for padding token (zeros)
    embedding_matrix = np.zeros((vocab_size + 1, vector_size), dtype=np.float32)
    
    # Word to index mapping
    word_to_idx = {"<PAD>": 0}
    
    for i, word in enumerate(model.wv.index_to_key):
        idx = i + 1
        embedding_matrix[idx] = model.wv[word]
        word_to_idx[word] = idx
    
    # Save embedding matrix
    np.save(output_path, embedding_matrix)
    logger.info(f"Saved embedding matrix ({vocab_size + 1} x {vector_size}) to: {output_path}")
    
    return word_to_idx


def main():
    """Main function."""
    config_path = get_config_path("cnn_bigru.yaml")
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Paths - use utils functions to ensure paths are in maldetconv directory
    # Input sequences from shared data directory
    sequences_path = get_data_path("maldetconv", "processed", "api_sequences.json")
    # Output Word2Vec model and vectors to maldetconv/data/ directory
    model_output = get_local_data_path("word2vec_model.bin")
    vectors_output = get_local_data_path("word2vec_vectors.npy")
    
    # Create output directory
    model_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load sequences
    logger.info(f"Loading sequences from: {sequences_path}")
    sequences = load_sequences(sequences_path)
    
    # Train Word2Vec
    model = train_word2vec(sequences, config)
    
    # Save model
    model.save(str(model_output))
    logger.info(f"Saved Word2Vec model to: {model_output}")
    
    # Save embeddings
    word_to_idx = save_embeddings(model, vectors_output, config['word2vec']['vector_size'])
    
    # Save word to index mapping
    word_to_idx_path = vectors_output.parent / "word_to_idx.json"
    with open(word_to_idx_path, 'w', encoding='utf-8') as f:
        json.dump(word_to_idx, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved word-to-index mapping to: {word_to_idx_path}")
    
    logger.info("[+] Word2Vec training completed!")


if __name__ == "__main__":
    main()

