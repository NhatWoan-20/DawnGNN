#!/usr/bin/env python3
"""
Generate One-Hot Encodings for API calls.

Reads all graphs to collect unique API names, creates a vocabulary,
and generates one-hot encoding vectors for each API.
"""

import pickle
import json
import numpy as np
import networkx as nx
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Set, List
import sys

# Add utils to path
DAWNGNN_ONEHOT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(DAWNGNN_ONEHOT_ROOT / "src"))
from utils.paths import PROJECT_ROOT, get_data_path, get_config_path
from utils.logger import setup_logger

logger = setup_logger("onehot_embed")

CONFIG_FILE_PATH = get_config_path('onehot.yaml')


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
                'input_graphs_dir': 'data/graphs',
                'embeddings_dir': 'data/embeddings_onehot',
                'vocabulary_file': 'data/embeddings_onehot/vocabulary.json'
            }
        }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def collect_all_apis(graphs_dir: Path) -> Set[str]:
    """Collect all unique API names from all graphs."""
    all_apis = set()
    graph_files = list(graphs_dir.glob("*.gpickle"))
    
    logger.info(f"Collecting APIs from {len(graph_files)} graphs...")
    
    for graph_path in tqdm(graph_files, desc="Collecting APIs"):
        try:
            with open(graph_path, 'rb') as f:
                G: nx.DiGraph = pickle.load(f)
            
            for node_name in G.nodes():
                if node_name:  # Skip empty strings
                    all_apis.add(node_name)
        except Exception as e:
            logger.warning(f"Error loading graph {graph_path.name}: {e}")
            continue
    
    logger.info(f"Found {len(all_apis)} unique APIs")
    return all_apis


def create_vocabulary(apis: Set[str]) -> Dict[str, int]:
    """Create vocabulary mapping API name to index."""
    # Sort APIs for consistent ordering
    sorted_apis = sorted(list(apis))
    vocabulary = {api: idx for idx, api in enumerate(sorted_apis)}
    return vocabulary


def create_onehot_vectors(vocabulary: Dict[str, int], output_dir: Path, unknown_strategy: str = 'zeros'):
    """Create and save one-hot encoding vectors for each API."""
    vocab_size = len(vocabulary)
    logger.info(f"Creating one-hot vectors (vocab_size={vocab_size})...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create one-hot vector for each API
    for api_name, idx in tqdm(vocabulary.items(), desc="Creating vectors"):
        # Create one-hot vector: all zeros except at index idx
        vector = np.zeros(vocab_size, dtype=np.float32)
        vector[idx] = 1.0
        
        # Save vector
        output_path = output_dir / f"{api_name}.npy"
        np.save(output_path, vector)
    
    # Create UNKNOWN vector
    if unknown_strategy == 'zeros':
        unknown_vector = np.zeros(vocab_size, dtype=np.float32)
    elif unknown_strategy == 'uniform':
        # Uniform distribution (1/vocab_size for each dimension)
        unknown_vector = np.ones(vocab_size, dtype=np.float32) / vocab_size
    else:
        unknown_vector = np.zeros(vocab_size, dtype=np.float32)
    
    unknown_path = output_dir / "__UNKNOWN_API__.npy"
    np.save(unknown_path, unknown_vector)
    logger.info(f"Created UNKNOWN vector using strategy: {unknown_strategy}")
    
    logger.info(f"[+] Created {len(vocabulary) + 1} one-hot vectors in {output_dir}")


def main():
    # Load config
    config = load_config(CONFIG_FILE_PATH)
    
    # Get paths
    graphs_dir = PROJECT_ROOT / config['paths']['input_graphs_dir']
    embeddings_dir = PROJECT_ROOT / config['paths']['embeddings_dir']
    vocab_file = PROJECT_ROOT / config['paths']['vocabulary_file']
    unknown_strategy = config.get('embedding', {}).get('unknown_api_strategy', 'zeros')
    
    if not graphs_dir.exists():
        logger.error(f"Graphs directory not found: {graphs_dir}")
        return
    
    # Step 1: Collect all unique APIs
    all_apis = collect_all_apis(graphs_dir)
    
    if len(all_apis) == 0:
        logger.error("No APIs found in graphs!")
        return
    
    # Step 2: Create vocabulary
    vocabulary = create_vocabulary(all_apis)
    
    # Step 3: Save vocabulary
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    logger.info(f"[+] Saved vocabulary ({len(vocabulary)} APIs) to {vocab_file}")
    
    # Step 4: Create one-hot vectors
    create_onehot_vectors(vocabulary, embeddings_dir, unknown_strategy)
    
    logger.info(f"\n[+] One-hot embedding generation complete!")
    logger.info(f"   Vocabulary size: {len(vocabulary)}")
    logger.info(f"   Embedding dimension: {len(vocabulary)}")
    logger.info(f"   Embeddings saved to: {embeddings_dir}")


if __name__ == "__main__":
    main()

