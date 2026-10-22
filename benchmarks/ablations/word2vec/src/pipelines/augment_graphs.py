#!/usr/bin/env python3
"""
Augment graphs with Word2Vec encoding vectors.
"""

import pickle
import numpy as np
import networkx as nx
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
import sys

# Add utils to path
DAWNGNN_WORD2VEC_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(DAWNGNN_WORD2VEC_ROOT / "src"))
from utils.paths import PROJECT_ROOT, get_config_path
from utils.logger import setup_logger

logger = setup_logger("augment_graphs")

CONFIG_FILE_PATH = get_config_path('word2vec.yaml')


def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {path}. Using defaults.")
        return {
            'paths': {
                'embeddings_dir': 'data/embeddings_word2vec',
                'input_graphs_dir': 'data/graphs',
                'output_augmented_dir': 'data/graphs_augmented_word2vec',
            },
            'embedding': {
                'unknown_api_strategy': 'zeros'
            }
        }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def load_api_embeddings(embeddings_dir: Path) -> Dict[str, np.ndarray]:
    """Load all Word2Vec encoding vectors from embeddings directory."""
    api_embeddings: Dict[str, np.ndarray] = {}
    
    logger.info(f"Loading Word2Vec vectors from: {embeddings_dir}")
    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        return api_embeddings

    files = list(embeddings_dir.glob("*.npy"))
    if not files:
        logger.error(f"No .npy files found in {embeddings_dir}")
        return api_embeddings
        
    for file_path in files:
        api_name = file_path.stem
        try:
            vector = np.load(file_path)
            api_embeddings[api_name] = vector
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")

    logger.info(f"[+] Loaded {len(api_embeddings)} Word2Vec vectors")
    return api_embeddings


def augment_graph(input_graphs_dir: Path, output_augmented_dir: Path, 
                  api_embeddings: Dict[str, np.ndarray], unknown_api_key: str):
    """Augment graphs with Word2Vec encoding vectors."""
    output_augmented_dir.mkdir(parents=True, exist_ok=True)
    
    graph_files = list(input_graphs_dir.glob("*.gpickle"))
    if not graph_files:
        logger.error(f"No .gpickle files found in {input_graphs_dir}")
        return

    processed_count = 0
    unaugmented_graphs_count = 0 
    
    logger.info(f"Augmenting {len(graph_files)} graphs with Word2Vec encodings...")
    
    for graph_path in tqdm(graph_files, desc="Augmenting Graphs"):
        try:
            with open(graph_path, 'rb') as f:
                G: nx.DiGraph = pickle.load(f)

            known_api_count_per_graph = 0

            # Assign Word2Vec vectors to each node
            for node_name in G.nodes():
                feature_vector = api_embeddings.get(node_name)
                
                if feature_vector is not None:
                    G.nodes[node_name]['x'] = feature_vector
                    known_api_count_per_graph += 1
                else:
                    # Unknown API, use UNKNOWN vector
                    G.nodes[node_name]['x'] = api_embeddings[unknown_api_key]
            
            # Check if graph is fully unknown
            if known_api_count_per_graph == 0 and len(G.nodes) > 0:
                unaugmented_graphs_count += 1
            
            # Save augmented graph
            out_path = output_augmented_dir / graph_path.name
            with open(out_path, 'wb') as f:
                pickle.dump(G, f)
            
            processed_count += 1
            
        except Exception as e:
            logger.warning(f"Error processing graph {graph_path.name}: {e}")
            continue

    logger.info(f"\n[+] Completed! Augmented and saved {processed_count} graphs to {output_augmented_dir}")
    logger.info(f"   Total graphs processed: {processed_count}")
    logger.info(f"   Fully unknown graphs: {unaugmented_graphs_count}")
    logger.info(f"   Augmented graphs: {processed_count - unaugmented_graphs_count}")


def main():
    # Load config
    config = load_config(CONFIG_FILE_PATH)
    
    # Get paths
    embeddings_dir = PROJECT_ROOT / config['paths']['embeddings_dir']
    input_graphs_dir = PROJECT_ROOT / config['paths']['input_graphs_dir']
    output_augmented_dir = PROJECT_ROOT / config['paths']['output_augmented_dir']
    unknown_key = '__UNKNOWN_API__'

    # Load Word2Vec embeddings
    api_embeddings = load_api_embeddings(embeddings_dir)
    
    if not api_embeddings or len(api_embeddings) <= 1:
        logger.error("No valid embeddings found. Please run generate_word2vec_embeddings.py first.")
        return

    # Augment graphs
    augment_graph(input_graphs_dir, output_augmented_dir, api_embeddings, unknown_key)


if __name__ == "__main__":
    main()

