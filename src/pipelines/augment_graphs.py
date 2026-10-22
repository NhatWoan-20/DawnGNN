#!/usr/bin/env python3
"""
augment_graphs.py — Attach BERT API semantic embeddings as node features to NetworkX graphs.

Methodology (DawnGNN Phase 3 - Semantic Graph Augmentation):
------------------------------------------------------------
1. Bridging Topology & Documentation Semantics:
   Purely structural Graph Neural Networks treat node identities as uninformative indices.
   In DawnGNN, each node represents an API whose semantic functionality (file creation,
   registry manipulation, thread injection, etc.) is captured by its 768-d BERT vector.
2. Node Feature Assignment:
   - For every vertex v in G = (V, E):
     If v exists in the embedding vocabulary:
         G.nodes[v]['x'] = e_v in R^768  (pre-computed BERT [CLS] embedding)
     Else (undocumented Win32 API, internal kernel syscall, or obfuscated symbol):
         G.nodes[v]['x'] = 0 in R^768    (zero-vector fallback)
3. Augmented Graph Serialization:
   - Augmented graphs are written to data/graphs_augmented/<sha256>.gpickle.
   - The node attribute 'x' is directly recognized and converted into PyTorch Geometric
     node feature tensors: data.x in R^{|V| x 768}.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict

import networkx as nx
import numpy as np
from tqdm import tqdm
import yaml

# -----------------------------------------------------------------------------
# Path and Utility Bootstrap (Supports both direct script and module execution)
# -----------------------------------------------------------------------------
try:
    from dawngnn.utils.logger import setup_logger
    from dawngnn.utils.paths import ensure_dir, get_config_path, get_project_root
except ImportError:
    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from utils.logger import setup_logger
    from utils.paths import ensure_dir, get_config_path, get_project_root

PROJECT_ROOT = get_project_root()
DAWNGNN_ROOT = PROJECT_ROOT

logger = setup_logger("augment_graphs")
CONFIG_FILE_PATH = get_config_path("gnn.yaml")

# Fallback default configuration if YAML file is unavailable
DEFAULT_CONFIG = {
    "paths": {
        "embeddings_dir": "data/embeddings",
        "input_graphs_dir": "data/graphs",
        "output_augmented_dir": "data/graphs_augmented",
    },
    "embedding": {
        "default_size": 768,
        "unknown_api_strategy": "zeros",
    },
}


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file or return fallback default config.

    Args:
        path: Path to gnn.yaml configuration file.

    Returns:
        Dictionary containing pipeline paths and embedding parameters.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using default configuration.")
        return DEFAULT_CONFIG
    except Exception as e:
        logger.warning(f"Error reading YAML file {e}. Using default configuration.")
        return DEFAULT_CONFIG


def load_api_embeddings(embeddings_dir: Path, default_size: int, strategy: str) -> Dict[str, np.ndarray]:
    """
    Load all pre-computed BERT embedding vectors from the embeddings directory into memory.

    Args:
        embeddings_dir: Directory containing per-API .npy embedding files.
        default_size: Dimensionality of embedding vectors (768 for bert-base-uncased).
        strategy: Handling strategy for out-of-vocabulary APIs ('zeros' allocates R^768 zero vector).

    Returns:
        Dictionary mapping API function names to their NumPy float32 embedding arrays.
    """
    api_embeddings: Dict[str, np.ndarray] = {}

    logger.info(f"Loading embedding vectors from: {embeddings_dir}")
    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory {embeddings_dir} does not exist.")
        return api_embeddings

    files = list(embeddings_dir.glob("*.npy"))
    if not files:
        logger.error(f"No .npy files found in {embeddings_dir}.")
        return api_embeddings

    for file_path in files:
        api_name = file_path.stem
        try:
            vector = np.load(file_path)
            api_embeddings[api_name] = vector
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")

    # Out-of-vocabulary (OOV) representation fallback
    if strategy == "zeros":
        api_embeddings["__UNKNOWN_API__"] = np.zeros(default_size, dtype=np.float32)

    logger.info(f"Successfully loaded {len(api_embeddings)} embedding vectors.")
    return api_embeddings


def augment_graph(
    input_graphs_dir: Path,
    output_augmented_dir: Path,
    api_embeddings: Dict[str, np.ndarray],
    unknown_api_key: str,
):
    """
    Iterate over raw NetworkX graphs, assign node feature attribute 'x', and save augmented graphs.

    Args:
        input_graphs_dir: Directory containing raw .gpickle graph files.
        output_augmented_dir: Target directory for semantically-augmented .gpickle graphs.
        api_embeddings: Dictionary of pre-loaded BERT embedding vectors.
        unknown_api_key: Key for OOV vector (e.g., '__UNKNOWN_API__').
    """
    output_augmented_dir.mkdir(parents=True, exist_ok=True)

    graph_files = list(input_graphs_dir.glob("*.gpickle"))
    if not graph_files:
        logger.error(f"No .gpickle graph files found in {input_graphs_dir}.")
        return

    processed_count = 0
    unaugmented_graphs_count = 0

    logger.info(f"Augmenting {len(graph_files)} graphs with semantic embeddings...")

    for graph_path in tqdm(graph_files, desc="Augmenting Graphs"):
        try:
            with open(graph_path, "rb") as f:
                G: nx.DiGraph = pickle.load(f)

            known_api_count_per_graph = 0

            # Assign embedding features to each graph node
            for node_name in G.nodes():
                feature_vector = api_embeddings.get(node_name)
                if feature_vector is not None:
                    G.nodes[node_name]["x"] = feature_vector
                    known_api_count_per_graph += 1
                else:
                    G.nodes[node_name]["x"] = api_embeddings[unknown_api_key]

            # Track completely unaugmented graphs (100% unknown APIs)
            if known_api_count_per_graph == 0 and len(G.nodes) > 0:
                unaugmented_graphs_count += 1

            # Save augmented graph
            out_path = output_augmented_dir.joinpath(graph_path.name)
            with open(out_path, "wb") as f:
                pickle.dump(G, f)

            processed_count += 1

        except Exception as e:
            logger.warning(f"Error processing graph {graph_path.name}: {e}")
            continue

    logger.info(f"Finished! Augmented and saved {processed_count} graphs to {output_augmented_dir}")
    logger.info(f"Total graphs processed: {processed_count}")
    logger.info(f"Graphs with 100% unknown APIs: {unaugmented_graphs_count}")
    logger.info(f"Graphs with >=1 documented API: {processed_count - unaugmented_graphs_count}")


def main():
    # 1. Load config
    config = load_config(CONFIG_FILE_PATH)

    # 2. Build paths from configuration
    EMBEDDINGS_DIR = PROJECT_ROOT.joinpath(config["paths"]["embeddings_dir"])
    INPUT_GRAPHS_DIR = PROJECT_ROOT.joinpath(config["paths"]["input_graphs_dir"])
    OUTPUT_AUGMENTED_DIR = PROJECT_ROOT.joinpath(config["paths"]["output_augmented_dir"])

    DEFAULT_SIZE = config["embedding"]["default_size"]
    UNKNOWN_STRATEGY = config["embedding"]["unknown_api_strategy"]
    UNKNOWN_KEY = "__UNKNOWN_API__"

    # 3. Load embeddings
    api_embeddings = load_api_embeddings(EMBEDDINGS_DIR, DEFAULT_SIZE, UNKNOWN_STRATEGY)
    if not api_embeddings or len(api_embeddings) <= 1:
        logger.error("No valid embeddings available. Stopping graph augmentation.")
        return

    # 4. Augment graphs
    augment_graph(INPUT_GRAPHS_DIR, OUTPUT_AUGMENTED_DIR, api_embeddings, UNKNOWN_KEY)


if __name__ == "__main__":
    main()