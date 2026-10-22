#!/usr/bin/env python3
"""
csv_to_graphs.py — Convert temporal API call traces from CSV into NetworkX DiGraphs.

Methodology (DawnGNN Phase 1):
------------------------------
1. Dynamic Malware Analysis Traces:
   Each sample in the dataset contains a chronological sequence of Win32 API calls
   recorded during sandbox execution.
2. Markovian State Transition Modeling:
   - Unique API call names are mapped to graph vertices (nodes).
   - Consecutive calls (api_t -> api_{t+1}) form directed edges representing execution flow transitions.
   - Transition frequencies between API calls are accumulated as edge weights:
     weight(u, v) = count(transitions from u to v).
3. Graph Persistence & Partitioning:
   - NetworkX DiGraph instances are serialized as .gpickle files (data/graphs/<sha256>.gpickle).
   - Graph manifest (graphs_manifest.json) and binary ground-truth labels (labels.json) are generated.
   - Reproducible train / validation / test splits (80/10/10 with seed=42) are generated when requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import random
import sys
from typing import List, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Path and Utility Bootstrap (Supports both direct script and module execution)
# -----------------------------------------------------------------------------
try:
    from dawngnn.utils.logger import setup_logger
    from dawngnn.utils.paths import get_data_path, get_project_root
except ImportError:
    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from utils.logger import setup_logger
    from utils.paths import get_data_path, get_project_root

PROJECT_ROOT = get_project_root()
DAWNGNN_ROOT = PROJECT_ROOT
logger = setup_logger("csv_to_graphs")


def normalize_api(name: str) -> str:
    """
    Normalize a Windows API function name by stripping leading/trailing whitespace.
    
    Args:
        name: Raw API name string from CSV record.
        
    Returns:
        Stripped string, or empty string if input is invalid/non-string.
    """
    if not isinstance(name, str):
        return ""
    return name.strip()


def build_graph_from_api_list(api_list: List[str]) -> nx.DiGraph:
    """
    Construct a directed, weighted API call transition graph from a chronological trace.

    Each unique API function becomes a graph node. Consecutive calls (u -> v)
    establish a directed edge whose 'weight' attribute reflects the transition frequency.

    Args:
        api_list: Chronological sequence of Windows API call names.

    Returns:
        nx.DiGraph containing unique API nodes and weighted transition edges.
    """
    G = nx.DiGraph()

    # Step 1: Add all unique API calls as nodes
    for api in api_list:
        if not api:
            continue
        if not G.has_node(api):
            G.add_node(api, api_name=api)

    # Step 2: Add directed edges with frequency-based weights
    for a, b in zip(api_list, api_list[1:]):
        if not a or not b:
            continue
        if G.has_edge(a, b):
            G[a][b]["weight"] += 1
        else:
            G.add_edge(a, b, weight=1)

    return G


def make_splits(
    samples: List[str], train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Partition sample SHA256 identifiers into train, validation, and test splits.

    Uses a fixed random seed (default: 42) to ensure 100% reproducible partitions
    matching the experimental setup in the published paper.

    Args:
        samples: List of unique sample IDs (SHA256 strings).
        train_frac: Proportion of samples allocated to training (default: 0.8 = 80%).
        val_frac: Proportion allocated to validation (default: 0.1 = 10%).
        seed: Random seed for deterministic shuffling (default: 42).

    Returns:
        Tuple of (train_ids, val_ids, test_ids).
    """
    random.seed(seed)
    samples = list(samples)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]
    return train, val, test


def main(args):
    """
    Main execution pipeline for raw CSV parsing and graph generation.

    Reads raw CSV containing SHA256 hashes, binary malware labels, and chronological API traces.
    Converts each sequence into an individual NetworkX DiGraph, serializes it to disk,
    and updates manifest and label index files. Optionally generates reproducible splits.
    """
    # Resolve CSV path (supports absolute and project-relative paths)
    if Path(args.csv).is_absolute():
        csv_path = Path(args.csv)
    else:
        csv_path = PROJECT_ROOT / args.csv

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Ensure output directories exist
    seq_out = get_data_path("intermediate")
    seq_out.mkdir(parents=True, exist_ok=True)

    graphs_out = get_data_path("graphs")
    graphs_out.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    df.fillna("", inplace=True)

    if args.sha_col not in df.columns:
        raise ValueError(
            f"Column '{args.sha_col}' not found in CSV. Available columns: {list(df.columns)}"
        )
    if args.label_col not in df.columns:
        raise ValueError(
            f"Column '{args.label_col}' not found in CSV. Available columns: {list(df.columns)}"
        )

    labels = {}
    manifest = {}

    api_cols = list(df.columns)[args.api_start_col :]

    logger.info(f"Processing {len(df)} rows -> building graphs ...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sha = str(row[args.sha_col]).strip()
        if not sha:
            continue

        # Parse binary label
        label = row[args.label_col]
        try:
            label = int(label)
        except (ValueError, TypeError):
            label = -1

        # Parse API call sequence
        api_calls = []
        for c in api_cols:
            v = row[c]
            if v is None:
                continue
            s = str(v).strip()
            if s == "" or s.lower() in ("nan", "none"):
                continue
            api_calls.append(normalize_api(s))
        if len(api_calls) == 0:
            continue

        # Build graph
        try:
            G = build_graph_from_api_list(api_calls)
            out_path = graphs_out / f"{sha}.gpickle"

            with open(out_path, "wb") as f:
                pickle.dump(G, f)

            manifest[sha] = str(out_path)
            labels[sha] = int(label)
        except Exception as e:
            logger.warning(f"Error processing sample {sha}: {e}")
            continue

    try:
        with open(graphs_out / "graphs_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        raise IOError(f"Error saving manifest: {e}")

    try:
        with open(seq_out / "labels.json", "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
    except Exception as e:
        raise IOError(f"Error saving labels: {e}")

    logger.info(f"Saved {len(manifest)} graphs to {graphs_out}")
    logger.info(f"Saved labels to {seq_out / 'labels.json'}")

    if args.make_splits:
        try:
            train, val, test = make_splits(list(manifest.keys()), args.train_frac, args.val_frac, args.seed)
            splits_dir = get_data_path("splits")
            splits_dir.mkdir(parents=True, exist_ok=True)

            with open(splits_dir / "train.json", "w", encoding="utf-8") as f:
                json.dump(train, f, indent=2)
            with open(splits_dir / "val.json", "w", encoding="utf-8") as f:
                json.dump(val, f, indent=2)
            with open(splits_dir / "test.json", "w", encoding="utf-8") as f:
                json.dump(test, f, indent=2)

            logger.info(
                f"Saved splits to {splits_dir} (train={len(train)}, val={len(val)}, test={len(test)})"
            )
        except Exception as e:
            logger.warning(f"Error creating splits: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert API trace CSV to NetworkX DiGraphs.")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "MalBehavD-V1-dataset.csv"),
        help="Path to raw dataset CSV",
    )
    parser.add_argument("--sha-col", type=str, default="sha256")
    parser.add_argument("--label-col", type=str, default="labels")
    parser.add_argument("--api-start-col", type=int, default=2)
    parser.add_argument("--make-splits", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
