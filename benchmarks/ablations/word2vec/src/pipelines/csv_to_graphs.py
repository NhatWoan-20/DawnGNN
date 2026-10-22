#!/usr/bin/env python3
"""
CSV -> Graphs (no intermediate .txt files)

Reads a dataset CSV where:
- column 'sha256' is the sample id
- column 'labels' is 0/1 label
- API calls are in subsequent columns (3rd column index 2 onward) in temporal order

Produces:
- data/graphs/<sha256>.gpickle    (one graph per sample)
- data/graphs/graphs_manifest.json
- data/intermediate/labels.json
- optionally data/splits/{train,val,test}.json
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import networkx as nx
from tqdm import tqdm
import random
import pickle
import sys

# Add utils to path
DAWNGNN_WORD2VEC_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(DAWNGNN_WORD2VEC_ROOT / "src"))
from utils.paths import PROJECT_ROOT, get_data_path

def normalize_api(name: str):
    """Normalize API names."""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    return s

def build_graph_from_api_list(api_list):
    """Build directed weighted API graph."""
    G = nx.DiGraph()
    
    for api in api_list:
        if not api:
            continue
        if not G.has_node(api):
            G.add_node(api, api_name=api)

    for a, b in zip(api_list, api_list[1:]):
        if not a or not b:
            continue
        if G.has_edge(a, b):
            G[a][b]['weight'] += 1
        else:
            G.add_edge(a, b, weight=1)

    return G

def make_splits(samples, train_frac=0.8, val_frac=0.1, seed=42):
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
    # Use absolute paths from project root
    if Path(args.csv).is_absolute():
        csv_path = Path(args.csv)
    else:
        csv_path = PROJECT_ROOT / args.csv
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    seq_out = get_data_path("intermediate")
    seq_out.mkdir(parents=True, exist_ok=True)

    graphs_out = get_data_path("graphs")
    graphs_out.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    df.fillna('', inplace=True)

    if args.sha_col not in df.columns:
        raise ValueError(f"Column '{args.sha_col}' not found in CSV. Available columns: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise ValueError(f"Column '{args.label_col}' not found in CSV. Available columns: {list(df.columns)}")

    labels = {}
    manifest = {}

    api_cols = list(df.columns)[args.api_start_col:]

    print(f"Processing {len(df)} rows -> building graphs ...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sha = str(row[args.sha_col]).strip()
        if not sha:
            continue

        # Parse label
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

            # Save graph using pickle
            with open(out_path, "wb") as f:
                pickle.dump(G, f)

            manifest[sha] = str(out_path)
            labels[sha] = int(label)
        except Exception as e:
            print(f"Warning: Error processing sample {sha}: {e}")
            continue

    try:
        with open(graphs_out / "graphs_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        raise IOError(f"Error saving manifest: {e}")

    try:
        with open(seq_out / "labels.json", "w") as f:
            json.dump(labels, f, indent=2)
    except Exception as e:
        raise IOError(f"Error saving labels: {e}")

    print(f"Saved {len(manifest)} graphs to {graphs_out}")
    print(f"Saved labels to {seq_out / 'labels.json'}")

    if args.make_splits:
        try:
            train, val, test = make_splits(list(manifest.keys()), args.train_frac, args.val_frac, args.seed)
            splits_dir = get_data_path("splits")
            splits_dir.mkdir(parents=True, exist_ok=True)

            with open(splits_dir / "train.json", "w") as f:
                json.dump(train, f, indent=2)
            with open(splits_dir / "val.json", "w") as f:
                json.dump(val, f, indent=2)
            with open(splits_dir / "test.json", "w") as f:
                json.dump(test, f, indent=2)

            print(f"Saved splits to {splits_dir} (train={len(train)}, val={len(val)}, test={len(test)})")
        except Exception as e:
            print(f"Warning: Error creating splits: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "MalBehavD-V1-dataset.csv"))
    parser.add_argument("--sha-col", type=str, default="sha256")
    parser.add_argument("--label-col", type=str, default="labels")
    parser.add_argument("--api-start-col", type=int, default=2)
    parser.add_argument("--make-splits", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

