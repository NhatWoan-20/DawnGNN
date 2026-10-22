#!/usr/bin/env python3
"""
preprocess.py — Data sanity-check and split preparation for DawnGNN.

This script validates the raw CSV dataset and (optionally) regenerates
the train / val / test splits used by all downstream pipeline stages.
It is a lightweight wrapper around csv_to_graphs.py's split logic,
useful when you want to validate CSV integrity or re-split without re-building graphs.

Usage:
    # Validate CSV only
    python -m dawngnn.pipelines.preprocess
    # Or: python src/pipelines/preprocess.py

    # Validate + regenerate splits
    python -m dawngnn.pipelines.preprocess --make-splits

    # Custom CSV / split ratios
    python -m dawngnn.pipelines.preprocess --csv data/raw/custom.csv --make-splits \
        --train-frac 0.7 --val-frac 0.15 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

try:
    from dawngnn.utils.logger import setup_logger
    from dawngnn.utils.paths import ensure_dir, get_config_path, get_data_path
except ImportError:
    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from utils.logger import setup_logger
    from utils.paths import ensure_dir, get_config_path, get_data_path

logger = setup_logger("preprocess")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data_config() -> dict:
    """Load configs/data.yaml if present, else return sensible defaults."""
    try:
        with open(get_config_path("data.yaml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("configs/data.yaml not found — using built-in defaults.")
        return {}


def validate_csv(csv_path: Path, sha_col: str, label_col: str, api_start_col: int) -> pd.DataFrame:
    """
    Load and validate the raw dataset CSV.

    Checks:
    - File exists and is readable
    - Required columns are present
    - Labels are binary (0 / 1)
    - At least one API call column exists
    - Reports class distribution

    Returns:
        Validated DataFrame.
    """
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    logger.info(f"Loading dataset from {csv_path} …")
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)

    df.fillna("", inplace=True)
    logger.info(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # --- Column checks ---
    if sha_col not in df.columns:
        logger.error(f"SHA column '{sha_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)
    if label_col not in df.columns:
        logger.error(f"Label column '{label_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    api_cols = list(df.columns)[api_start_col:]
    if len(api_cols) == 0:
        logger.error(f"No API call columns found (api_start_col={api_start_col}).")
        sys.exit(1)
    logger.info(f"  SHA col: '{sha_col}'  |  Label col: '{label_col}'  |  API cols: {len(api_cols)}")

    # --- Label checks ---
    labels = df[label_col].str.strip()
    unique_labels = set(labels.unique())
    valid_labels = {"0", "1"}
    invalid = unique_labels - valid_labels
    if invalid:
        logger.warning(f"Unexpected label values found (will be treated as -1): {invalid}")

    label_counts = labels.value_counts()
    for lbl, count in label_counts.items():
        name = "Malware" if lbl == "1" else "Benign" if lbl == "0" else "Unknown"
        logger.info(f"  Class {lbl} ({name}): {count:,} samples ({100 * count / len(df):.1f}%)")

    # --- Duplicate SHA check ---
    dupes = df[sha_col].duplicated().sum()
    if dupes:
        logger.warning(f"  {dupes} duplicate SHA256 values detected.")
    else:
        logger.info("  No duplicate SHA256 values [OK]")

    # --- Empty API sequence check ---
    empty_rows = df[api_cols].apply(lambda row: all(v == "" for v in row), axis=1).sum()
    if empty_rows:
        logger.warning(f"  {empty_rows} rows with completely empty API sequences.")

    logger.info("Dataset validation passed [OK]")
    return df


def make_splits(
    sha_list: list,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Randomly partition *sha_list* into train / val / test splits."""
    random.seed(seed)
    shas = list(sha_list)
    random.shuffle(shas)
    n = len(shas)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return shas[:n_train], shas[n_train: n_train + n_val], shas[n_train + n_val:]


def save_splits(train: list, val: list, test: list, splits_dir: Path) -> None:
    """Write split lists as JSON files."""
    ensure_dir(splits_dir)
    for name, ids in [("train", train), ("val", val), ("test", test)]:
        path = splits_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
        logger.info(f"  Saved {name}: {len(ids):,} samples -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DawnGNN — dataset validation and split preparation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Load defaults from data.yaml
    data_cfg = load_data_config()
    raw_cfg = data_cfg.get("raw", {})
    splits_cfg = data_cfg.get("splits", {})

    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / raw_cfg.get("csv_path", "data/raw/MalBehavD-V1-dataset.csv")),
        help="Path to the raw dataset CSV.",
    )
    parser.add_argument("--sha-col", default=raw_cfg.get("sha_col", "sha256"))
    parser.add_argument("--label-col", default=raw_cfg.get("label_col", "labels"))
    parser.add_argument("--api-start-col", type=int, default=int(raw_cfg.get("api_start_col", 2)))
    parser.add_argument(
        "--make-splits",
        action="store_true",
        help="Regenerate train/val/test split JSON files.",
    )
    parser.add_argument("--train-frac", type=float, default=float(splits_cfg.get("train_frac", 0.8)))
    parser.add_argument("--val-frac", type=float, default=float(splits_cfg.get("val_frac", 0.1)))
    parser.add_argument("--seed", type=int, default=int(splits_cfg.get("seed", 42)))
    args = parser.parse_args()

    # Validate
    df = validate_csv(Path(args.csv), args.sha_col, args.label_col, args.api_start_col)

    if args.make_splits:
        sha_list = df[args.sha_col].str.strip().tolist()
        train, val, test = make_splits(sha_list, args.train_frac, args.val_frac, args.seed)
        splits_dir = PROJECT_ROOT / splits_cfg.get("dir", "data/splits")
        logger.info(
            f"Creating splits (train={args.train_frac:.0%}, "
            f"val={args.val_frac:.0%}, "
            f"test={1-args.train_frac-args.val_frac:.0%}...")
        save_splits(train, val, test, splits_dir)
        logger.info("Splits created [OK]")
    else:
        logger.info("Pass --make-splits to generate train/val/test split files.")


if __name__ == "__main__":
    main()
