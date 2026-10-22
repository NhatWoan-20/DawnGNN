#!/usr/bin/env python3
"""
embed_api.py — Generate BERT semantic embeddings for Windows API documentation.

Methodology (DawnGNN Phase 2 - Semantic Embedding):
---------------------------------------------------
1. Domain Semantic Knowledge:
   Unlike traditional malware detection techniques that treat API calls as arbitrary categorical
   tokens (One-Hot) or statistical co-occurrences (Word2Vec), DawnGNN leverages the natural language
   documentation published on Microsoft Learn.
2. Contextual Representation Extraction:
   - Tokenizes API functional descriptions using bert-base-uncased.
   - Truncates/pads to max_length (default: 256 tokens).
   - Passes tokens through the 12-layer bidirectional Transformer encoder.
   - Extracts the 768-dimensional [CLS] token representation from the last hidden state:
     h_api = outputs.last_hidden_state[0, 0, :] in R^768.
3. Fast Random-Access Storage:
   - Each vector is saved as an individual NumPy binary file: data/embeddings/<api_name>.npy.
   - This enables O(1) file-based lookups during subsequent graph node augmentation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
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

logger = setup_logger("embed_api")
CONFIG_FILE_PATH = get_config_path("bert.yaml")


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load BERT model hyperparameters and data paths from YAML configuration file.

    Args:
        path: Absolute Path to the bert.yaml configuration file.

    Returns:
        Dictionary containing model name, max_length, input_json, and output_dir.
    """
    logger.info(f"Loading configuration from: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found at: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading YAML config: {e}")
        raise


def get_embedding(text: str, tokenizer, model, max_length: int) -> np.ndarray:
    """
    Extract a 768-dimensional BERT [CLS] embedding vector for input documentation text.

    Args:
        text: Natural language string describing API functionality and parameters.
        tokenizer: HuggingFace AutoTokenizer (e.g., bert-base-uncased).
        model: HuggingFace AutoModel in eval mode.
        max_length: Maximum sequence length for tokenization (truncation applied).

    Returns:
        np.ndarray of shape (768,) containing the normalized continuous embedding.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract [CLS] token vector (first token of the first batch sequence)
    cls_vec = outputs.last_hidden_state[0][0]
    return cls_vec.cpu().numpy()


def main():
    # 1. Load config
    config = load_config(CONFIG_FILE_PATH)

    INPUT_JSON = config["data"]["input_json"]
    OUTPUT_DIR = config["data"]["output_dir"]
    MODEL_NAME = config["model"]["name"]
    MAX_LENGTH = config["model"]["max_length"]

    # 2. Initialize Model and Tokenizer
    logger.info(f"Loading BERT model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # 3. Load API JSON Documentation
    input_path = PROJECT_ROOT.joinpath(INPUT_JSON)
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            api_docs = json.load(f)
        logger.info(f"Loaded {len(api_docs)} API descriptions from {input_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}. Please check JSON path.")
        return

    # 4. Create Output Directory
    output_directory_path = PROJECT_ROOT.joinpath(OUTPUT_DIR)
    os.makedirs(output_directory_path, exist_ok=True)
    logger.info(f"Embeddings directory ready at: {output_directory_path}")

    # 5. Generate and Save Embeddings
    total_files_saved = 0
    for api_name, description in tqdm(api_docs.items(), desc="Embedding APIs"):
        vec = get_embedding(description, tokenizer, model, MAX_LENGTH)

        output_path = output_directory_path.joinpath(f"{api_name}.npy")
        np.save(output_path, vec)
        total_files_saved += 1

    logger.info(f"Done! Created and saved {total_files_saved} embeddings in {output_directory_path}")


if __name__ == "__main__":
    main()