"""
infer.py — Standalone inference pipeline for DawnGNN Windows malware detection.

Inference Workflow:
-------------------
1. Input Sequence Normalization:
   Accepts a raw list of Win32 API names from CLI arguments or text files.
2. Dynamic Graph Construction:
   Builds a directed, weighted transition graph (NetworkX) from consecutive API calls.
3. Semantic Node Augmentation:
   Queries the local embedding cache for pre-computed 768-d BERT documentation embeddings.
   If an unseen API is encountered, falls back to on-the-fly BERT inference or zero-vector.
4. GAT Classification:
   Converts the graph to a PyTorch Geometric Data object and evaluates it with the trained
   GATClassifier (models/gnn/best_model.pt).
5. Result & Interpretability:
   Outputs binary classification verdict (Malware vs. Benign), softmax confidence probabilities,
   and graph degree centrality of the top-k most active API nodes.

Usage:
    python -m dawngnn.serving.infer --api-sequence "NtCreateFile,NtWriteFile,NtClose"
    python -m dawngnn.serving.infer --api-sequence-file path/to/sequence.txt
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml


try:
    from dawngnn.utils.graph_utils import build_graph_from_api_list, normalize_api
    from dawngnn.utils.logger import setup_logger
    from dawngnn.utils.paths import get_config_path, get_data_path, get_model_path, get_project_root
except ImportError:
    import sys
    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from utils.graph_utils import build_graph_from_api_list, normalize_api
    from utils.logger import setup_logger
    from utils.paths import get_config_path, get_data_path, get_model_path, get_project_root

logger = setup_logger("infer")
PROJECT_ROOT = get_project_root()

# ---------------------------------------------------------------------------
# Lazy-load heavy deps (transformers / torch_geometric) only when needed
# ---------------------------------------------------------------------------
_bert_tokenizer = None
_bert_model = None


def _get_bert(model_name: str = "bert-base-uncased"):
    """Load BERT tokenizer + model once and cache them."""
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading BERT model '{model_name}' …")
        _bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _bert_model = AutoModel.from_pretrained(model_name)
        _bert_model.eval()
    return _bert_tokenizer, _bert_model


try:
    from dawngnn.models import GATClassifier
except ImportError:
    from models import GATClassifier


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[Path] = None) -> dict:
    """Load gnn.yaml configuration."""
    path = config_path or get_config_path("gnn.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_api_embeddings(embeddings_dir: Path, default_size: int = 768) -> Dict[str, np.ndarray]:
    """Load all pre-computed BERT API embeddings (.npy) into a dict."""
    api_embeddings: Dict[str, np.ndarray] = {}
    if not embeddings_dir.exists():
        logger.warning(f"Embeddings directory not found: {embeddings_dir}")
        return api_embeddings
    for npy_file in embeddings_dir.glob("*.npy"):
        api_embeddings[npy_file.stem] = np.load(npy_file)
    api_embeddings["__UNKNOWN_API__"] = np.zeros(default_size, dtype=np.float32)
    logger.info(f"Loaded {len(api_embeddings)} API embeddings")
    return api_embeddings


def embed_api_name(
    api_name: str,
    api_embeddings: Dict[str, np.ndarray],
    bert_model_name: str = "bert-base-uncased",
    max_length: int = 256,
    fallback: str = "__UNKNOWN_API__",
) -> np.ndarray:
    """
    Return the embedding vector for *api_name*.

    Look-up order:
    1. Pre-computed cache (api_embeddings dict)
    2. On-the-fly BERT inference
    3. Zero vector (fallback)
    """
    if api_name in api_embeddings:
        return api_embeddings[api_name]

    # Try on-the-fly BERT (slower — only hits for truly unseen APIs)
    try:
        tokenizer, bert = _get_bert(bert_model_name)
        inputs = tokenizer(api_name, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = bert(**inputs)
        vec = outputs.last_hidden_state[0][0].numpy()
        api_embeddings[api_name] = vec  # cache for this session
        return vec
    except Exception as e:
        logger.warning(f"BERT embedding failed for '{api_name}': {e}. Using zero vector.")
        return api_embeddings.get(fallback, np.zeros(768, dtype=np.float32))


def build_pyg_graph(
    api_sequence: List[str],
    api_embeddings: Dict[str, np.ndarray],
    bert_model_name: str = "bert-base-uncased",
    default_size: int = 768,
):
    """
    Convert an API call sequence into an augmented PyG Data object ready for inference.

    Returns:
        ``torch_geometric.data.Data`` with node features (x) and edge_index.
    """
    from torch_geometric.data import Data

    # 1. Build NetworkX graph
    G = build_graph_from_api_list(api_sequence)

    if G.number_of_nodes() == 0:
        raise ValueError("API sequence produced an empty graph — check input.")

    # 2. Assign BERT node features
    node_list = list(G.nodes())
    mapping = {n: i for i, n in enumerate(node_list)}
    node_features = [
        embed_api_name(n, api_embeddings, bert_model_name=bert_model_name, fallback="__UNKNOWN_API__")
        for n in node_list
    ]
    x = torch.tensor(np.stack(node_features), dtype=torch.float)

    # 3. Build edge index
    edges = [[mapping[u], mapping[v]] for u, v in G.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 1), dtype=torch.long)

    # batch index (single graph -> all zeros)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, batch=batch)
    return data, node_list


def load_model(model_path: Path, config: dict, device: torch.device) -> GATClassifier:
    """Load the best trained GATClassifier from a checkpoint file."""
    mc = config["model"]
    model = GATClassifier(
        in_channels=mc["input_dim"],
        hidden_channels=mc["hidden_dim"],
        out_channels=2,
        num_layers=mc["num_layers"],
        num_heads=mc["num_heads"],
        dropout=mc["dropout"],
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(f"Loaded model from {model_path}")
    return model


# ---------------------------------------------------------------------------
# High-level predict function (used by api_server.py as well)
# ---------------------------------------------------------------------------

def predict(
    api_sequence: List[str],
    model: GATClassifier,
    api_embeddings: Dict[str, np.ndarray],
    device: torch.device,
    config: dict,
    top_k_nodes: int = 5,
) -> dict:
    """
    Run end-to-end inference on a raw API call sequence.

    Args:
        api_sequence: List of API call names in temporal order.
        model: Loaded GATClassifier.
        api_embeddings: Pre-computed BERT embeddings keyed by API name.
        device: Torch device.
        config: Loaded gnn.yaml config dict.
        top_k_nodes: Number of high-degree nodes to include in the explanation.

    Returns:
        dict with keys: label, label_name, malware_probability, benign_probability,
        unique_apis, graph_stats, top_apis.
    """
    bert_cfg = {}
    try:
        bert_cfg_path = get_config_path("bert.yaml")
        with open(bert_cfg_path, "r") as f:
            bert_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        pass

    bert_model_name = bert_cfg.get("model", {}).get("name", "bert-base-uncased")
    default_size = config.get("embedding", {}).get("default_size", 768)

    # Clean sequence
    api_sequence = [normalize_api(a) for a in api_sequence if normalize_api(a)]
    if not api_sequence:
        raise ValueError("Empty API sequence after normalization.")

    # Build graph
    data, node_list = build_pyg_graph(api_sequence, api_embeddings, bert_model_name, default_size)
    data = data.to(device)

    # Build NetworkX graph for stats
    G = build_graph_from_api_list(api_sequence)

    # Inference
    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    label = int(np.argmax(probs))
    label_name = "Malware" if label == 1 else "Benign"

    # Top-k most connected API nodes (by degree — lightweight explanation)
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top_apis = [{"api": name, "degree": deg} for name, deg in degrees[:top_k_nodes]]

    return {
        "label": label,
        "label_name": label_name,
        "malware_probability": float(probs[1]),
        "benign_probability": float(probs[0]),
        "unique_apis": len(node_list),
        "total_calls": len(api_sequence),
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "top_apis": top_apis,
    }


_cached_infer_state: Dict[str, Any] = {}


def predict_sequence(
    api_sequence: List[str],
    explain: bool = True,
    top_k_nodes: int = 5,
    model_path: Optional[Path] = None,
) -> dict:
    """
    High-level convenience function for interactive environments (Jupyter demo notebooks).

    Automatically initializes model, embedding dictionary, and configuration on first call,
    caching them for rapid subsequent inference.

    Args:
        api_sequence: List of Windows API call names in temporal order.
        explain: If True, includes top-k degree centrality APIs in the result.
        top_k_nodes: Number of top active API nodes to extract.
        model_path: Optional explicit path to model checkpoint.

    Returns:
        Dictionary with verdict, probabilities, confidence, and graph statistics.
    """
    global _cached_infer_state
    if "model" not in _cached_infer_state or _cached_infer_state["model"] is None:
        config = load_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings_dir = PROJECT_ROOT / config["paths"]["embeddings_dir"]
        api_embeddings = load_api_embeddings(
            embeddings_dir, config.get("embedding", {}).get("default_size", 768)
        )
        mp = model_path or get_model_path("gnn", "best_model.pt")
        model = load_model(mp, config, device)
        _cached_infer_state = {
            "config": config,
            "device": device,
            "api_embeddings": api_embeddings,
            "model": model,
        }

    state = _cached_infer_state
    res = predict(
        api_sequence=api_sequence,
        model=state["model"],
        api_embeddings=state["api_embeddings"],
        device=state["device"],
        config=state["config"],
        top_k_nodes=top_k_nodes,
    )

    confidence = max(res["malware_probability"], res["benign_probability"])
    return {
        "prediction": res["label_name"],
        "predicted_label": res["label"],
        "confidence": confidence,
        "probabilities": {
            "benign": res["benign_probability"],
            "malware": res["malware_probability"],
        },
        "graph_stats": res["graph_stats"],
        "top_apis": res["top_apis"] if explain else [],
        "total_calls": res["total_calls"],
        "unique_apis": res["unique_apis"],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="DawnGNN — single-sample malware inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/serving/infer.py --api-sequence "NtCreateFile,NtWriteFile,NtClose"
  python src/serving/infer.py --api-sequence-file my_sequence.txt
  python src/serving/infer.py --api-sequence "OpenProcess,VirtualAllocEx" --model models/gnn/best_model.pt
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--api-sequence",
        type=str,
        help="Comma-separated list of API call names.",
    )
    group.add_argument(
        "--api-sequence-file",
        type=str,
        help="Path to a text file with one API call per line.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (default: models/gnn/best_model.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to gnn.yaml config (default: configs/gnn.yaml).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top API nodes to show in explanation (default: 5).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Load API sequence
    if args.api_sequence:
        api_sequence = [a.strip() for a in args.api_sequence.split(",")]
    else:
        seq_path = Path(args.api_sequence_file)
        if not seq_path.exists():
            logger.error(f"File not found: {seq_path}")
            sys.exit(1)
        api_sequence = [ln.strip() for ln in seq_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    logger.info(f"Input sequence: {len(api_sequence)} API calls, {len(set(api_sequence))} unique")

    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load embeddings
    embeddings_dir = PROJECT_ROOT / config["paths"]["embeddings_dir"]
    api_embeddings = load_api_embeddings(embeddings_dir, config.get("embedding", {}).get("default_size", 768))

    # Load model
    model_path = Path(args.model) if args.model else get_model_path("gnn", "best_model.pt")
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}. Run training first.")
        sys.exit(1)
    model = load_model(model_path, config, device)

    # Predict
    result = predict(api_sequence, model, api_embeddings, device, config, top_k_nodes=args.top_k)

    # Pretty print
    print("\n" + "=" * 55)
    print("  DawnGNN Malware Detection — Inference Result")
    print("=" * 55)
    print(f"  Verdict       : {result['label_name']} (class {result['label']})")
    print(f"  Malware prob  : {result['malware_probability']:.4f}")
    print(f"  Benign  prob  : {result['benign_probability']:.4f}")
    print(f"  Total API calls : {result['total_calls']}")
    print(f"  Unique API nodes: {result['unique_apis']}")
    print(f"  Graph edges     : {result['graph_stats']['edges']}")
    print(f"\n  Top {args.top_k} most connected APIs:")
    for entry in result["top_apis"]:
        print(f"    [{entry['degree']:3d}] {entry['api']}")
    print("=" * 55 + "\n")

    return result


if __name__ == "__main__":
    main()
