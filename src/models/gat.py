"""
gat.py — Graph Attention Network (GAT) Classifier for Malware Detection.

This module re-exports the canonical GATClassifier from dawngnn.pipelines.train_gnn.
Re-exporting from a single authoritative source guarantees 100% architectural and numerical
parity with the trained paper checkpoint (models/gnn/best_model.pt) while enabling clean
modular imports:
    from dawngnn.models import GATClassifier
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from dawngnn.pipelines.train_gnn import GATClassifier
except ImportError:
    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from pipelines.train_gnn import GATClassifier

__all__ = ["GATClassifier"]
