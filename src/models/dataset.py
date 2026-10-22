"""
dataset.py — Malware Graph Dataset loader for PyTorch Geometric.

This module re-exports the canonical MalwareGraphDataset from dawngnn.pipelines.train_gnn.
Re-exporting from a single authoritative source guarantees 100% data processing consistency
with published experimental results while providing a clean public interface:
    from dawngnn.models import MalwareGraphDataset
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from dawngnn.pipelines.train_gnn import MalwareGraphDataset
except ImportError:
    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from pipelines.train_gnn import MalwareGraphDataset

__all__ = ["MalwareGraphDataset"]
