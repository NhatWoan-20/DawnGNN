"""
Models package for DawnGNN.
Re-exports canonical model and dataset definitions from the core training pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from dawngnn.pipelines.train_gnn import GATClassifier, MalwareGraphDataset
except ImportError:
    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from pipelines.train_gnn import GATClassifier, MalwareGraphDataset

__all__ = ["GATClassifier", "MalwareGraphDataset"]
