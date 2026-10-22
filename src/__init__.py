"""
DawnGNN — Documentation-Augmented Windows Malware Detection via GNN.

A Graph Attention Network with BERT-augmented API embeddings for Windows malware detection.
"""

__version__ = "1.0.0"
__author__ = "DawnGNN Contributors"

# Re-export main components for convenience
try:
    from dawngnn.models import GATClassifier, MalwareGraphDataset
    from dawngnn.utils import get_project_root, get_config_path, setup_logger

    __all__ = [
        "__version__",
        "__author__",
        "GATClassifier",
        "MalwareGraphDataset",
        "get_project_root",
        "get_config_path",
        "setup_logger",
    ]
except ImportError:
    # Allow package to be imported even if optional dependencies aren't installed
    __all__ = ["__version__", "__author__"]
