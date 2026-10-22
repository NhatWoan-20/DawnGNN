"""Path management utilities for DawnGNN-OneHot."""

from pathlib import Path
from typing import Optional

def _find_project_root(start_path: Path) -> Path:
    """Find the main DawnGNN repository root containing data/ and src/."""
    for parent in [start_path] + list(start_path.parents):
        if (parent / 'data').is_dir() and (parent / 'src').is_dir():
            return parent
    return start_path.parents[len(start_path.parents) - 1]


# Get project root (dawngnn_onehot directory)
_UTILS_PATH = Path(__file__).resolve()
DAWNGNN_ONEHOT_ROOT = _UTILS_PATH.parent.parent.parent
PROJECT_ROOT = _find_project_root(_UTILS_PATH)  # Main DawnGNN repository root


def get_project_root() -> Path:
    """Get the DawnGNN-OneHot root directory."""
    return DAWNGNN_ONEHOT_ROOT


def get_config_path(config_name: str) -> Path:
    """Get path to a config file."""
    return DAWNGNN_ONEHOT_ROOT / 'configs' / config_name


def get_data_path(*subpaths: str) -> Path:
    """Get path to data directory (shared with parent project)."""
    return PROJECT_ROOT / 'data' / Path(*subpaths)


def get_model_path(*subpaths: str) -> Path:
    """Get path to model directory."""
    return DAWNGNN_ONEHOT_ROOT / 'models' / Path(*subpaths)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path

