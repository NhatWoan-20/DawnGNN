"""Path management utilities for DawnGNN."""

from pathlib import Path
from typing import Optional

# Resolve project root: src/utils/paths.py -> src/utils -> src -> project root
_UTILS_PATH = Path(__file__).resolve()
PROJECT_ROOT = _UTILS_PATH.parent.parent.parent
DAWNGNN_ROOT = PROJECT_ROOT


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def get_config_path(config_name: str) -> Path:
    """Get absolute path to a config file inside configs/."""
    return PROJECT_ROOT / "configs" / config_name


def get_data_path(*subpaths: str) -> Path:
    """Get absolute path to a file/directory inside data/."""
    return PROJECT_ROOT / "data" / Path(*subpaths)


def get_model_path(*subpaths: str) -> Path:
    """Get absolute path to a file/directory inside models/."""
    return PROJECT_ROOT / "models" / Path(*subpaths)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, create if it doesn't. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
