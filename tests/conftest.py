"""Pytest configuration and shared fixtures for DawnGNN tests."""

import sys
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Ensure src is in sys.path and dawngnn is importable even without editable install
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import dawngnn
except ImportError:
    import src
    sys.modules["dawngnn"] = src


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root directory."""
    return PROJECT_ROOT
