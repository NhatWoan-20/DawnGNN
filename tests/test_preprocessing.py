"""
Test suite for data preprocessing pipeline.
Validates CSV parsing, graph construction, and data splits.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


def test_splits_exist(project_root):
    """Verify train/val/test split files exist."""
    splits_dir = project_root / "data" / "splits"
    assert splits_dir.exists(), f"Splits directory not found: {splits_dir}"

    for split in ["train.json", "val.json", "test.json"]:
        split_file = splits_dir / split
        assert split_file.exists(), f"Split file not found: {split_file}"


def test_splits_valid_json(project_root):
    """Verify split files contain valid JSON with expected structure."""
    splits_dir = project_root / "data" / "splits"

    for split in ["train.json", "val.json", "test.json"]:
        split_file = splits_dir / split
        with open(split_file, "r") as f:
            data = json.load(f)
        
        assert isinstance(data, list), f"{split} should contain a list"
        assert len(data) > 0, f"{split} should not be empty"
        
        # Check first item is a valid sample identifier (string SHA256 hash or dict)
        if data:
            first_item = data[0]
            if isinstance(first_item, dict):
                assert "sha256" in first_item or "id" in first_item, f"{split} items missing identifier"
            else:
                assert isinstance(first_item, str) and len(first_item) == 64, f"{split} item should be a 64-char SHA256 string"


def test_labels_file_exists(project_root):
    """Verify labels file exists and contains valid data."""
    labels_file = project_root / "data" / "intermediate" / "labels.json"
    assert labels_file.exists(), f"Labels file not found: {labels_file}"

    with open(labels_file, "r") as f:
        labels = json.load(f)
    
    assert isinstance(labels, dict), "Labels should be a dictionary"
    assert len(labels) > 0, "Labels dictionary should not be empty"
