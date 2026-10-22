"""
Test suite for dataset loading and graph construction.
"""

from pathlib import Path
import pytest

from dawngnn.models.dataset import MalwareGraphDataset


def test_dataset_import():
    """Verify MalwareGraphDataset can be imported."""
    assert MalwareGraphDataset is not None
    assert hasattr(MalwareGraphDataset, "__init__")


@pytest.mark.integration
def test_dataset_instantiation(tmp_path):
    """Test that dataset can be instantiated with minimal config."""
    test_graph_dir = tmp_path / "graphs_augmented"
    test_graph_dir.mkdir()
    labels_file = tmp_path / "labels.json"
    labels_file.write_text("{}", encoding="utf-8")
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "train.json").write_text("[]", encoding="utf-8")

    config = {
        "paths": {
            "output_augmented_dir": str(test_graph_dir),
            "labels_file": str(labels_file),
            "splits_dir": str(splits_dir),
        }
    }

    try:
        dataset = MalwareGraphDataset(root=tmp_path, config=config, split_type="train")
        assert dataset is not None
    except Exception as e:
        pytest.fail(f"Dataset instantiation failed: {e}")
