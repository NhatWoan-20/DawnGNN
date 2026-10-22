"""Unit tests for the GATClassifier model."""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Try importing PyTorch Geometric — skip entire module if not installed
# ---------------------------------------------------------------------------
try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import global_mean_pool

    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _PYG_AVAILABLE, reason="torch_geometric not installed")

if _PYG_AVAILABLE:
    from dawngnn.models import GATClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_dummy_data(num_nodes: int = 6, in_channels: int = 768, num_graphs: int = 1):
    """Create a minimal synthetic PyG Data object for testing."""
    x = torch.randn(num_nodes * num_graphs, in_channels)
    # Simple chain: 0->1->2->3->4->5 per graph
    edge_list = []
    for g in range(num_graphs):
        offset = g * num_nodes
        for i in range(num_nodes - 1):
            edge_list.append([offset + i, offset + i + 1])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    batch = torch.repeat_interleave(torch.arange(num_graphs), num_nodes)
    y = torch.randint(0, 2, (num_graphs,))
    return Data(x=x, edge_index=edge_index, batch=batch, y=y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGATClassifier:
    def test_output_shape(self):
        """Model should output (batch_size, 2) logits."""
        model = GATClassifier(in_channels=768, hidden_channels=64, out_channels=2,
                              num_layers=2, num_heads=4, dropout=0.0)
        model.eval()
        data = make_dummy_data(num_nodes=6, in_channels=768, num_graphs=4)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_single_graph(self):
        """Model should work on a single graph (batch_size=1)."""
        model = GATClassifier(in_channels=32, hidden_channels=16, out_channels=2,
                              num_layers=2, num_heads=2, dropout=0.0)
        model.eval()
        data = make_dummy_data(num_nodes=3, in_channels=32, num_graphs=1)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (1, 2)

    def test_3_layer_model(self):
        """3-layer GAT should not crash."""
        model = GATClassifier(in_channels=64, hidden_channels=32, out_channels=2,
                              num_layers=3, num_heads=4, dropout=0.1)
        model.eval()
        data = make_dummy_data(num_nodes=8, in_channels=64, num_graphs=2)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (2, 2)

    def test_output_not_nan(self):
        """No NaN values in logits."""
        model = GATClassifier(in_channels=768, hidden_channels=128, out_channels=2,
                              num_layers=3, num_heads=8, dropout=0.4)
        model.eval()
        data = make_dummy_data(num_nodes=10, in_channels=768, num_graphs=8)
        with torch.no_grad():
            out = model(data)
        assert not torch.isnan(out).any(), "NaN values found in model output"

    def test_dropout_off_in_eval(self):
        """Repeated forward passes in eval mode should return identical results."""
        model = GATClassifier(in_channels=32, hidden_channels=16, out_channels=2,
                              num_layers=2, num_heads=2, dropout=0.5)
        model.eval()
        data = make_dummy_data(num_nodes=4, in_channels=32, num_graphs=2)
        with torch.no_grad():
            out1 = model(data)
            out2 = model(data)
        assert torch.allclose(out1, out2), "Eval mode outputs should be deterministic"
