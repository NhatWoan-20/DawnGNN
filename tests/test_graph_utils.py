"""Unit tests for graph construction utilities."""

import pickle

import networkx as nx
import pytest

from dawngnn.utils.graph_utils import (
    build_graph_from_api_list,
    load_graph,
    normalize_api,
    save_graph,
)


# ---------------------------------------------------------------------------
# normalize_api
# ---------------------------------------------------------------------------

class TestNormalizeApi:
    def test_strips_whitespace(self):
        assert normalize_api("  NtCreateFile  ") == "NtCreateFile"

    def test_non_string_returns_empty(self):
        assert normalize_api(None) == ""
        assert normalize_api(123) == ""

    def test_empty_string(self):
        assert normalize_api("") == ""

    def test_no_op_for_clean_name(self):
        assert normalize_api("OpenProcess") == "OpenProcess"


# ---------------------------------------------------------------------------
# build_graph_from_api_list
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_basic_sequence(self):
        apis = ["NtCreateFile", "NtWriteFile", "NtClose"]
        G = build_graph_from_api_list(apis)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 3
        assert G.has_edge("NtCreateFile", "NtWriteFile")
        assert G.has_edge("NtWriteFile", "NtClose")

    def test_edge_weight_increments(self):
        apis = ["A", "B", "A", "B"]
        G = build_graph_from_api_list(apis)
        assert G["A"]["B"]["weight"] == 2

    def test_empty_sequence_returns_empty_graph(self):
        G = build_graph_from_api_list([])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_single_api_no_edges(self):
        G = build_graph_from_api_list(["NtCreateFile"])
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0

    def test_filters_empty_strings(self):
        apis = ["NtCreateFile", "", "NtClose"]
        G = build_graph_from_api_list(apis)
        # "" is stripped and skipped -> no edge between NtCreateFile and NtClose
        assert G.number_of_nodes() == 2
        assert not G.has_edge("NtCreateFile", "NtClose")

    def test_node_has_api_name_attribute(self):
        G = build_graph_from_api_list(["OpenProcess"])
        assert G.nodes["OpenProcess"]["api_name"] == "OpenProcess"

    def test_self_loop_not_created_for_repeated_calls(self):
        # Consecutive same-API calls should still create a self-loop (valid)
        apis = ["A", "A"]
        G = build_graph_from_api_list(apis)
        assert G.has_edge("A", "A")
        assert G["A"]["A"]["weight"] == 1


# ---------------------------------------------------------------------------
# save_graph / load_graph round-trip
# ---------------------------------------------------------------------------

class TestGraphIO:
    def test_round_trip(self, tmp_path):
        apis = ["NtCreateFile", "NtWriteFile", "NtClose"]
        G = build_graph_from_api_list(apis)
        path = tmp_path / "test.gpickle"
        save_graph(G, path)
        G2 = load_graph(path)
        assert G2.number_of_nodes() == G.number_of_nodes()
        assert G2.number_of_edges() == G.number_of_edges()

    def test_creates_parent_dirs(self, tmp_path):
        G = build_graph_from_api_list(["A"])
        deep_path = tmp_path / "nested" / "dir" / "g.gpickle"
        save_graph(G, deep_path)
        assert deep_path.exists()
