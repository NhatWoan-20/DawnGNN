"""Graph processing utilities for DawnGNN."""

import pickle
from pathlib import Path
from typing import List

import networkx as nx


def normalize_api(name: str) -> str:
    """Strip whitespace from an API name string. Returns empty string for non-strings."""
    if not isinstance(name, str):
        return ""
    return name.strip()


def build_graph_from_api_list(api_list: List[str]) -> nx.DiGraph:
    """
    Build a directed, weighted API-call graph from a temporal call sequence.

    Each unique API call becomes a node.  Consecutive calls (a -> b) create a
    directed edge whose ``weight`` attribute counts how many times that
    transition occurred in the sequence.

    Args:
        api_list: Ordered list of API call names.

    Returns:
        :class:`networkx.DiGraph` with node attribute ``api_name`` and edge
        attribute ``weight``.
    """
    G = nx.DiGraph()

    for api in api_list:
        api = normalize_api(api)
        if not api:
            continue
        if not G.has_node(api):
            G.add_node(api, api_name=api)

    for a, b in zip(api_list, api_list[1:]):
        a = normalize_api(a)
        b = normalize_api(b)
        if not a or not b:
            continue
        if G.has_edge(a, b):
            G[a][b]["weight"] += 1
        else:
            G.add_edge(a, b, weight=1)

    return G


def save_graph(graph: nx.Graph, path: Path) -> None:
    """Pickle a NetworkX graph to *path*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: Path) -> nx.Graph:
    """Load a pickled NetworkX graph from *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)
