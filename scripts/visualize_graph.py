#!/usr/bin/env python3
"""
visualize_graph.py - Visual representation of API call transition graphs for DawnGNN.

Features:
    1. Single-sample graph visualization with node categorisation & degree scaling.
    2. Side-by-side comparative visualization (Benign vs Malware) highlighting topological differences.
    3. Categorizes APIs by functional domain (Process, Memory, Registry, File, Network, Sync/Crypto).
    4. Saves publication-quality figures to results/figures/.

Usage:
    # Generate side-by-side comparison of a Benign vs Malware graph:
    python scripts/visualize_graph.py --compare

    # Visualize a specific sample by SHA256:
    python scripts/visualize_graph.py --sha 5c18291c481a192ed5003084dab2d8a117fd3736359218fee2aea1a164544c9e

    # Visualize directly from a .gpickle file:
    python scripts/visualize_graph.py --file data/graphs/0a0b6c1073ab6b8759d40a364b8552dd565c20a74f1250005edc6a91e476bc4f.gpickle
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server execution
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GRAPHS_DIR = DATA_DIR / "graphs"
LABELS_PATH = DATA_DIR / "intermediate" / "labels.json"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API Functional Categorization Palette (Color-coded by system domain)
# ---------------------------------------------------------------------------
CATEGORY_COLORS = {
    "Process & Thread": "#E74C3C",  # Vibrant Red
    "Memory Operations": "#9B59B6", # Royal Purple
    "Registry Access":   "#E67E22", # Amber Orange
    "File & I/O":        "#2980B9", # Ocean Blue
    "Network & Socket":  "#27AE60", # Emerald Green
    "Sync & System":     "#7F8C8D", # Neutral Slate Gray
}

def categorize_api(api_name: str) -> str:
    """Categorizes a Windows API into its primary security functional role."""
    name = api_name.lower()
    
    # Process & Thread manipulation
    if any(k in name for k in ["process", "thread", "terminate", "resume", "suspend", "kill"]):
        return "Process & Thread"
    
    # Memory management & injection
    if any(k in name for k in ["alloc", "virtual", "memory", "protect", "free", "heap"]):
        return "Memory Operations"
    
    # Registry manipulation
    if any(k in name for k in ["reg", "key", "value"]):
        return "Registry Access"
        
    # File & Disk I/O
    if any(k in name for k in ["file", "dir", "path", "volume", "read", "write"]):
        return "File & I/O"
        
    # Networking & Sockets
    if any(k in name for k in ["socket", "recv", "send", "connect", "bind", "wsa", "net", "http", "internet"]):
        return "Network & Socket"
        
    # Synchronization, Library, Cryptography & Info
    return "Sync & System"


def load_graph(gpickle_path: Path) -> nx.DiGraph:
    """Loads a NetworkX DiGraph from a .gpickle serialized file."""
    if not gpickle_path.exists():
        raise FileNotFoundError(f"Graph file not found: {gpickle_path}")
    with open(gpickle_path, "rb") as f:
        return pickle.load(f)


def draw_subgraph_on_axis(
    G: nx.DiGraph,
    ax: plt.Axes,
    title: str,
    subtitle: str,
    max_nodes: int = 40,
    seed: int = 42
) -> None:
    """Draws a styled directed Markov transition graph on a Matplotlib axis."""
    # If graph is too large, take the top nodes by total degree to maintain readability
    if G.number_of_nodes() > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)[:max_nodes]
        G_viz = G.subgraph(top_nodes).copy()
    else:
        G_viz = G.copy()

    # Calculate layout using spring layout with tuned repulsive force
    pos = nx.spring_layout(G_viz, k=1.4 / (G_viz.number_of_nodes() ** 0.5 + 1e-5), seed=seed, iterations=60)

    # Node styling
    node_colors = [CATEGORY_COLORS[categorize_api(str(node))] for node in G_viz.nodes()]
    node_degrees = dict(G_viz.degree())
    node_sizes = [max(180, min(1200, 100 + deg * 55)) for deg in node_degrees.values()]

    # Draw nodes
    nx.draw_networkx_nodes(
        G_viz,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        edgecolors="#2C3E50",
        linewidths=1.2,
    )

    # Edge styling (frequency weights determine alpha and thickness)
    edges = list(G_viz.edges(data=True))
    if edges:
        weights = [d.get("weight", 1) for _, _, d in edges]
        max_w = max(weights) if weights else 1
        edge_widths = [max(0.8, min(3.5, 0.8 + (w / max_w) * 2.5)) for w in weights]
        edge_alphas = [max(0.3, min(0.85, 0.3 + (w / max_w) * 0.5)) for w in weights]
        
        for (u, v, _), w, alpha in zip(edges, edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G_viz,
                pos,
                edgelist=[(u, v)],
                ax=ax,
                width=w,
                alpha=alpha,
                edge_color="#5D6D7E",
                arrows=True,
                arrowsize=11,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.1",
            )

    # Node labels (truncated for visual cleanliness)
    labels = {
        node: str(node)[:14] + "…" if len(str(node)) > 15 else str(node)
        for node in G_viz.nodes()
    }
    nx.draw_networkx_labels(
        G_viz,
        pos,
        labels=labels,
        ax=ax,
        font_size=7.5,
        font_family="sans-serif",
        font_weight="bold",
        font_color="#1A252F",
    )

    # Formatting axis & stats
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#2C3E50")
    
    # Statistical watermark box
    stats_text = (
        f"{subtitle}\n"
        f"• Total APIs (Nodes): {G.number_of_nodes()}\n"
        f"• Transitions (Edges): {G.number_of_edges()}\n"
        f"• Density: {nx.density(G):.4f}"
    )
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#BDC3C7", alpha=0.9),
    )

    ax.axis("off")


def visualize_comparison(output_path: Optional[Path] = None) -> Path:
    """Finds a representative Benign and Malware sample and generates a side-by-side comparison figure."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"labels.json not found at {LABELS_PATH}. Run csv_to_graphs.py first.")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels: Dict[str, int] = json.load(f)

    # Find representative samples with a reasonable number of nodes (between 15 and 35) for aesthetic display
    benign_sha: Optional[str] = None
    malware_sha: Optional[str] = None

    for sha, lbl in labels.items():
        g_file = GRAPHS_DIR / f"{sha}.gpickle"
        if not g_file.exists():
            continue
        try:
            g = load_graph(g_file)
            if 18 <= g.number_of_nodes() <= 38 and g.number_of_edges() >= 20:
                if lbl == 0 and benign_sha is None:
                    benign_sha = sha
                elif lbl == 1 and malware_sha is None:
                    malware_sha = sha
            if benign_sha and malware_sha:
                break
        except Exception:
            continue

    # Fallback to any valid samples if specific range is not matched
    if not benign_sha or not malware_sha:
        for sha, lbl in labels.items():
            g_file = GRAPHS_DIR / f"{sha}.gpickle"
            if g_file.exists():
                if lbl == 0 and not benign_sha:
                    benign_sha = sha
                elif lbl == 1 and not malware_sha:
                    malware_sha = sha
            if benign_sha and malware_sha:
                break

    if not benign_sha or not malware_sha:
        raise ValueError("Could not find both Benign and Malware graph samples in data/graphs/.")

    G_benign = load_graph(GRAPHS_DIR / f"{benign_sha}.gpickle")
    G_malware = load_graph(GRAPHS_DIR / f"{malware_sha}.gpickle")

    # Setup side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    fig.suptitle(
        "DawnGNN — Dynamic API Transition Graphs Comparison",
        fontsize=16,
        fontweight="bold",
        color="#2C3E50",
        y=0.98,
    )

    draw_subgraph_on_axis(
        G_benign,
        axes[0],
        title="[0] Benign Software Call Graph",
        subtitle=f"SHA: {benign_sha[:16]}… (Safe System Activity)",
    )

    draw_subgraph_on_axis(
        G_malware,
        axes[1],
        title="[1] Windows Malware Call Graph",
        subtitle=f"SHA: {malware_sha[:16]}… (Intensive Injection / Mutex Clusters)",
    )

    # Legend for functional domains
    legend_patches = [
        mpatches.Patch(color=color, label=cat) for cat, color in CATEGORY_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(CATEGORY_COLORS),
        fontsize=9,
        frameon=True,
        facecolor="#F4F6F6",
        edgecolor="#D5D8DC",
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    out = output_path or (FIGURES_DIR / "graph_comparison_benign_vs_malware.png")
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved graph comparison visualization to: {out}")
    return out


def visualize_single(sha_or_path: str, output_path: Optional[Path] = None) -> Path:
    """Visualizes a single graph given its SHA256 or .gpickle path."""
    if Path(sha_or_path).suffix == ".gpickle" and Path(sha_or_path).exists():
        file_path = Path(sha_or_path)
        sha = file_path.stem
    else:
        sha = sha_or_path.strip()
        file_path = GRAPHS_DIR / f"{sha}.gpickle"

    if not file_path.exists():
        raise FileNotFoundError(f"Graph file does not exist: {file_path}")

    G = load_graph(file_path)

    label_str = "Unknown"
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
            if sha in labels:
                label_str = "Malware (1)" if labels[sha] == 1 else "Benign (0)"

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    draw_subgraph_on_axis(
        G,
        ax,
        title=f"DawnGNN API Call Graph — {label_str}",
        subtitle=f"SHA256: {sha}",
    )

    legend_patches = [
        mpatches.Patch(color=color, label=cat) for cat, color in CATEGORY_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        frameon=True,
        facecolor="#F4F6F6",
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    out = output_path or (FIGURES_DIR / f"graph_{sha[:16]}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved single graph visualization to: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize DawnGNN API call transition graphs.")
    parser.add_argument("--compare", action="store_true", help="Generate side-by-side Benign vs Malware comparison.")
    parser.add_argument("--sha", type=str, help="SHA256 hash of the graph sample to visualize.")
    parser.add_argument("--file", type=str, help="Direct path to a .gpickle file to visualize.")
    parser.add_argument("--output", type=str, help="Custom output PNG image path.")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None

    if args.sha:
        visualize_single(args.sha, out_p)
    elif args.file:
        visualize_single(args.file, out_p)
    else:
        # Default behavior: generate comparative visualization
        visualize_comparison(out_p)


if __name__ == "__main__":
    main()
