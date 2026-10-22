#!/usr/bin/env python3
"""
plot_results.py - Visualize DawnGNN experiment results.

Generates:
    1. Grouped bar chart comparing 5 methods across 5 metrics (results/figures/comparison_bars.png)
    2. Radar/spider chart across metrics (results/figures/radar_chart.png)
    3. Results summary table figure (results/figures/results_table.png)

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --csv results/evaluation_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette (accessible, non-generic)
# ---------------------------------------------------------------------------
PALETTE = {
    "DawnGNN (BERT)":    "#4C72B0",   # deep blue
    "DawnGNN (OneHot)":  "#DD8452",   # amber
    "DawnGNN (Word2Vec)":"#55A868",   # teal green
    "MalDetConv":        "#C44E52",   # crimson
    "MalDy":             "#8172B2",   # purple
}

METRIC_LABELS = {
    "Accuracy":           "Accuracy",
    "Precision (Macro)":  "Precision\n(Macro)",
    "Recall (Macro)":     "Recall\n(Macro)",
    "F1-Score (Macro)":   "F1-Score\n(Macro)",
    "AUC-ROC":            "AUC-ROC",
}

# Hardcoded results (used when no CSV is supplied or CSV is missing columns)
_HARDCODED = [
    {
        "Method": "DawnGNN (BERT)",
        "Description": "GAT + BERT Embeddings",
        "Accuracy": 0.9529,
        "Precision (Macro)": 0.9548,
        "Recall (Macro)": 0.9516,
        "F1-Score (Macro)": 0.9527,
        "AUC-ROC": 0.9721,
    },
    {
        "Method": "DawnGNN (OneHot)",
        "Description": "GAT + One-Hot Encoding",
        "Accuracy": 0.9490,
        "Precision (Macro)": 0.9488,
        "Recall (Macro)": 0.9491,
        "F1-Score (Macro)": 0.9489,
        "AUC-ROC": 0.9705,
    },
    {
        "Method": "MalDetConv",
        "Description": "CNN + BiGRU + Word2Vec",
        "Accuracy": 0.9490,
        "Precision (Macro)": 0.9496,
        "Recall (Macro)": 0.9483,
        "F1-Score (Macro)": 0.9488,
        "AUC-ROC": 0.9737,
    },
    {
        "Method": "DawnGNN (Word2Vec)",
        "Description": "GAT + Word2Vec",
        "Accuracy": 0.9294,
        "Precision (Macro)": 0.9296,
        "Recall (Macro)": 0.9288,
        "F1-Score (Macro)": 0.9292,
        "AUC-ROC": 0.9731,
    },
    {
        "Method": "MalDy",
        "Description": "Ensemble ML (RF+XGB+SVM)",
        "Accuracy": 0.5333,
        "Precision (Macro)": 0.5357,
        "Recall (Macro)": 0.5355,
        "F1-Score (Macro)": 0.5332,
        "AUC-ROC": 0.5068,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(csv_path: Path) -> pd.DataFrame:
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        required = {"Method", "Accuracy", "F1-Score (Macro)"}
        if required.issubset(set(df.columns)):
            print(f"[INFO] Loaded results from {csv_path}")
            return df
        print(f"[WARN] CSV missing required columns — using hardcoded results.")
    else:
        print("[INFO] No CSV supplied — using hardcoded results.")
    return pd.DataFrame(_HARDCODED)


def apply_style():
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    })


# ---------------------------------------------------------------------------
# Plot 1: Grouped bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bars(df: pd.DataFrame, out_path: Path):
    metrics = list(METRIC_LABELS.keys())
    methods = df["Method"].tolist()
    n_methods = len(methods)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.75 / n_methods

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for i, method in enumerate(methods):
        row = df[df["Method"] == method].iloc[0]
        values = [row.get(m, 0) for m in metrics]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.9,
            label=method,
            color=PALETTE.get(method, f"C{i}"),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
        )
        # Annotate the DawnGNN (BERT) bars with values
        if method == "DawnGNN (BERT)":
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=PALETTE[method],
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=11)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "DawnGNN vs Baselines — Malware Detection Performance\n"
        "(MalBehavD-V1 Dataset, test set)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(n_metrics - 0.05, 0.952, "0.95", fontsize=8, color="gray", va="bottom", ha="right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Saved bar chart -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Metrics radar / spider chart
# ---------------------------------------------------------------------------

def plot_radar(df: pd.DataFrame, out_path: Path):
    metrics = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "AUC-ROC"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=10)
    ax.set_ylim(0.45, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.50", "0.60", "0.70", "0.80", "0.90", "1.00"], fontsize=7)
    ax.grid(alpha=0.3)

    for _, row in df.iterrows():
        method = row["Method"]
        values = [row.get(m, 0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle="solid",
                color=PALETTE.get(method, "gray"), label=method)
        ax.fill(angles, values, alpha=0.07, color=PALETTE.get(method, "gray"))

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Performance Radar Chart", fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Saved radar chart -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Summary table as a figure
# ---------------------------------------------------------------------------

def plot_table(df: pd.DataFrame, out_path: Path):
    cols = ["Method", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "AUC-ROC"]
    display_df = df[cols].copy()

    # Format floats
    for col in cols[1:]:
        display_df[col] = display_df[col].apply(lambda x: f"{float(x):.4f}")

    fig, ax = plt.subplots(figsize=(13, 3))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    col_labels = ["Method", "Accuracy", "Precision\n(Macro)", "Recall\n(Macro)", "F1-Score\n(Macro)", "AUC-ROC"]
    table = ax.table(
        cellText=display_df.values,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Header style
    for col_idx in range(len(col_labels)):
        table[0, col_idx].set_facecolor("#4C72B0")
        table[0, col_idx].set_text_props(color="white", fontweight="bold")

    # Highlight best row (DawnGNN BERT)
    best_row_idx = display_df[display_df["Method"] == "DawnGNN (BERT)"].index
    if len(best_row_idx):
        row_num = display_df.index.get_loc(best_row_idx[0]) + 1
        for col_idx in range(len(col_labels)):
            table[row_num, col_idx].set_facecolor("#EBF3FB")
            table[row_num, col_idx].set_text_props(fontweight="bold")

    ax.set_title(
        "Comparison of Malware Detection Methods (MalBehavD-V1, test set)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[INFO] Saved table figure -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot DawnGNN experiment results.")
    parser.add_argument(
        "--csv",
        default=str(RESULTS_DIR / "evaluation_summary.csv"),
        help="Path to evaluation_summary.csv (default: results/evaluation_summary.csv).",
    )
    args = parser.parse_args()

    apply_style()
    df = load_results(Path(args.csv))

    plot_comparison_bars(df, FIGURES_DIR / "comparison_bars.png")
    plot_radar(df, FIGURES_DIR / "radar_chart.png")
    plot_table(df, FIGURES_DIR / "results_table.png")

    print(f"\n[Done] All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
