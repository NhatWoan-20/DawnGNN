# DawnGNN Benchmark Results

This directory contains experimental evaluation results across all 5 benchmarked methods on the **MalBehavD-V1** test split ($N = 255$ samples, 80/10/10 reproducible partition with seed 42).

## 🏆 Comparative Evaluation Summary

| Method | Architecture / Encoding | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | AUC-ROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DawnGNN (BERT)** ⭐ | **GAT + Win32 Doc Semantic Embeddings** | **0.9529** | **0.9548** | **0.9516** | **0.9527** | 0.9721 |
| DawnGNN (OneHot) | GAT + One-Hot Node Representations | 0.9490 | 0.9488 | 0.9491 | 0.9489 | 0.9705 |
| MalDetConv | CNN + BiGRU + Word2Vec (Sequence Model) | 0.9490 | 0.9496 | 0.9483 | 0.9488 | **0.9737** |
| DawnGNN (Word2Vec) | GAT + Word2Vec Node Representations | 0.9294 | 0.9296 | 0.9288 | 0.9292 | 0.9731 |
| MalDy | Feature Hashing + TF-IDF + Voting Ensemble | 0.5333 | 0.5357 | 0.5355 | 0.5332 | 0.5068 |

## 📈 Key Insights

1. **Semantic Augmentation Superiority:**
   DawnGNN (BERT) outperforms purely topological/one-hot encoding (+0.4 pp accuracy, +0.38 pp F1) and Word2Vec graph encoding (+2.35 pp accuracy, +2.35 pp F1), demonstrating that natural language functional documentation provides critical domain intelligence that complements graph structural connectivity.
2. **Graph vs. Sequence:**
   DawnGNN achieves higher overall accuracy and F1-score compared to sequential deep learning (MalDetConv CNN+BiGRU) by effectively capturing non-sequential API interaction dependencies.
3. **Graph vs. Classical ML:**
   Graph Neural Networks decisively outperform classical n-gram feature hashing ensembles (MalDy), which struggle with high-order API co-occurrence patterns.

## Files in this Directory

- `evaluation_summary.csv` — Machine-readable CSV containing raw precision floating-point metrics for all methods.
- `evaluation_summary.md` — Markdown table formatted for documentation reports.

## Regenerating Results & Visualizations

To re-aggregate test metrics from benchmark subdirectories or generate publication plots:

```bash
# Aggregate benchmark metrics
python scripts/aggregate_results.py

# Generate comparison plots (saved to results/figures/)
python scripts/plot_results.py
```

