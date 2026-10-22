# DawnGNN Benchmark Suite

This directory contains the implementations, configurations, and evaluation results of the **4 baseline and ablation methods** compared against **DawnGNN (BERT)**.

---

## 📊 Summary of Benchmarked Methods

| Method | Type | Description | Directory |
| :--- | :--- | :--- | :--- |
| **DawnGNN (BERT)** | Proposed | Graph Attention Network + Win32 Doc Embeddings | `src/` (root) |
| **DawnGNN (OneHot)** | Ablation | GAT with One-Hot node representations | [`ablations/onehot/`](./ablations/onehot/) |
| **DawnGNN (Word2Vec)** | Ablation | GAT with pre-trained Word2Vec embeddings | [`ablations/word2vec/`](./ablations/word2vec/) |
| **MalDetConv** | Baseline | Sequential CNN + BiGRU architecture (Word2Vec) | [`maldetconv/`](./maldetconv/) |
| **MalDy** | Baseline | Dynamic feature extraction + Voting Classifier (RF, XGBoost, SVM) | [`maldy/`](./maldy/) |

---

## 🏆 Comparative Results (Test Set, N = 255)

| Method | Accuracy | F1-Score (Macro) | Precision (Macro) | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: |
| **DawnGNN (BERT)** *(Ours)* | **0.9529** | **0.9527** | **0.9548** | **0.9721** |
| DawnGNN (OneHot) | 0.9490 | 0.9489 | 0.9488 | 0.9705 |
| MalDetConv | 0.9490 | 0.9488 | 0.9496 | **0.9737** |
| DawnGNN (Word2Vec) | 0.9294 | 0.9292 | 0.9296 | 0.9731 |
| MalDy | 0.5333 | 0.5332 | 0.5357 | 0.5068 |

---

## 📁 Structure

```
benchmarks/
├── README.md                          # This file
├── requirements.txt                   # Consolidated benchmark dependencies
├── maldetconv/                        # CNN + BiGRU baseline
│   ├── configs/                       # Hyperparameter YAMLs
│   ├── src/                           # Preprocessing, dataset, model, training
│   ├── models/test_evaluation.json    # Evaluation metrics
│   └── README.md
├── maldy/                             # Classical ensemble baseline
│   ├── configs/
│   ├── src/                           # Dynamic feature extraction & training
│   ├── models/test_evaluation.json
│   └── README.md
└── ablations/
    ├── onehot/                        # DawnGNN without semantic augmentation
    │   ├── configs/
    │   ├── src/
    │   └── models/test_evaluation.json
    └── word2vec/                      # DawnGNN with Word2Vec representations
        ├── configs/
        ├── src/
        └── models/test_evaluation.json
```

To aggregate all benchmark results into a single comparison table, run from repository root:
```bash
python scripts/aggregate_results.py
python scripts/plot_results.py
```
