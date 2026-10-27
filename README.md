# DawnGNN — Windows Malware Detection via Documentation-Augmented GNN

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cose.2024.103788-blue.svg)](https://doi.org/10.1016/j.cose.2024.103788)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!NOTE]
> **Independent Implementation / Reproduction:**
> This repository provides an independent open-source PyTorch / PyG reproduction of the research paper:  
> **"DawnGNN: Documentation augmented windows malware detection using graph neural network"**  
> *Authors:* Pengbin Feng, Le Gai, Li Yang, Qin Wang, Teng Li, Ning Xi, Jianfeng Ma  
> *Publication:* **Computers & Security** (Volume 140, 103788), Elsevier, May 2024  
> 🔗 **DOI:** [10.1016/j.cose.2024.103788](https://doi.org/10.1016/j.cose.2024.103788)
>
> An end-to-end Machine Learning pipeline that transforms Windows API call sequences into semantically-enriched graphs and classifies them as **Malware** or **Benign** using a Graph Attention Network (GAT).

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Benchmark Results](#benchmark-results)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [REST API & CLI Inference](#rest-api--cli-inference)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting & FAQs](#troubleshooting--faqs)
- [Project Structure](#project-structure)
- [End-to-End Checklist](#end-to-end-checklist)
- [Citation & References](#citation--references)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Traditional malware detection models treat API call sequences as categorical indices (One-Hot) or statistical co-occurrences (Word2Vec), missing the critical domain semantics of what each API actually performs.

**DawnGNN** bridges this gap through a multi-stage approach:
1. **Markovian State Transition Modeling**: Converts raw temporal API traces into directed, weighted call graphs.
2. **Semantic Augmentation**: Enriches each API node with a 768-dimensional continuous representation extracted from natural language Windows API documentation via BERT (`bert-base-uncased`).
3. **Graph Attention Network (GAT)**: Employs multi-head attention to dynamically weight transitions between critical system operations.

| Component | Technology | Description |
|---|---|---|
| **Graph Construction** | NetworkX DiGraph | Directed transition graph with frequency edge weights |
| **Semantic Embeddings** | BERT (`bert-base-uncased`) | 768-d `[CLS]` token vectors of Win32 documentation |
| **Classifier** | PyG `GATClassifier` | 3-layer GAT with multi-head attention & global mean pooling |
| **Dataset** | MalBehavD-V1 | 2,571 samples (binary: Malware / Benign) |
| **Serving** | FastAPI + Uvicorn | Production REST API with OpenAPI `/docs` & batch inference |

---

## Architecture

```
API Call Sequence (CSV)
        │
        ▼
  ┌─────────────┐
  │ csv_to_     │  Directed weighted graph per sample
  │ graphs.py   │  (nodes = API names, edges = call transitions)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ crawl_      │  Win32 API documentation from Microsoft Learn
  │ api.py      │  → api_docs.json  (pre-built, skip if already present)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ embed_      │  BERT [CLS] 768-d embeddings per API name
  │ api.py      │  → data/embeddings/*.npy
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ augment_    │  Assign BERT node features to graphs
  │ graphs.py   │  → data/graphs_augmented/*.gpickle
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐       ┌─────────────────────────────────────┐
  │ train_      │       │  GATClassifier                      │
  │ gnn.py      │──────▶│  • 3 × GATConv layers (8 heads)     │
  └──────┬──────┘       │  • Global Readout Pooling (Mean)    │
         │              │  • Linear Classifier (Dropout 0.4)  │
         ▼              └─────────────────────────────────────┘
  ┌─────────────┐
  │ evaluate.py │  Accuracy (95.29%), F1, AUC-ROC, Confusion Matrix
  └─────────────┘
```

---

## Benchmark Results

Evaluated on the held-out test split of **MalBehavD-V1** ($N = 255$ samples, 80/10/10 partition, random seed 42):

| Method | Architecture / Representation | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | AUC-ROC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **DawnGNN (BERT)** ⭐ | **GAT + Win32 Doc Semantic Embeddings** | **0.9529** | **0.9548** | **0.9516** | **0.9527** | 0.9721 |
| DawnGNN (OneHot) | GAT + One-Hot Node Representations | 0.9490 | 0.9488 | 0.9491 | 0.9489 | 0.9705 |
| MalDetConv | CNN + BiGRU + Word2Vec (Sequence Model) | 0.9490 | 0.9496 | 0.9483 | 0.9488 | **0.9737** |
| DawnGNN (Word2Vec) | GAT + Word2Vec Node Representations | 0.9294 | 0.9296 | 0.9288 | 0.9292 | 0.9731 |
| MalDy | Feature Hashing + TF-IDF + Voting Ensemble | 0.5333 | 0.5357 | 0.5355 | 0.5332 | 0.5068 |

> Full machine-readable metrics: [`results/evaluation_summary.csv`](results/evaluation_summary.csv)  
> Confusion Matrix on Test Set: $TN = 131, FP = 3, FN = 9, TP = 112$ (Support: 134 Benign, 121 Malware, Total: 255).

**Key finding**: Augmenting the call graph with BERT semantic embeddings (+0.4 pp accuracy vs. One-Hot, +2.4 pp vs. Word2Vec) validates that *domain knowledge from documentation* complements *graph structural connectivity*.

---

## System Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **Python**: 3.10+ (Python 3.8+ compatible)
- **Memory (RAM)**: Minimum 8GB (16GB recommended)
- **GPU (Optional)**: NVIDIA GPU with CUDA 11.8 or 12.1 (CPU execution fully supported)
- **Disk Storage**: At least 5GB free space

---

## Installation

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/NhatWoan-20/DawnGNN.git
cd DawnGNN

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 2. Install PyTorch (Hardware-Specific)

Check your CUDA version using `nvidia-smi` (if GPU is present):

```bash
# GPU — CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# GPU — CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision torchaudio
```

### 3. Install PyTorch Geometric (PyG)

Install PyG matching your PyTorch and CUDA versions:

```bash
# Torch 2.2.0 + CUDA 12.1:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch_geometric

# Torch 2.2.0 + CUDA 11.8:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch_geometric

# CPU only:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch_geometric
```

### 4. Install DawnGNN Package

Install the DawnGNN package in editable mode with development and serving dependencies:

```bash
# Complete installation (Core ML + Serving + Dev tools)
pip install -e .[dev,serve]

# Or base installation only:
pip install -e .
```

---

## Dataset Preparation

The repository includes the pre-processed benchmark dataset and Win32 API documentation:
- `data/raw/MalBehavD-V1-dataset.csv`: 2,571 samples (`sha256,labels,0,1,2...`).
- `data/api_docs/api_docs.json`: Pre-crawled functional documentation for 293 Win32 APIs.

To validate the dataset integrity without modifying existing data:
```bash
python -m dawngnn.pipelines.preprocess
```

---

## Pipeline Walkthrough

Run all commands from the repository root (`DawnGNN/`).

### Step 1 — Build Raw API Call Graphs
Converts chronological API traces into individual directed, weighted NetworkX graphs and partitions the dataset into reproducible 80/10/10 train/val/test splits.

```bash
python -m dawngnn.pipelines.csv_to_graphs --make-splits --train-frac 0.8 --val-frac 0.1 --seed 42
```
* **Output artifacts**:
  * `data/graphs/<sha256>.gpickle`
  * `data/graphs/graphs_manifest.json`
  * `data/intermediate/labels.json`
  * `data/splits/{train,val,test}.json`

### Step 2 — Crawl API Documentation *(Optional — Pre-built)*
> [!NOTE]
> `data/api_docs/api_docs.json` is already provided in the repository. Skip this step unless you wish to refresh documentation from Microsoft Learn.

```bash
python -m dawngnn.pipelines.crawl_api
```

### Step 3 — Generate BERT Semantic Embeddings
Extracts 768-dimensional `[CLS]` token continuous embeddings from `bert-base-uncased` for each documented API function:

```bash
python -m dawngnn.pipelines.embed_api
```
* **Output artifacts**: `data/embeddings/<api_name>.npy` (768-d float32 vectors).

### Step 4 — Semantic Graph Augmentation
Assigns the 768-dimensional BERT embedding as node feature tensor `x` for each node in the NetworkX graphs (with zero-vector fallback for undocumented APIs):

```bash
python -m dawngnn.pipelines.augment_graphs
```
* **Output artifacts**: `data/graphs_augmented/<sha256>.gpickle`.

### Step 5 — Train the GAT Classifier
Trains the 3-layer Graph Attention Network with multi-head attention, Adam optimizer, `ReduceLROnPlateau` scheduler, and early stopping:

```bash
python -m dawngnn.pipelines.train_gnn
```
* **Configuration**: Configurable in `configs/gnn.yaml`.
* **Output artifacts**:
  * `models/gnn/best_model.pt` (Best model weights based on validation accuracy)
  * `models/gnn/training_history.json` (Epoch-by-epoch loss and metrics)

### Step 6 — Evaluate on the Test Set
Evaluates the best checkpoint on the held-out test split ($N = 255$ samples):

```bash
python -m dawngnn.pipelines.evaluate
```
* **Output artifacts**: `models/gnn/test_evaluation.json`.
* **Console output**: Comprehensive metrics report (Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix).

---

## REST API & CLI Inference

### 1. Launch FastAPI Server

Start the high-performance inference server with automatic reloading:

```bash
uvicorn dawngnn.serving.api_server:app --host 0.0.0.0 --port 8000 --reload
```

* Interactive OpenAPI / Swagger UI: **http://localhost:8000/docs**
* Health check: `curl http://localhost:8000/health`
* Model info: `curl http://localhost:8000/model/info`

### 2. Single-Sample Prediction via HTTP

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "api_sequence": [
      "NtCreateFile", "NtWriteFile", "VirtualAllocEx",
      "WriteProcessMemory", "CreateRemoteThread", "NtClose"
    ],
    "top_k": 5
  }'
```

**JSON Response:**
```json
{
  "label": 1,
  "label_name": "Malware",
  "malware_probability": 0.9823,
  "benign_probability": 0.0177,
  "unique_apis": 6,
  "total_calls": 6,
  "graph_stats": {"nodes": 6, "edges": 5},
  "top_apis": [
    {"api": "NtCreateFile", "degree": 2},
    {"api": "WriteProcessMemory", "degree": 2}
  ],
  "latency_ms": 12.4
}
```

### 3. Command-Line (CLI) Inference

Predict directly from the terminal without starting the REST API:

```bash
# Pass sequence directly:
python -m dawngnn.serving.infer \
  --api-sequence "NtCreateFile,NtWriteFile,VirtualAllocEx,WriteProcessMemory,CreateRemoteThread,NtClose"

# Or pass a text file containing one API per line:
python -m dawngnn.serving.infer --api-sequence-file sample_trace.txt
```

---

## Testing

Run unit tests directly with `pytest`:

```bash
# Run all unit tests
pytest tests/ -v

# Run with test coverage report
pytest tests/ -v --cov=dawngnn --cov-report=html

# Run specific test suites
pytest tests/test_graph_utils.py -v
pytest tests/test_model.py -v
pytest tests/test_dataset.py -v
pytest tests/test_preprocessing.py -v
```

*Note: Tests requiring PyG are automatically skipped if PyTorch Geometric is not installed.*

---

## Configuration

Hyperparameters and paths are centrally managed in `configs/`:

* **`configs/gnn.yaml`** — GAT architecture & training:
  ```yaml
  model:
    input_dim: 768        # BERT hidden dimension
    hidden_dim: 128
    num_layers: 3
    num_heads: 8
    dropout: 0.4
    pooling_type: "mean"  # "mean", "sum", or "max"
  training:
    learning_rate: 0.0005
    weight_decay: 0.0005
    epochs: 100
    batch_size: 32
    early_stopping:
      enabled: false
      patience: 20
    lr_scheduler:
      enabled: true
      factor: 0.3
      patience: 5
  ```

* **`configs/bert.yaml`** — Semantic embedding model:
  ```yaml
  model:
    name: "bert-base-uncased"
    max_length: 256
  ```

* **`configs/data.yaml`** — Dataset paths and split ratios:
  ```yaml
  splits:
    train_frac: 0.8
    val_frac: 0.1
    seed: 42
  ```

---

## Troubleshooting & FAQs

### 1. `CUDA out of memory`
* **Solution**: Decrease `batch_size` in `configs/gnn.yaml` (e.g., from `32` to `16` or `8`). Alternatively, set `device: "cpu"` under `training` in `configs/gnn.yaml`.

### 2. `ModuleNotFoundError: No module named 'dawngnn'`
* **Solution**: Ensure your virtual environment is activated and reinstall the package in editable mode:
  ```bash
  pip install -e .[dev,serve]
  ```

### 3. `FileNotFoundError: CSV not found`
* **Solution**: Ensure all commands are executed from the **project root directory** (`DawnGNN/`), not from within `src/` or `pipelines/`.

### 4. `Empty graph error`
* **Solution**: Ensure the input trace contains at least one valid, non-empty Windows API call name.

---

## Project Structure

```
DawnGNN/
├── benchmarks/              # Baseline models & ablation studies
│   ├── requirements.txt     # Consolidated benchmark dependencies
│   ├── maldetconv/          # CNN + BiGRU baseline
│   ├── maldy/               # Feature hashing + Voting ensemble baseline
│   ├── ablations/
│   │   ├── onehot/          # GAT with One-Hot node representations
│   │   └── word2vec/        # GAT with Word2Vec node representations
│   └── README.md
├── configs/                 # YAML configuration files
│   ├── gnn.yaml             # GAT hyperparameters & training config
│   ├── bert.yaml            # BERT model selection & tokenizer config
│   └── data.yaml            # Dataset paths & split configuration
├── data/
│   ├── raw/                 # MalBehavD-V1-dataset.csv (2,571 samples)
│   ├── api_docs/            # Win32 API documentation (api_docs.json)
│   ├── intermediate/        # Sample label mappings (labels.json)
│   └── splits/              # Reproducible train / val / test JSON splits
├── models/
│   └── gnn/
│       ├── best_model.pt    # Pre-trained checkpoint (95.29% Accuracy)
│       ├── test_evaluation.json
│       └── training_history.json
├── notebooks/
│   └── demo.ipynb           # Interactive 7-step Jupyter demo notebook
├── results/
│   ├── evaluation_summary.csv # Benchmark metrics across 5 methods
│   ├── evaluation_summary.md  # Formatted Markdown report
│   └── README.md
├── scripts/
│   ├── aggregate_results.py # Metric aggregator script
│   └── plot_results.py      # Publication-ready visualizations
├── src/
│   ├── models/              # PyTorch Geometric model definitions
│   │   ├── __init__.py
│   │   ├── gat.py           # Modular GATClassifier
│   │   └── dataset.py       # MalwareGraphDataset loader
│   ├── pipelines/           # End-to-end data & training pipeline
│   │   ├── preprocess.py    # Dataset validation & split generator
│   │   ├── crawl_api.py     # Microsoft Learn API docs scraper
│   │   ├── csv_to_graphs.py # Trace CSV to NetworkX call graphs
│   │   ├── embed_api.py     # BERT semantic embedding generator
│   │   ├── augment_graphs.py# Attaching embeddings to graph nodes
│   │   ├── train_gnn.py     # Training loop with early stopping & scheduler
│   │   └── evaluate.py      # Test set evaluation & confusion matrix
│   ├── serving/
│   │   ├── api_server.py    # FastAPI REST server with OpenAPI /docs
│   │   └── infer.py         # Standalone CLI & library inference
│   └── utils/
│       ├── paths.py         # Dynamic root path management
│       ├── logger.py        # Centralized logging
│       └── graph_utils.py   # NetworkX graph operations
├── tests/
│   ├── conftest.py          # Pytest path fixtures
│   ├── test_dataset.py      # Unit tests for Dataset loader
│   ├── test_graph_utils.py  # Unit tests for graph construction
│   ├── test_model.py        # Unit tests for GATClassifier
│   └── test_preprocessing.py# Unit tests for data preprocessing
├── pyproject.toml           # Package build metadata
├── requirements.txt         # Core dependencies
└── README.md                # Project documentation & complete guide
```

---

## End-to-End Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] PyTorch & PyTorch Geometric installed for your hardware
- [ ] DawnGNN package installed: `pip install -e .[dev,serve]`
- [ ] Raw dataset verified: `data/raw/MalBehavD-V1-dataset.csv`
- [ ] Step 1: `python -m dawngnn.pipelines.csv_to_graphs --make-splits`
- [ ] Step 2: `python -m dawngnn.pipelines.embed_api`
- [ ] Step 3: `python -m dawngnn.pipelines.augment_graphs`
- [ ] Step 4: `python -m dawngnn.pipelines.train_gnn`
- [ ] Step 5: `python -m dawngnn.pipelines.evaluate` (Target: ~95.29% Accuracy)
- [ ] REST API verified: `uvicorn dawngnn.serving.api_server:app --port 8000 --reload`
- [ ] Unit tests passing: `pytest tests/ -v`

---

## 📖 Citation & References

If you find this repository, implementation, or the original methodology useful in your academic research or projects, please cite the original research paper:

```bibtex
@article{feng2024dawngnn,
  title={DawnGNN: Documentation augmented windows malware detection using graph neural network},
  author={Feng, Pengbin and Gai, Le and Yang, Li and Wang, Qin and Li, Teng and Xi, Ning and Ma, Jianfeng},
  journal={Computers \& Security},
  volume={140},
  pages={103788},
  year={2024},
  month={may},
  publisher={Elsevier},
  issn={0167-4048},
  doi={10.1016/j.cose.2024.103788},
  url={https://doi.org/10.1016/j.cose.2024.103788}
}
```

**APA Citation:**
> Feng, P., Gai, L., Yang, L., Wang, Q., Li, T., Xi, N., & Ma, J. (2024). DawnGNN: Documentation augmented windows malware detection using graph neural network. *Computers & Security*, 140, 103788. https://doi.org/10.1016/j.cose.2024.103788

---

## 🙏 Acknowledgements

This project is an independent open-source reproduction built for educational and research evaluation purposes, based on the concepts and architecture proposed by Feng et al. in *Computers & Security* (Elsevier, 2024). All rights to the underlying scientific methodology belong to the original authors and Elsevier.

---

## License

This project is released under the [MIT License](LICENSE).
