# DawnGNN Data Directory Structure

This directory contains all data used in the DawnGNN pipeline.

## Directory Layout

```
data/
├── raw/                      # Raw input dataset (committed to repo)
│   └── MalBehavD-V1-dataset.csv # 2,570 samples (1,285 Benign, 1,285 Malware)
│
├── api_docs/                 # Win32 API documentation (committed to repo)
│   └── api_docs.json         # 293 documented Win32 APIs
│
├── splits/                   # Train/val/test splits (committed to repo)
│   ├── train.json            # 2,040 training SHA256 IDs (80%)
│   ├── val.json              # 255 validation SHA256 IDs (10%)
│   └── test.json             # 255 testing SHA256 IDs (10%)
│
├── intermediate/             # Intermediate label index (committed to repo)
│   └── labels.json           # SHA256 -> binary label (0: Benign, 1: Malware)
│
├── graphs/                   # Generated NetworkX DiGraphs (gitignored)
│   ├── <sha256>.gpickle      # Directed weighted API transition graph
│   └── graphs_manifest.json  # Map of SHA256 -> graph file path
│
├── graphs_augmented/         # BERT-augmented graphs (gitignored)
│   └── <sha256>.gpickle      # Graph with node attribute 'x' in R^768
│
└── embeddings/               # Pre-computed BERT API embeddings (gitignored)
    ├── NtCreateFile.npy      # 768-d numpy float32 embedding
    ├── NtWriteFile.npy
    └── ...
```

## Data Pipeline Flow & Transformation Logic

The DawnGNN data engineering pipeline converts raw tabular API traces into semantically enriched graph tensors through a deterministic 4-stage pipeline:

```
MalBehavD-V1-dataset.csv (2,570 raw rows)
         │
         │  [csv_to_graphs.py]
         ├──▶ Filter 27 empty/NaN traces ──▶ 2,543 valid samples
         ├──▶ Build Markov transition graphs ──▶ data/graphs/<sha256>.gpickle
         ├──▶ Extract ground-truth labels ──▶ data/intermediate/labels.json (2,543 entries)
         └──▶ Partition 80/10/10 splits ──▶ data/splits/{train,val,test}.json
```

---

### Step 1: Raw Data & Preprocessing (`csv_to_graphs.py`)

1. **Input Data** (`raw/MalBehavD-V1-dataset.csv`):
   - **Origin**: 2,570 execution traces gathered via automated **Cuckoo Sandbox** dynamic analysis (1,285 Benign, 1,285 Malware).
   - **Format**: `sha256,labels,0,1,2,...` where each column from index 2 onwards represents an API called in chronological order.

2. **Data Cleaning & Pruning (2,570 rows $\rightarrow$ 2,543 graphs)**:
   - During parsing, strings are stripped, and blank cells or `NaN`/`None` placeholders are discarded.
   - **Why 2,543 instead of 2,570?** Exactly **27 samples** in the raw CSV contained no valid API calls (empty execution runs or failed sandbox hook captures). The pipeline safely prunes these invalid rows (`if len(api_calls) == 0: continue`) to prevent training on empty disconnected graphs.
   - **Post-cleaning balance**: 1,276 Benign vs 1,267 Malware (maintaining a nearly perfect 50.2% : 49.8% balance).

3. **Markovian Graph Construction**:
   - For each valid sample, consecutive API calls $(API_i, API_{i+1})$ are transformed into a **directed edge** $u \rightarrow v$ in a NetworkX `DiGraph`.
   - Repeated transitions increment the edge attribute `weight` (frequency count), capturing loop iterations and execution intensity.
   - Graph structures are serialized as binary pickle files: `data/graphs/<sha256>.gpickle`.

4. **Ground-Truth Label Indexing (`intermediate/labels.json`)**:
   - Stores a lightweight hash map: `{"<sha256>": 0 | 1}` (where `0` = Benign, `1` = Malware).
   - **Design Purpose**: Acts as an $O(1)$ in-memory label lookup cache. During PyTorch DataLoader training, workers query `labels.json` instantly by sample ID instead of repeatedly loading and parsing the 2.3 MB raw CSV file, eliminating disk I/O bottlenecks.

5. **Reproducible Partitioning (`splits/`)**:
   - When run with `--make-splits`, partitions the 2,543 valid sample IDs into reproducible splits (seed 42):
     - `splits/train.json`: **2,034 samples** (80%)
     - `splits/val.json`: **254 samples** (10%)
     - `splits/test.json`: **255 samples** (10%)

---

### Step 2: Semantic Documentation Extraction (`crawl_api.py` & `embed_api.py`)

1. **Win32 API Documentation** (`api_docs/api_docs.json`):
   - Contains natural language functional summaries for 293 unique Win32 APIs scraped from Microsoft Learn.
2. **BERT Node Embeddings** (`embeddings/<api_name>.npy`):
   - Passes each API's documentation through `bert-base-uncased` to extract the 768-dimensional `[CLS]` token continuous representation.
   - Captures latent system intent (e.g., distinguishing memory allocation from process injection).

---

### Step 3: Graph Semantic Augmentation (`augment_graphs.py`)

- Ingests `data/graphs/<sha256>.gpickle` and binds the 768-d BERT vector as the node feature tensor `x` for every node (API) in the graph.
- Unseen/undocumented APIs automatically receive a zero-vector fallback.
- Outputs production-ready augmented graphs to `data/graphs_augmented/<sha256>.gpickle` for direct consumption by the PyG `GATClassifier`.

## Size Information

| Directory | Size | Count | Committed |
|-----------|------|-------|-----------|
| `raw/` | ~2.3 MB | 1 file | ✅ Yes |
| `api_docs/` | ~330 KB | 1 file | ✅ Yes |
| `splits/` | ~180 KB | 3 files | ✅ Yes |
| `intermediate/` | ~170 KB | 1 file | ✅ Yes |
| `graphs/` | ~15 MB | 2,543 files | ❌ No (Gitignored) |
| `embeddings/` | ~1.5 MB | ~300 files | ❌ No (Gitignored) |
| `graphs_augmented/` | ~50 MB | 2,543 files | ❌ No (Gitignored) |

## Regenerating Data

To regenerate all data from scratch:

```bash
# Step 1: Build raw graphs + splits
python -m dawngnn.pipelines.csv_to_graphs --make-splits

# Step 2: Generate BERT embeddings (uses data/api_docs/api_docs.json)
python -m dawngnn.pipelines.embed_api

# Step 3: Augment graphs with embeddings
python -m dawngnn.pipelines.augment_graphs
```

## Notes

- Generated directories (`graphs/`, `embeddings/`, `graphs_augmented/`) are gitignored
- Only raw data and metadata are committed to the repository
- Pipeline scripts automatically create missing directories
- Generated data is ~1.3 GB total (excluded from repo)

## References

1. **MalBehavD-V1 Dataset**: Maniriho, P., Mahmood, A. N., & Chowdhury, M. J. M. (2023). *API-MalDetect: Automated malware detection framework for Windows based on API calls and deep learning techniques*. Journal of Network and Computer Applications, 218, 103704. [DOI: 10.1016/j.jnca.2023.103704](https://doi.org/10.1016/j.jnca.2023.103704) | [GitHub: mpasco/MalbehavD-V1](https://github.com/mpasco/MalbehavD-V1)
2. **DawnGNN Methodology**: Feng, P., Gai, L., Yang, L., Wang, Q., Li, T., Xi, N., & Ma, J. (2024). *DawnGNN: Documentation augmented windows malware detection using graph neural network*. Computers & Security, 140, 103788. [DOI: 10.1016/j.cose.2024.103788](https://doi.org/10.1016/j.cose.2024.103788)

