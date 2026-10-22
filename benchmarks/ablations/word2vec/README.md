# DawnGNN-Word2Vec: Graph Neural Network với Word2Vec Embeddings

Phương pháp phát hiện mã độc sử dụng **GAT (Graph Attention Network)** với **Word2Vec embeddings** cho node features.

## 📋 Tổng Quan

DawnGNN-Word2Vec chuyển đổi chuỗi API calls thành đồ thị và sử dụng:
- **Graph Neural Network (GAT)** để học từ cấu trúc đồ thị
- **Word2Vec embeddings** để encode mỗi API thành dense vector
- **Context-aware encoding** - học từ co-occurrence patterns trong API sequences

## 🏗️ Pipeline

1. **csv_to_graphs.py** - Chuyển CSV → Đồ thị NetworkX
2. **generate_word2vec_embeddings.py** - Train Word2Vec model và tạo embeddings
3. **augment_graphs.py** - Gán Word2Vec vectors vào node features
4. **train_gnn.py** - Train GAT model
5. **evaluate.py** - Đánh giá trên test set

## 🚀 Cách Sử Dụng

### Setup

```bash
cd benchmarks/ablations/word2vec
pip install -r ../../requirements.txt
```

### Chạy Pipeline

```bash
# Bước 1: Tạo đồ thị (optional, chỉ cần cho GNN training)
python src/pipelines/csv_to_graphs.py --make-splits

# Bước 2: Train Word2Vec và tạo embeddings
# Word2Vec đọc trực tiếp từ CSV để giữ thứ tự temporal của API calls
python src/pipelines/generate_word2vec_embeddings.py

# Bước 3: Augment đồ thị với Word2Vec vectors
python src/pipelines/augment_graphs.py

# Bước 4: Train model
python src/pipelines/train_gnn.py

# Bước 5: Đánh giá
python src/pipelines/evaluate.py
```

## 📁 Cấu Trúc

```
benchmarks/ablations/word2vec/
├── configs/
│   └── word2vec.yaml          # Config cho Word2Vec và GNN training
├── src/
│   ├── pipelines/             # Pipeline scripts (augment, embed, train, evaluate)
│   └── utils/                 # Utilities (paths, logger)
├── models/                    # Lưu model checkpoint và test_evaluation.json
└── README.md
```

## ⚙️ Config

- `configs/word2vec.yaml` - Word2Vec và GNN training config

### Word2Vec Parameters

- `vector_size`: Embedding dimension (default: 128)
- `window`: Context window size (default: 5)
- `min_count`: Minimum frequency (default: 1)
- `sg`: Training algorithm (0=CBOW, 1=skip-gram)
- `epochs`: Number of training epochs

## 📊 Kết Quả

- Model: `models/gnn/best_model.pt`
- Evaluation: `models/gnn/test_evaluation.json`
- Training history: `models/gnn/training_history.json`
- Word2Vec model: `data/embeddings_word2vec/word2vec.model`

## 🔗 So Sánh với DawnGNN

| Aspect | DawnGNN | DawnGNN-Word2Vec |
|--------|---------|------------------|
| **Embedding** | BERT (768D) | Word2Vec (128D) |
| **Training** | Pre-trained | Trained on dataset |
| **Semantic Info** | Yes (from docs) | Yes (from co-occurrence) |
| **Dimension** | Fixed (768) | Configurable (128) |

## ⚠️ Lưu Ý

- Word2Vec được train trên chính dataset của bạn
- Embeddings học từ co-occurrence patterns trong API sequences
- Có thể điều chỉnh `vector_size` trong config (thường 128-256)

