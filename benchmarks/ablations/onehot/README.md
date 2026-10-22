# DawnGNN-OneHot: Graph Neural Network với One-Hot Encoding

Phương pháp phát hiện mã độc sử dụng **GAT (Graph Attention Network)** với **One-Hot Encoding** cho node features.

## 📋 Tổng Quan

DawnGNN-OneHot chuyển đổi chuỗi API calls thành đồ thị và sử dụng:
- **Graph Neural Network (GAT)** để học từ cấu trúc đồ thị
- **One-Hot Encoding** để encode mỗi API thành vector one-hot
- **Vocabulary-based encoding** - mỗi API có một unique index trong vocabulary

## 🏗️ Pipeline

1. **csv_to_graphs.py** - Chuyển CSV → Đồ thị NetworkX
2. **generate_onehot_embeddings.py** - Tạo vocabulary và one-hot vectors
3. **augment_graphs.py** - Gán one-hot vectors vào node features
4. **train_gnn.py** - Train GAT model
5. **evaluate.py** - Đánh giá trên test set

## 🚀 Cách Sử Dụng

### Setup

```bash
cd benchmarks/ablations/onehot
pip install -r ../../requirements.txt
```

### Chạy Pipeline

```bash
# Bước 1: Tạo đồ thị
python src/pipelines/csv_to_graphs.py --make-splits

# Bước 2: Tạo one-hot embeddings (tạo vocabulary và vectors)
python src/pipelines/generate_onehot_embeddings.py

# Bước 3: Augment đồ thị với one-hot vectors
python src/pipelines/augment_graphs.py

# Bước 4: Train model
python src/pipelines/train_gnn.py

# Bước 5: Đánh giá
python src/pipelines/evaluate.py
```

## 📁 Cấu Trúc

```
benchmarks/ablations/onehot/
├── configs/
│   └── onehot.yaml          # Config cho One-Hot encoding và GNN training
├── src/
│   ├── pipelines/           # Pipeline scripts (augment, embed, train, evaluate)
│   └── utils/               # Utilities (paths, logger)
├── models/                   # Lưu model checkpoint và test_evaluation.json
└── README.md
```

## ⚙️ Config

- `configs/onehot.yaml` - One-Hot encoding và GNN training config

### Lưu Ý về Input Dimension

Sau khi chạy `generate_onehot_embeddings.py`, bạn cần cập nhật `input_dim` trong `configs/onehot.yaml` với vocabulary size. Hoặc script sẽ tự động đọc từ vocabulary file.

## 📊 Kết Quả

- Model: `models/gnn/best_model.pt`
- Evaluation: `models/gnn/test_evaluation.json`
- Training history: `models/gnn/training_history.json`
- Vocabulary: `data/embeddings_onehot/vocabulary.json`

## 🔗 So Sánh với DawnGNN

| Aspect | DawnGNN | DawnGNN-OneHot |
|--------|---------|----------------|
| **Embedding** | BERT (768D) | One-Hot (vocab_size) |
| **Vocabulary** | API docs | All unique APIs in dataset |
| **Dimension** | Fixed (768) | Variable (vocab_size) |
| **Semantic Info** | Yes (from docs) | No (just indices) |

## ⚠️ Lưu Ý

- One-Hot encoding có thể tạo ra vectors rất lớn nếu vocabulary size lớn
- Không có semantic information như BERT embeddings
- Input dimension phụ thuộc vào số lượng unique APIs trong dataset

