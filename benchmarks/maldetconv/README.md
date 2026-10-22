# MalDetConv: CNN + BiGRU với Word2Vec Embeddings

Phương pháp phát hiện mã độc sử dụng **CNN + Bidirectional GRU** với **Word2Vec embeddings**.

## 📋 Tổng Quan

MalDetConv xử lý API sequences như text và sử dụng:
- **Word2Vec** để tạo embeddings cho API calls (128D)
- **CNN** để trích xuất features cục bộ từ sequences
- **BiGRU** để capture dependencies dài hạn trong API sequences
- **Padding/Truncation** để chuẩn hóa độ dài sequences

## 🏗️ Pipeline

1. **preprocess.py** - Chuẩn bị dữ liệu (CSV → API sequences, padding)
2. **train_word2vec.py** - Train Word2Vec embeddings
3. **train_cnn_bigru.py** - Train CNN+BiGRU model
4. **evaluate.py** - Đánh giá trên test set

## 🚀 Cách Sử Dụng

### Setup

```bash
cd benchmarks/maldetconv
pip install -r ../requirements.txt
```

### Chạy Pipeline

```bash
# Bước 1: Chuẩn bị dữ liệu
python src/preprocess.py --make-splits

# Bước 2: Train Word2Vec
python src/train_word2vec.py

# Bước 3: Train CNN+BiGRU
python src/train_cnn_bigru.py

# Bước 4: Đánh giá
python src/evaluate.py
```

## 📁 Cấu Trúc

```
benchmarks/maldetconv/
├── configs/
│   └── cnn_bigru.yaml    # Config cho CNN+BiGRU
├── src/
│   ├── preprocess.py     # Data preprocessing
│   ├── train_word2vec.py # Word2Vec training
│   ├── model.py          # CNN+BiGRU architecture
│   ├── dataset.py        # PyTorch Dataset
│   ├── train_cnn_bigru.py # Training script
│   └── evaluate.py        # Evaluation script
├── models/               # Lưu checkpoint và test_evaluation.json
└── README.md
```

## ⚙️ Config

- `configs/cnn_bigru.yaml` - Model và training config

## 📊 Kết Quả

- Word2Vec: `data/word2vec_model.bin`, `data/word2vec_vectors.npy`
- Model: `models/maldetconv/best_model.pt`
- Evaluation: `models/maldetconv/test_evaluation.json`
- Training history: `models/maldetconv/training_history.json`

## 🔗 Liên Kết

- Data: Dùng chung với `../data/`
- Splits: Dùng chung với `../data/splits/`

