# MalDy: Phát Hiện Mã Độc với Feature Engineering và Ensemble Learning

Phương pháp phát hiện mã độc **MalDy** sử dụng **Traditional Machine Learning** với quy trình xử lý đặc trưng và mô hình học tập hợp trên bộ dữ liệu MalBehavD-V1.

## 📋 Tổng Quan

MalDy sử dụng quy trình xử lý đặc trưng và mô hình học như sau:

1. **Đầu vào**: Chuỗi cuộc gọi API (API Call Sequences)
2. **Trích xuất đặc trưng**: Sử dụng N-grams để tạo các chuỗi con từ chuỗi API gốc
3. **Vector hóa**: Áp dụng Feature Hashing (Hashing Trick) để giảm chiều dữ liệu
4. **Đánh giá trọng số**: Kết hợp với TF-IDF để đánh giá trọng số
5. **Mô hình phân loại**: Sử dụng Ensemble Learning (VotingClassifier hoặc RandomForestClassifier)

## 🏗️ Pipeline

### Bước 1: Tạo N-grams
- Từ chuỗi API gốc, tạo các chuỗi con (N-grams)
- Ví dụ: N=3 → Unigrams, Bigrams, Trigrams

### Bước 2: Feature Hashing (HashingVectorizer)
- Áp dụng Hashing Trick để giảm chiều dữ liệu
- Xử lý được vocabulary lớn mà không cần lưu trữ toàn bộ

### Bước 3: TF-IDF Transformation (TfidfTransformer)
- Đánh giá trọng số cho các features đã hash
- Tăng trọng số cho các features quan trọng

### Bước 4: Ensemble Learning
- **VotingClassifier**: Kết hợp nhiều base models (RF, XGBoost, SVM)
- **RandomForestClassifier**: Ensemble của decision trees

## 🚀 Cách Sử Dụng

### Setup

```bash
cd benchmarks/maldy
pip install -r ../requirements.txt
```

### Chạy Pipeline

```bash
# Bước 1: Chuẩn bị dữ liệu (CSV → API sequences)
python src/preprocess.py --make-splits

# Bước 2: Trích xuất features (N-grams → Hashing → TF-IDF)
python src/extract_features.py

# Bước 3: Train ensemble model
python src/train_models.py

# Bước 4: Đánh giá model
python src/evaluate.py
```

## 📁 Cấu Trúc

```
benchmarks/maldy/
├── configs/
│   └── maldy.yaml        # Config cho feature extraction và ensemble
├── src/
│   ├── preprocess.py     # Bước 1: Chuẩn bị dữ liệu
│   ├── extract_features.py # Bước 2-3: N-grams + Hashing + TF-IDF
│   ├── train_models.py   # Bước 4: Train Ensemble Learning
│   ├── evaluate.py       # Đánh giá: Accuracy, Precision, Recall, F1, Confusion Matrix
│   └── utils.py          # Utility functions
├── models/               # Trained models và results
└── README.md
```

## ⚙️ Config

File `configs/maldy.yaml` chứa cấu hình:

### Features
```yaml
features:
  ngram_range: [1, 3]  # N-grams: Unigrams, Bigrams, Trigrams
  n_features: 10000     # Số lượng hash features
```

### Models
```yaml
models:
  random_forest:
    enabled: true
    n_estimators: 100
    # ... other params
```

### Ensemble
```yaml
ensemble:
  type: voting  # 'voting' or 'random_forest'
  voting: soft  # 'hard' or 'soft'
```

## 📊 Đánh Giá

Script `evaluate.py` tính toán và in ra các chỉ số:

- ✅ **Accuracy**: Độ chính xác tổng thể
- ✅ **Precision**: Độ chính xác (per-class và macro)
- ✅ **Recall**: Độ nhạy (per-class và macro)
- ✅ **F1-Score**: F1 score (per-class và macro)
- ✅ **Confusion Matrix**: Ma trận nhầm lẫn
- ✅ **AUC-ROC**: Area Under ROC Curve (nếu có)

### Ví Dụ Output

```
EVALUATION RESULTS: VotingClassifier (soft, rf, xgboost, svm)
============================================================
Accuracy:  0.6500
Precision: 0.6450 (macro)
Recall:    0.6500 (macro)
F1-Score:  0.6475 (macro)
AUC-ROC:   0.7200

Confusion Matrix:
                Predicted
              Benign  Malware
Actual Benign     120      14
      Malware      15     106
```

## 🔧 Thư Viện Sử Dụng

- **scikit-learn**: HashingVectorizer, TfidfTransformer, VotingClassifier, RandomForestClassifier
- **pandas**: Xử lý dữ liệu CSV
- **numpy**: Xử lý arrays
- **joblib**: Lưu/load models và pipelines

## 📈 Phương Pháp Kỹ Thuật

### 1. N-grams
Tạo các chuỗi con từ chuỗi API:
- **Unigrams (1-gram)**: Từng API call riêng lẻ
- **Bigrams (2-gram)**: Cặp API calls liên tiếp
- **Trigrams (3-gram)**: Bộ 3 API calls liên tiếp

### 2. Feature Hashing
- Sử dụng `HashingVectorizer` từ scikit-learn
- Giảm chiều dữ liệu từ vocabulary lớn xuống số lượng features cố định
- Không cần lưu trữ vocabulary, tiết kiệm memory

### 3. TF-IDF Transformation
- Sử dụng `TfidfTransformer` để đánh giá trọng số
- Tăng trọng số cho các features quan trọng
- Giảm trọng số cho các features phổ biến

### 4. Ensemble Learning
- **VotingClassifier**: Kết hợp predictions từ nhiều base models
  - Soft voting: Trung bình probabilities
  - Hard voting: Majority vote
- **RandomForestClassifier**: Ensemble của decision trees

## 💡 Ưu Điểm

- ✅ Không cần GPU
- ✅ Training nhanh
- ✅ Dễ interpret
- ✅ Xử lý được vocabulary lớn (nhờ Hashing)
- ✅ Ensemble learning cải thiện performance
- ✅ Phù hợp cho baseline comparison

## 📊 Kết Quả

Sau khi chạy pipeline:

- **Features**: `data/maldy/features/` (train_features.npy, val_features.npy, test_features.npy)
- **Pipeline**: `data/maldy/features/feature_pipeline.pkl` (HashingVectorizer + TfidfTransformer)
- **Model**: `models/maldy/ensemble_model.pkl`
- **Results**: `models/maldy/evaluation_results.json`, `test_evaluation.json`

## 🔗 Liên Kết

- Data: Dùng chung với `../data/`
- Splits: Dùng chung với `../data/splits/`

## 📝 Lưu Ý

- Pipeline sử dụng **HashingVectorizer + TfidfTransformer** (không phải TfidfVectorizer)
- Đây là phương pháp chính xác theo mô tả kỹ thuật của MalDy
- Feature Hashing giúp xử lý vocabulary lớn mà không cần lưu trữ toàn bộ
