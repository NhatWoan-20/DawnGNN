# Evaluation Summary — 5 Malware Detection Methods

| Method | Description | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | AUC-ROC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **DawnGNN (BERT)** | GAT + BERT Embeddings (Semantic Augmentation) | **0.9529** | **0.9548** | **0.9516** | **0.9527** | 0.9721 |
| DawnGNN (OneHot) | GAT + One-Hot Encoding | 0.9490 | 0.9488 | 0.9491 | 0.9489 | 0.9705 |
| MalDetConv | CNN + BiGRU + Word2Vec Embeddings (Sequence-based) | 0.9490 | 0.9496 | 0.9483 | 0.9488 | **0.9737** |
| DawnGNN (Word2Vec) | GAT + Word2Vec Embeddings | 0.9294 | 0.9296 | 0.9288 | 0.9292 | 0.9731 |
| MalDy | VotingClassifier (soft, rf, xgboost, svm) | 0.5333 | 0.5357 | 0.5355 | 0.5332 | 0.5068 |

### Key Takeaways:
- **Highest Accuracy**: DawnGNN (BERT) (0.9529)
- **Highest F1-Score**: DawnGNN (BERT) (0.9527)
- **Highest AUC-ROC**: MalDetConv (0.9737), closely followed by DawnGNN (BERT) (0.9721)
