# Thông Tin Về Git Configuration

## 📁 File `.gitignore` Đã Được Tạo

### ✅ Các File ĐƯỢC PUSH lên Git (tracked):

#### 1. **Code Python** (.py files)
- Tất cả các file Python script
- Ví dụ: `model_comparison.py`, `ablation_study.py`, etc.

#### 2. **File Kết Quả Phân Tích Nhỏ** (CSV nhỏ, < 100KB)
- `component_ablation_results.csv` (294 bytes)
- `detailed_metrics_results.csv` (478 bytes)
- `feature_ablation_results.csv` (3.5 KB)
- `final_metrics_table.csv` (305 bytes)
- `shap_feature_importance.csv` (659 bytes)
- `shap_feature_correlations.csv` (854 bytes)
- `traditional_papers_analysis.csv` (1.3 KB)
- `real_huggingface_papers_analysis.csv` (1.3 KB)

#### 3. **File Hình Ảnh & Visualization**
- Tất cả file `.png` (biểu đồ, kết quả)
- Ví dụ: `ablation_study_results.png`, `final_report_visualization.png`, etc.

#### 4. **File JSON**
- `final_report.json`
- `thesis_structure_50_pages.json`
- `dataset_sources_report.json`
- etc.

#### 5. **README và Documentation**
- `README.md`

---

### ❌ Các File KHÔNG PUSH lên Git (gitignored):

#### 1. **Dataset Lớn** (quá nặng cho Git)
- `dataset/2019-Oct.csv` - **460 MB** ⚠️ (Kaggle dataset gốc)
- `cosmetics_dataset.csv` - **5.5 MB**
- `cosmetics_processed.csv` - **9.6 MB**
- `real_cosmetics_dataset.csv` - **8.8 MB**
- `processed_data.csv` - (file xử lý)

#### 2. **Model Files** (.pkl - Machine Learning models)
- `best_model_xgboost.pkl` - **339 KB**
- `best_model_lightgbm.pkl` - **348 KB**
- `best_model_random_forest.pkl` - **8.1 MB** ⚠️
- `best_model_logistic_regression.pkl` - **1 KB**
- `scaler.pkl` - **1.9 KB**

#### 3. **Lý do KHÔNG push:**
- ⚠️ **Quá lớn**: GitHub giới hạn file < 100MB
- 🔄 **Có thể tạo lại**: Dataset và model có thể train lại từ code
- 💾 **Tiết kiệm dung lượng**: Repository sẽ nhẹ hơn, clone nhanh hơn

---

## 📝 Hướng Dẫn Sử Dụng

### Khi Clone Repository
```bash
git clone <your-repo-url>
cd phuc-doan-totnghiep
```

### Download Dataset Gốc
Vì file `2019-Oct.csv` không có trên Git, bạn cần download từ Kaggle:
- **Link**: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
- **Đặt vào**: `dataset/2019-Oct.csv`

### Tạo Lại Dataset Mỹ Phẩm
```bash
python download_cosmetics_dataset.py
```

### Train Lại Model
```bash
python model_comparison.py
```

---

## 🔧 Cấu Trúc `.gitignore`

```gitignore
# Dataset Files (lớn)
dataset/2019-Oct.csv
cosmetics_dataset.csv
cosmetics_processed.csv
real_cosmetics_dataset.csv

# Model Files (pkl)
*.pkl

# Nhưng GIỮ LẠI các file CSV phân tích nhỏ
!final_metrics_table.csv
!component_ablation_results.csv
# ... etc
```

---

## 📊 Tổng Kết

| Loại File | Push lên Git? | Lý do |
|-----------|---------------|-------|
| Code (.py) | ✅ Yes | Code nguồn quan trọng |
| Dataset lớn (.csv > 1MB) | ❌ No | Quá nặng |
| Model (.pkl) | ❌ No | Có thể train lại |
| Kết quả phân tích nhỏ (.csv < 100KB) | ✅ Yes | Quan trọng & nhẹ |
| Hình ảnh (.png) | ✅ Yes | Visualization quan trọng |
| JSON reports | ✅ Yes | Kết quả nghiên cứu |

---

## 💡 Tips

1. **Nếu cần chia sẻ dataset**: Sử dụng Google Drive, Kaggle, hoặc dịch vụ khác
2. **Nếu cần chia sẻ model**: Sử dụng model registry (MLflow, Hugging Face, etc.)
3. **Git LFS**: Nếu cần push file lớn, cân nhắc dùng Git Large File Storage

---

Tạo bởi: Git Configuration Script
Ngày: 2025-10-22

