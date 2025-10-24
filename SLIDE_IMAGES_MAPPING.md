# MAPPING HÌNH ẢNH CHO SLIDES

## Tổng quan
- **Tổng số hình đã tạo:** 9 hình mới + 10 hình có sẵn = **19 hình**
- **Thư mục:** `slide_images/` (hình mới) và root folder (hình có sẵn)
- **Độ phân giải:** 300 DPI (chất lượng cao cho in ấn)

---

## HÌNH ẢNH MỚI TẠO (trong folder `slide_images/`)

### ✅ Slide 3: Đặt vấn đề
**File:** `slide_images/slide03_class_distribution.png`
- **Nội dung:** Pie chart class distribution
- **Data:** Purchase 5.96% vs Non-Purchase 94.04%
- **Highlight:** Imbalance ratio 15.78:1
- **Màu sắc:** Red (Purchase), Blue (Non-Purchase)

### ✅ Slide 4: Mục tiêu nghiên cứu
**File:** `slide_images/slide04_system_diagram.png`
- **Nội dung:** System architecture diagram
- **Boxes:** INPUT → SYSTEM → OUTPUT
- **System:** XGBoost + SMOTE + 24 Features
- **Output:** AUC 89.84%

### ✅ Slide 5: Hệ thống gợi ý
**File:** `slide_images/slide05_recommendation_types.png`
- **Nội dung:** 3 loại hệ thống gợi ý
- **Types:** Collaborative, Content-based, Hybrid
- **Highlight:** Hybrid System (đồ án này)
- **Arrows:** Pointing to Hybrid from both types

### ✅ Slide 6: XGBoost & SMOTE
**File:** `slide_images/slide06_xgboost_smote.png`
- **Nội dung:** 3 phần
  - Top: XGBoost ensemble (5 trees)
  - Bottom left: Before SMOTE (imbalanced)
  - Bottom right: After SMOTE (balanced)
- **Visualization:** Scatter plots showing balance

### ✅ Slide 7: Dataset
**File:** `slide_images/slide07_dataset_overview.png`
- **Nội dung:** 4 charts
  - Event type distribution (bar)
  - Class distribution (bar)
  - Dataset statistics (text box)
  - Class percentage (pie)
- **Layout:** 2x2 grid

### ✅ Slide 8: Feature Engineering
**File:** `slide_images/slide08_feature_engineering.png`
- **Nội dung:** 7 nhóm features
- **Groups:** Temporal, User Behavior, Product Info, Product Metrics, Interaction, Session Context, Encoded
- **Total:** 24 features
- **Design:** Colored boxes for each group

### ✅ Slide 9: Phương pháp nghiên cứu
**File:** `slide_images/slide09_methodology_flowchart.png`
- **Nội dung:** Pipeline 6 bước
- **Steps:** Preprocessing → Features → Split → SMOTE → Training → Evaluation
- **Arrows:** Connecting each step
- **Final result:** AUC 89.84%

### ✅ Slide 12: Cross-validation
**File:** `slide_images/slide12_cross_validation.png`
- **Nội dung:** 5-fold CV results
- **Chart type:** Bar chart
- **Data:** 5 folds với AUC scores
- **Mean line:** 89.84% (đỏ, dash)
- **Std dev:** ±0.10%

### ✅ Slide 19: Hướng phát triển
**File:** `slide_images/slide19_future_roadmap.png`
- **Nội dung:** Timeline roadmap
- **Phases:** Current → Phase 1 (DL) → Phase 2 (Real-time) → Phase 3 (Multi-domain)
- **Design:** Timeline với milestones
- **Bottom:** Feature items

---

## HÌNH ẢNH CÓ SẴN (trong root folder)

### ✅ Slide 10: So sánh Models
**File:** `model_selection_analysis.png` hoặc `final_report_visualization.png`
- **Nội dung:** Comparison của 4 models
- **Models:** Logistic Regression, Random Forest, LightGBM, XGBoost
- **Metrics:** AUC scores
- **Highlight:** XGBoost winner (89.84%)

### ✅ Slide 11: Feature Importance
**File:** `feature_importances.png`
- **Nội dung:** Top features by importance
- **Chart type:** Horizontal bar chart
- **Top feature:** cart_added_flag (28.47%)
- **Colors:** Gradient or highlighted top 3

### ✅ Slide 13: So sánh với Literature
**File:** `paper_comparison_detailed.png` hoặc `paper_comparison.png`
- **Nội dung:** Comparison table/chart
- **Papers:** LFDNN, XGBoost Purchase, Hybrid RF-LightFM, Đồ án
- **Metrics:** Dataset size, AUC, Imbalance
- **Highlight:** Đồ án có highest values

### ✅ Slide 14: ROC & PR Curves
**Files:** 
- `roc_curves_comparison.png` (ROC curves)
- `precision_recall_curves.png` (PR curves)
- **Nội dung:** Performance curves
- **ROC:** AUC 89.84%
- **PR:** Average Precision 78.2%

### ✅ Slide 15: Cross-domain Testing (Before)
**File:** `cosmetics_model_test_results.png`
- **Nội dung:** Full cosmetics dataset results
- **AUC:** 76.60% (drop from 89.84%)
- **Comparison:** Before vs After
- **Arrows:** Showing performance drop

### ✅ Slide 16: Cross-domain Testing (After)
**File:** `refined_cosmetics_test_results.png`
- **Nội dung:** Refined dataset results
- **AUC:** 95.29% (improvement!)
- **Products:** L'Oréal, Tarte
- **Highlight:** Green arrows showing improvement

### ✅ Slide 17: SHAP Analysis
**Files:**
- `shap_summary_plot.png` (recommended)
- `shap_bar_plot.png` (alternative)
- **Nội dung:** Feature contributions
- **Type:** SHAP summary plot
- **Features:** Top contributors highlighted

### ✅ Slide 18: Kết luận
**File:** `comprehensive_visual_summary.png`
- **Nội dung:** Overall summary
- **Achievements:** Checkmarks, metrics
- **Visual:** Infographic style
- **Key numbers:** All main results

### ✅ Backup Slide: Ablation Study
**File:** `ablation_study_results.png`
- **Nội dung:** Component analysis
- **Show:** Impact of each component
- **Data:** SMOTE, Features, Hyperparameters

---

## MAPPING NHANH (Quick Reference)

| Slide # | Tên Slide | File Name | Folder |
|---------|-----------|-----------|--------|
| 3 | Đặt vấn đề | slide03_class_distribution.png | slide_images/ |
| 4 | Mục tiêu | slide04_system_diagram.png | slide_images/ |
| 5 | Hệ thống gợi ý | slide05_recommendation_types.png | slide_images/ |
| 6 | XGBoost & SMOTE | slide06_xgboost_smote.png | slide_images/ |
| 7 | Dataset | slide07_dataset_overview.png | slide_images/ |
| 8 | Feature Engineering | slide08_feature_engineering.png | slide_images/ |
| 9 | Phương pháp | slide09_methodology_flowchart.png | slide_images/ |
| 10 | So sánh Models | model_selection_analysis.png | root |
| 11 | Feature Importance | feature_importances.png | root |
| 12 | Cross-validation | slide12_cross_validation.png | slide_images/ |
| 13 | So sánh Literature | paper_comparison_detailed.png | root |
| 14 | ROC/PR Curves | roc_curves_comparison.png + precision_recall_curves.png | root |
| 15 | Cross-domain (Before) | cosmetics_model_test_results.png | root |
| 16 | Cross-domain (After) | refined_cosmetics_test_results.png | root |
| 17 | SHAP Analysis | shap_summary_plot.png | root |
| 18 | Kết luận | comprehensive_visual_summary.png | root |
| 19 | Hướng phát triển | slide19_future_roadmap.png | slide_images/ |

---

## HƯỚNG DẪN SỬ DỤNG

### Cách 1: Copy từ 2 folders
```bash
# Copy tất cả hình từ slide_images vào PowerPoint
slide_images/*.png

# Copy các hình từ root folder
*.png (filter theo tên)
```

### Cách 2: Tổ chức lại (recommended)
1. Tạo folder mới: `presentation_images/`
2. Copy tất cả hình cần dùng vào đó
3. Đặt tên lại theo số slide (optional):
   ```
   slide03.png
   slide04.png
   slide05.png
   ...
   ```

### Khi tạo PowerPoint:
1. **Kích thước slide:** 16:9 (widescreen)
2. **Insert hình:** Insert > Picture > từ file
3. **Resize:** Giữ aspect ratio (Shift + kéo góc)
4. **Alignment:** Align center/middle
5. **Quality:** Compress pictures = high quality (220 PPI minimum)

### Tips:
- ✅ Sử dụng hình full width cho impact
- ✅ Add captions/labels nếu cần
- ✅ Consistent positioning across slides
- ✅ White/light background for contrast
- ✅ Test on projector trước khi present

---

## CHECKLIST

### Trước khi tạo PowerPoint:
- [ ] Đã có đủ 9 hình mới trong `slide_images/`
- [ ] Đã có đủ 10 hình cũ trong root folder
- [ ] Review tất cả hình (mở xem chất lượng)
- [ ] Quyết định color scheme cho slides

### Khi tạo slides:
- [ ] Insert đúng hình cho đúng slide
- [ ] Resize và position consistent
- [ ] Add title cho mỗi hình
- [ ] Check readability (text on images)

### Trước buổi bảo vệ:
- [ ] Test trên projector
- [ ] Backup slides (PDF + USB)
- [ ] Print handouts (optional)
- [ ] Có file gốc (.pptx)

---

## TÓM TẮT

✅ **9 hình mới** đã được tạo trong `slide_images/`
✅ **10 hình có sẵn** trong root folder
✅ **Tổng cộng 19 hình** cho 20 slides (Slide 1, 2, 20 không cần hình phức tạp)

**Tất cả hình đều:**
- ✅ High resolution (300 DPI)
- ✅ Professional design
- ✅ Color-coded
- ✅ Clear labels and titles
- ✅ Ready to use in presentation

**Next steps:**
1. Review tất cả hình (mở file để xem)
2. Tạo PowerPoint presentation
3. Insert hình theo mapping table
4. Add text content từ SLIDE_TRINH_BAY.md
5. Review và practice!

🎯 **Bạn đã sẵn sàng tạo presentation chuyên nghiệp!**


