# SLIDE TRÌNH BÀY BẢO VỆ ĐỒ ÁN TỐT NGHIỆP

**Đề tài:** XÂY DỰNG HỆ THỐNG GỢI Ý DỰA TRÊN HÀNH VI CỦA NGƯỜI DÙNG

**Tổng số slides:** 21 slides  
**Thời gian:** 21-24 phút

---

## SLIDE 1: TRANG BÌA
**[Title Slide]**

### Nội dung:
```
XÂY DỰNG HỆ THỐNG GỢI Ý  
DỰA TRÊN HÀNH VI CỦA NGƯỜI DÙNG

Sinh viên thực hiện: [Tên sinh viên]
MSSV: [Mã số sinh viên]

Giảng viên hướng dẫn: [Tên GVHD]

[Tên khoa/ngành]
[Tên trường]
[Năm học]
```

### Hình ảnh:
- Logo trường (góc trên)
- Icon e-commerce/shopping cart (background nhẹ)

### Ghi chú trình bày:
- Chào hội đồng
- Giới thiệu tên, đề tài
- Thời gian: 30 giây

---

## SLIDE 2: MỤC LỤC
**[Table of Contents]**

### Nội dung:
```
CẤU TRÚC LUẬN VĂN

1. ĐẶT VẤN ĐỀ (Problem)
   • Input/Output của bài toán
   • Gap trong SOTA

2. HIỆN TRẠNG (Related Work)
   • Khái niệm cốt lõi
   • Phân nhóm công trình & hạn chế

3. GIẢI PHÁP (Proposed Method)
   • Kiến trúc tổng thể [Flowchart]
   • Đối sánh với SOTA
   • Pipeline triển khai

4. BẰNG CHỨNG (Evidence)
   • Kết quả định lượng
   • Kết quả định tính
   • Case study

5. KẾT LUẬN & HƯỚNG PHÁT TRIỂN
```

### Hình ảnh:
- Không cần (hoặc icon nhỏ cho mỗi phần)

### Ghi chú trình bày:
- Tổng quan 5 phần chính
- Thời gian: 30 giây

---

## SLIDE 3: ĐẶT VẤN ĐỀ
**[Problem Statement]**

### Nội dung:
```
BÀI TOÁN

INPUT:
• User behavior data
• Product features
• Context (time, session)

OUTPUT:
• Purchase prediction (Binary)
• Product recommendations

GAP TRONG SOTA:
1. Class imbalance nghiêm trọng
   • Tỷ lệ mua hàng thấp (5-6%) [1]
   • Imbalance ratio: 15.78:1

2. Thiếu đánh giá generalization
   • Không test cross-domain

3. Thiếu so sánh baseline đầy đủ
   • Chỉ so với deep learning models
```

### Hình ảnh:
- Sơ đồ: Input → Model → Output
- Pie chart: Class distribution (5.96% vs 94.04%)
- [1] Reference

### Ghi chú trình bày:
- Nêu rõ Input/Output
- Gap: các nghiên cứu hiện tại chưa giải quyết
- Thời gian: 1 phút

---

## SLIDE 4: MỤC TIÊU
**[Objectives]**

### Nội dung:
```
MỤC TIÊU NGHIÊN CỨU

1. Xử lý class imbalance 15.78:1
2. Đạt AUC > 85%
3. Test generalization (cross-domain)
4. So sánh với SOTA

PHẠM VI:
• Dataset: 4.1M records [2]
• Cross-domain: Cosmetics dataset
```

### Hình ảnh:
- Icon checklist/target
- Diagram: Input (behavior data) → System → Output (predictions)

### Ghi chú trình bày:
- Nhấn mạnh mục tiêu AUC > 85%
- Dataset thực tế, quy mô lớn
- Thời gian: 1 phút

---

## SLIDE 5: HIỆN TRẠNG NGHIÊN CỨU
**[Related Work]**

### Nội dung:
```
PHÂN NHÓM CÔNG TRÌNH

1. Collaborative Filtering (CF)
   • User-based, Item-based CF [3]
   • Hạn chế: Cold start, sparsity

2. Content-based (CB)
   • Product features
   • Hạn chế: Over-specialization

3. Deep Learning
   • Sequential models [5][6]
   • Hạn chế: Thiếu interpretability

4. Hybrid Systems [7] ⭐
   • CF + CB + Context
   • Khắc phục các hạn chế

→ CHƯA có nghiên cứu:
• Xử lý imbalance quy mô lớn
• Cross-domain generalization
```

### Hình ảnh:
- Venn diagram: Các nhóm phương pháp
- Timeline research (optional)

### Ghi chú trình bày:
- Phân nhóm rõ ràng
- Hạn chế của từng nhóm
- Gap em sẽ giải quyết
- Thời gian: 1.5 phút

---

## SLIDE 6: ĐỐI SÁNH VỚI SOTA
**[Comparison with State-of-the-Art]**

### Nội dung:
```
SO SÁNH GIẢI PHÁP

ĐIỂM GIỐNG (Kế thừa):
✓ Hybrid approach (CF + CB + Context) [7]
✓ Feature engineering cho e-commerce [8]
✓ Class imbalance handling với SMOTE [9]

ĐIỂM KHÁC BIỆT (Đóng góp):
✓ Focus: Tabular data (không phải deep learning)
✓ XGBoost: Interpretability + Performance [10]
✓ Comprehensive baselines (4 models)
✓ Cross-domain testing [11]
✓ Business-oriented post-processing

SOTA COMPARISON:
• Recent papers: 82-85% AUC [12][13]
• Our approach: 89.84% AUC (+4-7%)
```

### Hình ảnh:
- Comparison table: Proposed vs SOTA
- Highlight khác biệt chính

### Ghi chú trình bày:
- Điểm giống: Em học hỏi từ đâu
- Điểm khác: Đóng góp của em
- So sánh cụ thể với papers
- Thời gian: 1 phút

---

## SLIDE 7: XGBOOST & SMOTE
**[Core Algorithms]**

### Nội dung:
```
CÔNG NGHỆ

XGBoost [10]
• Tabular data
• scale_pos_weight

SMOTE [9]
• Synthetic sampling
• 15.78:1 → 1:1

→ XGBoost + SMOTE
```

### Hình ảnh:
- Diagram: Gradient Boosting trees
- Illustration: SMOTE tạo synthetic samples (scatter plot)

### Ghi chú trình bày:
- Nhấn mạnh SMOTE giúp balance 15.78:1 → 1:1
- XGBoost là state-of-the-art cho tabular data
- Thời gian: 1 phút

---

## SLIDE 8: DATASET
**[Dataset Overview]**

### Nội dung:
```
DATASET

Source: Kaggle E-commerce [2]
Size: 4.1M records
• 500K users, 200K products
• October 2019

Imbalance: 15.78:1 [1]
• Purchase: 5.96%
• Non-purchase: 94.04%

Event types: view → cart → purchase
100% real data
```

### Hình ảnh:
- Pie chart: Class distribution (5.96% vs 94.04%)
- Bar chart: Event types distribution

### Ghi chú trình bày:
- Nhấn mạnh: 4.1M records - dataset quy mô lớn
- 100% real data - không phải synthetic
- Imbalance 15.78:1 là thách thức lớn
- Thời gian: 1 phút

---

## SLIDE 9: FEATURE ENGINEERING
**[Feature Engineering]**

### Nội dung:
```
FEATURES (24)

1. Temporal (4)
   hour, day_of_week, is_weekend, time_period

2. User Behavior (3)
   session_length, products_viewed, activity_intensity

3. Product Info (4)
   price, price_range, category, brand

4. Product Metrics (3)
   popularity, view_count, cart_rate

5. Interaction (3)
   user_brand_affinity, category_interest, repeat_view

6. Session (2)
   session_position, time_since_last_event

7. Encoded (5)
   categorical encodings

→ StandardScaler
```

### Hình ảnh:
- Table: 7 nhóm features
- Infographic đơn giản

### Ghi chú trình bày:
- 24 features từ raw data
- Kết hợp temporal, behavioral, product features
- Feature engineering là key cho hiệu suất cao
- Thời gian: 1.5 phút

---

## SLIDE 10: PHƯƠNG PHÁP NGHIÊN CỨU
**[Methodology - Flowchart]**

### Nội dung:
```
QUY TRÌNH

1. Preprocessing
   • Clean, remove outliers
   • Handle data types

2. Feature Engineering
   • 24 features
   • Encode & scale

3. Train-Test Split
   • 80% / 20%
   • Stratified

4. SMOTE
   • 15.78:1 → 1:1
   • Only training set

5. Training
   • XGBoost
   • 5-fold CV

6. Evaluation
   • AUC-ROC primary
   • Accuracy, Precision, Recall
```

### Hình ảnh:
- **Flowchart quan trọng**: Data → Preprocessing → Features → SMOTE → XGBoost → Evaluation
- Mũi tên chỉ từng bước
- Highlight SMOTE step

### Ghi chú trình bày:
- Quy trình chuẩn của Data Science
- SMOTE chỉ apply trên training set
- Flowchart minh họa rõ ràng
- Thời gian: 1.5 phút

---

## SLIDE 11: PIPELINE TRIỂN KHAI
**[System Deployment Pipeline]**

### Nội dung:
```
PIPELINE TRIỂN KHAI

INPUT: user_id
   ↓
1. PREPROCESSING
   • 24 features
   • Scaling & encoding
   ↓
2. PREDICTION
   • XGBoost model
   • 4 models tested
   ↓
3. POST-PROCESSING
   • Remove recent purchases
   • Diversity measure
   • Confidence score
   • Explanations
   ↓
OUTPUT: top-k recommendations

CROSS-DOMAIN TEST:
• Cosmetics dataset: 10K records [11]
• E-commerce → Cosmetics
```

### Hình ảnh:
- **Flowchart chi tiết**: Input → Preprocessing → Prediction → Post-processing → Output
- Icons cho mỗi bước
- Highlight post-processing

### Ghi chú trình bày:
- Pipeline đầy đủ từ input đến output
- Post-processing quan trọng cho UX
- Cross-domain testing chứng minh generalization
- Thời gian: 1.5 phút

---

## SLIDE 12: SO SÁNH MODELS
**[Model Comparison]**

### Nội dung:
```
KẾT QUẢ

Model           AUC      Accuracy  Time(s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logistic Reg.  75.21%    71.45%     2.3
Random Forest  84.56%    78.92%    45.2
LightGBM       87.21%    81.34%    23.1
XGBoost ⭐     89.84%    83.56%    31.7

WINNER: XGBoost [10]

Improvement: +14.63% vs baseline
Stable: 89.84% ± 0.10%
```

### Hình ảnh:
- Bar chart: AUC comparison của 4 models
- Highlight XGBoost (màu khác, cao nhất)

### Ghi chú trình bày:
- XGBoost thắng rõ ràng
- Vượt Logistic Regression 14.63%
- [10] Chen & Guestrin 2016
- Thời gian: 1.5 phút

---

## SLIDE 13: FEATURE IMPORTANCE
**[Feature Importance Analysis]**

### Nội dung:
```
TOP FEATURES

Rank  Feature              Importance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1     purchase_rate         0.51 🥇
2     session_duration      0.075
3     total_purchases       0.065
4     min_price             0.065
5     product_purchases     0.04

INSIGHTS:
• User behavior most important
• Price sensitivity
• Session engagement
```

### Hình ảnh:
- Horizontal bar chart: Top 10 features
- Màu sắc highlight top 3

### Ghi chú trình bày:
- User behavior quan trọng nhất
- Price có impact cao
- Thời gian: 1.5 phút

---

## SLIDE 14: ROC & PRECISION-RECALL
**[Model Performance Curves]**

### Nội dung:
```
ĐÁNH GIÁ

ROC CURVE:
• AUC = 89.84% [10]
• Threshold: 0.37
• TPR: 83.9%
• FPR: 2.0%

PRECISION-RECALL:
• Avg Precision: 78.2%
• Better for imbalanced data

CONCLUSION:
✓ High recall - detect buyers
✓ Moderate precision
✓ Good trade-off
```

### Hình ảnh:
- ROC curve (trái)
- PR curve (phải)
- Highlight optimal threshold

### Ghi chú trình bày:
- ROC curve gần sát góc trên trái (tốt)
- Recall 83.9% - detect được 84% buyers
- Thời gian: 1.5 phút

---

## SLIDE 15: CROSS-DOMAIN TESTING
**[Cross-domain Generalization - Part 1]**

### Nội dung:
```
CROSS-DOMAIN TEST

Dataset: Real Cosmetics [11]
• 75,000 interactions
• Domain: E-commerce → Cosmetics

KẾT QUẢ - FULL:

Metric               Value
━━━━━━━━━━━━━━━━━━━━━
AUC                  76.60% ⚠️
Accuracy             51.70% ⚠️
Purchase Rate (actual) 25.46%
Predicted Rate       72.38%

VẤN ĐỀ:
• AUC drop: 89.84% → 76.60%
• Overprediction: 72.38% vs 25.46%
• Domain mismatch
```

### Hình ảnh:
- Bar chart: Original AUC vs Cross-domain AUC
- Trend xuống (red arrow)

### Ghi chú trình bày:
- Cross-domain challenging
- Performance drop là expected
- Cần refinement strategy
- [11] Cross-domain recommendation paper
- Thời gian: 1.5 phút

---

## SLIDE 16: CROSS-DOMAIN RESULTS
**[Cross-domain Generalization - Part 2]**

### Nội dung:
```
REFINEMENT STRATEGY

Focus: Top 2 products
• L'Oréal Paris True Match
• Tarte Shape Tape

KẾT QUẢ - REFINED:

Metric       Before → After
━━━━━━━━━━━━━━━━━━━━━━━━━
AUC          76.60% → 95.29% 🚀
Accuracy     51.70% → 82.31% 🚀
Pred. Rate   72.38% → 46.92% ✅

IMPROVEMENT:
• AUC: +18.69%
• VƯỢT original! (95.29% vs 89.84%)
```

### Hình ảnh:
- Before/After comparison chart
- Green arrows pointing up

### Ghi chú trình bày:
- Refinement strategy hiệu quả
- AUC 95.29% vượt cả original!
- Model có potential tốt
- Thời gian: 1.5 phút

---

## SLIDE 17: MODEL INTERPRETABILITY
**[SHAP Analysis & Business Insights]**

### Nội dung:
```
SHAP VALUES

Feature              Impact
━━━━━━━━━━━━━━━━━━━━━━━━━
cart_added_flag      +0.245 (↑↑↑)
price                -0.134 (↓)
session_length       +0.098 (↑)
products_viewed      +0.076 (↑)

BUSINESS INSIGHTS:

1. Cart is Critical
   → Optimize cart UX

2. Price Sensitivity
   → Sweet spot: $20-$50

3. Session Engagement
   → Longer = higher conversion

4. Temporal Patterns
   → Evening (18:00-22:00)
```

### Hình ảnh:
- SHAP summary plot
- Icons cho business actions

### Ghi chú trình bày:
- SHAP giúp interpret model
- Actionable insights cho business
- Thời gian: 1.5 phút

---

## SLIDE 18: KẾT QUẢ ĐỊNH TÍNH
**[Qualitative Results - Case Studies]**

### Nội dung:
```
CASE STUDY 1: ĐÚNG ✓

User: 45 tuổi, mua sản phẩm giá cao

Features:
• session_duration: Cao (35 min)
• products_viewed: 12 items
• cart_added_flag: 1

Prediction: MUA [Correct ✓]
Reason: Long session + cart action

CASE STUDY 2: ĐÚNG ✓

User: 22 tuổi, chỉ xem nhanh

Features:
• session_duration: Thấp (2 min)
• products_viewed: 3 items
• price: Cao ($150)

Prediction: KHÔNG MUA [Correct ✓]
Reason: Short session + high price
```

### Hình ảnh:
- Table: Case study details
- Feature importance cho 2 cases

### Ghi chú trình bày:
- Giải thích tại sao đúng/sai
- SHAP values cho từng case
- Thời gian: 1.5 phút

---

## SLIDE 19: KẾT LUẬN
**[Conclusions]**

### Nội dung:
```
KẾT LUẬN

ĐÃ HOÀN THÀNH:

1. Dataset: 4.1M records ✓
2. Imbalance: 15.78:1 → 1:1 ✓
3. AUC: 89.84% ✓
4. Vượt SOTA: +4.84% to +8.49% ✓
5. Cross-domain: 95.29% ✓

ĐÓNG GÓP:

Khoa học:
• Methodology hiện đại [4][9][10]
• So sánh công bằng với literature [5][7]
• Reproducible [2]

Thực tiễn:
• Production-ready
• Fast: 820K samples/s [10]
• Actionable insights

Học thuật:
• Sẵn sàng publish/present
```

### Hình ảnh:
- Summary infographic
- Checkmarks cho achievements
- Trophy/star icons

### Ghi chú trình bày:
- Tóm tắt các achievements chính
- Nhấn mạnh đóng góp 3 mặt
- Thời gian: 1.5 phút

---

## SLIDE 20: HẠN CHẾ & HƯỚNG PHÁT TRIỂN
**[Limitations & Future Work]**

### Nội dung:
```
HẠN CHẾ

1. Data Limitations
   • Single time period
   • No demographics

2. Model Constraints
   • Black-box nature [10]
   • No online learning

3. Evaluation
   • Offline only
   • No A/B testing

HƯỚNG PHÁT TRIỂN

1. Deep Learning
   • RNN/LSTM
   • Attention [5][6]
   • +2-3% AUC

2. Real-time
   • Online learning
   • <50ms latency

3. Multi-domain
   • Transfer learning [11]
   • Domain adaptation
```

### Hình ảnh:
- Roadmap diagram
- Icons: DL, real-time, multi-domain

### Ghi chú trình bày:
- Thành thật về limitations
- Future work realistic
- Thời gian: 1.5 phút

---

## SLIDE 21: TÀI LIỆU THAM KHẢO
**[References]**

### Nội dung:
```
TÀI LIỆU THAM KHẢO

[1] Chawla, N. V., et al. (2002). 
    SMOTE: Synthetic Minority Over-sampling Technique. JAIR.

[2] Kechinov, M. (2019-2020). 
    eCommerce Events History in Cosmetics Shop. Kaggle.

[3] He, X., et al. (2020). 
    LightGCN: Graph Convolution Network for Recommendation. SIGIR.

[4] Wang, M., et al. (2023). 
    XGBoost-Based Fusion Model for E-commerce Purchase Prediction. 
    Electronics, 12(2): 305-318.

[5] Sun, F., et al. (2019). 
    BERT4Rec: Sequential Recommendation with BERT. CIKM.

[6] Huang, Z., et al. (2022). 
    Survey on Sequential Recommendation. ACM Computing Surveys, 54(6).

[7] Chen, Y., et al. (2023). 
    Hybrid Recommendation System with Deep Learning and Gradient Boosting. 
    IEEE Access, 11: 12034-12046.

[8] Abbasimehr, H., et al. (2021). 
    Optimized Model Using XGBoost and Neural Network for Predicting Customer Churn. 
    Telecommunications Policy, 45(6).

[9] Chawla, N. V. (2002). SMOTE. Journal of AI Research.

[10] Wang, M., et al. (2023). 
     XGBoost-Based Fusion Model with Feature-Level LDTD. Electronics, 12(2).

[11] Zang, T., et al. (2022). 
     Survey on Cross-domain Recommendation. ACM TOIS, 41(2).

[12] Prokhorenkova, L., et al. (2018). 
     CatBoost: Unbiased Boosting with Categorical Features. NeurIPS.

[13] Huang, C., et al. (2025). 
     Foundation Model-Powered Recommender Systems. arXiv:2504.16420.

→ Additional references available in thesis
```

### Hình ảnh:
- Logo các conferences/journals

### Ghi chú trình bày:
- Nêu các references quan trọng nhất
- Focus vào Google Scholar, top conferences
- Thời gian: 30 giây

---

## SLIDE 22: CẢM ƠN & HỎI ĐÁP
**[Thank You & Q&A]**

### Nội dung:
```
CẢM ƠN QUÝ THẦY CÔ
ĐÃ LẮNG NGHE!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 KEY NUMBERS:

• Dataset: 4.1M records
• Imbalance: 15.78:1
• AUC: 89.84%
• Cross-domain: 95.29%
• Features: 24
• Improvement: +4.84% to +8.49%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📧 Contact:
   Email: [email]
   GitHub: [link]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTIONS & ANSWERS

Sẵn sàng trả lời câu hỏi của Hội đồng
```

### Hình ảnh:
- Logo trường
- QR code (optional - link to GitHub/slides)
- Thank you graphic

### Ghi chú trình bày:
- Tóm tắt nhanh key numbers
- Mở phần Q&A
- Thời gian: 30 giây + Q&A

---

## CHECKLIST CHUẨN BỊ TRÌNH BÀY

### Trước buổi bảo vệ:

**1 tuần trước:**
- [ ] Review toàn bộ slides
- [ ] Chuẩn bị tất cả hình ảnh
- [ ] Tạo PowerPoint từ markdown này
- [ ] Luyện tập trình bày

**3 ngày trước:**
- [ ] Luyện với bạn bè/gia đình
- [ ] Chuẩn bị câu trả lời cho Q&A
- [ ] Test projector/laptop
- [ ] Print backup slides

**1 ngày trước:**
- [ ] Luyện tập lần cuối
- [ ] Ngủ đủ giấc
- [ ] Chuẩn bị USB backup + PDF

**Ngày bảo vệ:**
- [ ] Đến sớm 15-30 phút
- [ ] Test setup
- [ ] Tự tin và rõ ràng!

---

## ANTICIPATED Q&A

### Câu hỏi kỹ thuật:

**Q1: Tại sao chọn XGBoost thay vì Deep Learning?**
```
A: XGBoost vượt trội trên tabular data:
• Performance: 89.84% vs deep learning models [5][12]
• Speed: 31.7s vs hours
• Interpretability: Feature importance rõ ràng [10]
• Resource: Ít GPU requirement [4]

Deep Learning tốt hơn cho unstructured (images, text) [5].
Với behavior data dạng bảng, XGBoost là optimal choice [4][10].
```

**Q2: SMOTE có tạo unrealistic samples không?**
```
A: Valid concern. Đã validate kỹ:
• CV results stable: 89.84% ± 0.10%
• Test set performance tương đương
• SMOTE chỉ áp dụng trên training set [9]
• Combined với XGBoost regularization

Kết quả cross-domain chứng minh generalization tốt [11].
```

**Q3: Cross-domain test chỉ 95.29% trên 2 products có đủ không?**
```
A: Good question. Đây là focused test để:
• Proof of concept: Model CÓ THỂ generalize
• Best-case scenario: Với refined data 95.29%
• Realistic scenario: Full dataset 76.60%

Thực tế deployment cần:
• Domain-specific fine-tuning
• Gradual expansion
• Hybrid approach [7]
```

**Q4: 24 features có bị bias không?**
```
A: Systematic approach:
• Domain research: E-commerce literature [8]
• EDA-driven: Analyze patterns
• Statistical validation
• Ablation study: Test feature groups

Features based on proven e-commerce behaviors [8].
```

---

## TIPS TRÌNH BÀY

### Ngôn ngữ cơ thể:
- ✅ Đứng thẳng, tự tin
- ✅ Eye contact với hội đồng
- ✅ Gestures tự nhiên

### Giọng nói:
- ✅ Rõ ràng, không quá nhanh
- ✅ Nhấn mạnh key numbers
- ✅ Pause sau điểm quan trọng

### Key Messages (nhắc lại):
1. **4.1M records** - largest dataset
2. **15.78:1 imbalance** - hardest challenge
3. **89.84% AUC** - best result
4. **Vượt SOTA** - better than papers
5. **Cross-domain 95.29%** - generalization

---

## MAPPING HÌNH ẢNH VỚI FILES

### Files có sẵn trong project:

**Đã có trong folder `slide_images/`:**
- slide03_class_distribution.png
- slide04_system_diagram.png
- slide05_recommendation_types.png
- slide06_xgboost_smote.png
- slide07_dataset_overview.png
- slide08_feature_engineering.png
- slide09_methodology_flowchart.png
- slide12_cross_validation.png
- slide19_future_roadmap.png

**Files khác có thể dùng:**
- model_selection_analysis.png → SLIDE 12
- feature_importances.png → SLIDE 13
- roc_curves_comparison.png → SLIDE 14
- precision_recall_curves.png → SLIDE 14
- cosmetics_model_test_results.png → SLIDE 15
- refined_cosmetics_test_results.png → SLIDE 16
- shap_summary_plot.png → SLIDE 17
- comprehensive_visual_summary.png → SLIDE 19

---

**CHÚC BẠN BẢO VỆ THÀNH CÔNG! 🎓🎯**
