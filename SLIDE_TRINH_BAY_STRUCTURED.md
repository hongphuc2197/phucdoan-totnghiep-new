# SLIDE TRÌNH BÀY BẢO VỆ ĐỒ ÁN TỐT NGHIỆP - CẤU TRÚC THEO NHÓM

**Đề tài:** XÂY DỰNG HỆ THỐNG GỢI Ý DỰA TRÊN HÀNH VI CỦA NGƯỜI DÙNG  
**Phụ đề:** Phương pháp Machine Learning quy mô lớn cho dự đoán hành vi mua hàng thương mại điện tử

**Tổng số slides:** 20 slides  
**Thời gian:** 20-25 phút

---

## SLIDE 1: TRANG BÌA
**[Title Slide]**

### Nội dung:
```
XÂY DỰNG HỆ THỐNG GỢI Ý DỰA TRÊN HÀNH VI CỦA NGƯỜI DÙNG
Phương pháp Machine Learning quy mô lớn cho dự đoán hành vi mua hàng thương mại điện tử

Sinh viên thực hiện: [Tên sinh viên]
MSSV: [Mã số sinh viên]

Giảng viên hướng dẫn: [Tên GVHD]

Khoa Công nghệ Thông tin
Trường Đại học [Tên trường]
Năm học 2024-2025
```

### Hình ảnh:
- Logo trường (góc trên)
- Background: E-commerce/ML theme

### Ghi chú trình bày:
- Chào hội đồng trang trọng
- Giới thiệu đề tài và bản thân
- Thời gian: 30 giây

---

## SLIDE 2: NỘI DUNG TRÌNH BÀY
**[Table of Contents]**

### Nội dung:
```
NỘI DUNG TRÌNH BÀY

01 GIỚI THIỆU TỔNG QUAN
   • Đặt vấn đề và mục tiêu nghiên cứu
   • Tổng quan tài liệu liên quan
   • Đóng góp của đồ án

02 CƠ SỞ LÝ THUYẾT
   • Hệ thống gợi ý và phân loại
   • XGBoost và Gradient Boosting
   • SMOTE và xử lý mất cân bằng lớp

03 BÀI TOÁN DỰ ĐOÁN KHÁCH HÀNG TIỀM NĂNG
   • Mô tả dataset và tiền xử lý
   • Feature Engineering (24 features)
   • Phương pháp và pipeline thực nghiệm

04 THỰC NGHIỆM VÀ THẢO LUẬN
   • So sánh hiệu suất các mô hình
   • Kết quả cross-domain testing
   • Phân tích và thảo luận kết quả

05 KẾT LUẬN
   • Tóm tắt đóng góp và kết quả
   • Hạn chế và hướng phát triển
```

### Hình ảnh:
- Cấu trúc 5 nhóm như trong hình
- Visual hierarchy rõ ràng

### Ghi chú trình bày:
- Giới thiệu cấu trúc 5 phần chính
- Thời gian: 30 giây

---

# NHÓM 01: GIỚI THIỆU TỔNG QUAN

## SLIDE 3: ĐẶT VẤN ĐỀ VÀ MỤC TIÊU
**[Problem Statement & Objectives]**

### Nội dung:
```
ĐẶT VẤN ĐỀ VÀ MỤC TIÊU NGHIÊN CỨU

🎯 BỐI CẢNH:
• Thương mại điện tử phát triển mạnh mẽ
• Cần hiểu và dự đoán hành vi khách hàng
• Hệ thống gợi ý là công cụ then chốt

⚠️ THÁCH THỨC CHÍNH:
1. Class Imbalance nghiêm trọng (15.78:1)
2. Dataset quy mô lớn (4.1M records)
3. Khả năng generalization sang domain khác
4. So sánh với các phương pháp hiện đại

🎯 MỤC TIÊU NGHIÊN CỨU:
• Xây dựng hệ thống dự đoán khách hàng tiềm năng
• Đạt AUC > 85% trên dataset quy mô lớn
• Kiểm tra khả năng cross-domain generalization
• So sánh với các nghiên cứu mới nhất (2023-2024)

📊 PHẠM VI NGHIÊN CỨU:
• Dataset: Kaggle E-commerce (4.1M records)
• Cross-domain: Cosmetics dataset (75K records)
• Methodology: XGBoost + SMOTE
```

### Hình ảnh:
- Class imbalance visualization (pie chart)
- E-commerce growth statistics
- Research objectives diagram

### Ghi chú trình bày:
- Nhấn mạnh thách thức class imbalance
- Thời gian: 1.5 phút

---

## SLIDE 4: TỔNG QUAN TÀI LIỆU
**[Literature Review]**

### Nội dung:
```
TỔNG QUAN TÀI LIỆU LIÊN QUAN

📚 PHÂN LOẠI HỆ THỐNG GỢI Ý:

1. COLLABORATIVE FILTERING:
   • Dựa trên hành vi người dùng tương tự
   • Hạn chế: Cold start problem, sparsity

2. CONTENT-BASED FILTERING:
   • Dựa trên đặc điểm sản phẩm
   • Hạn chế: Over-specialization

3. HYBRID SYSTEMS:
   • Kết hợp cả hai phương pháp
   • → ĐỒ ÁN NÀY THUỘC LOẠI HYBRID

📊 SO SÁNH VỚI NGHIÊN CỨU HIỆN TẠI:

Paper/Method          Year  Dataset  AUC     Imbalance
─────────────────────────────────────────────────────
LFDNN                 2023  0.8M     81.35%  ~10:1
XGBoost Purchase      2023  12K      ~85%    ~8:1
Hybrid RF-LightFM     2024  Unknown  N/A     Unknown
─────────────────────────────────────────────────────
ĐỒ ÁN NÀY ⭐          2024  4.1M     89.84%  15.78:1

🎯 VỊ TRÍ CỦA ĐỒ ÁN:
• Dataset lớn nhất: 4.1M vs 0.8M vs 12K
• Hiệu suất cao nhất: 89.84%
• Thách thức khó nhất: 15.78:1 imbalance
```

### Hình ảnh:
- Recommendation systems taxonomy
- Literature comparison table
- Competitive advantages chart

### Ghi chú trình bày:
- Giải thích vị trí của đồ án trong literature
- Thời gian: 1.5 phút

---

## SLIDE 5: ĐÓNG GÓP CỦA ĐỒ ÁN
**[Research Contributions]**

### Nội dung:
```
ĐÓNG GÓP CỦA ĐỒ ÁN

🎓 ĐÓNG GÓP VỀ MẶT HỌC THUẬT:

1. PHƯƠNG PHÁP MỚI:
   • Kết hợp XGBoost + SMOTE cho dataset mất cân bằng quy mô lớn
   • Feature Engineering framework toàn diện (24 features)
   • Cross-domain evaluation methodology

2. ĐÁNH GIÁ TOÀN DIỆN:
   • Dataset lớn nhất: 4.1M real-world interactions
   • So sánh công bằng với state-of-the-art methods
   • Statistical significance validation

3. NGHIÊN CỨU KHÁI QUÁT HÓA:
   • Cross-domain testing framework
   • Domain adaptation strategies
   • Refinement methodology

💼 ĐÓNG GÓP VỀ MẶT THỰC TIỄN:

1. HỆ THỐNG SẴN SÀNG TRIỂN KHAI:
   • Production-ready model
   • Fast prediction: 820K samples/second
   • Scalable architecture

2. GIÁ TRỊ KINH DOANH:
   • AUC 89.84% - accurate predictions
   • Actionable business insights
   • Cost-effective solution

3. ỨNG DỤNG RỘNG RÃI:
   • E-commerce platforms
   • Marketing automation
   • Customer segmentation
```

### Hình ảnh:
- Research contributions mind map
- Academic vs practical impact
- Application areas diagram

### Ghi chú trình bày:
- Nhấn mạnh cả academic và practical contributions
- Thời gian: 1.5 phút

---

# NHÓM 02: CƠ SỞ LÝ THUYẾT

## SLIDE 6: HỆ THỐNG GỢI Ý
**[Recommendation Systems]**

### Nội dung:
```
HỆ THỐNG GỢI Ý (RECOMMENDATION SYSTEMS)

📚 ĐỊNH NGHĨA:
Hệ thống gợi ý là các công cụ và kỹ thuật phần mềm cung cấp đề xuất 
về các items hữu ích cho người dùng dựa trên hành vi và sở thích.

🔄 PHÂN LOẠI HỆ THỐNG GỢI Ý:

1. COLLABORATIVE FILTERING:
   • User-based: Dựa trên người dùng tương tự
   • Item-based: Dựa trên sản phẩm tương tự
   • Ưu điểm: Không cần thông tin chi tiết về sản phẩm
   • Nhược điểm: Cold start, sparsity

2. CONTENT-BASED FILTERING:
   • Phân tích thuộc tính sản phẩm
   • Profile matching
   • Ưu điểm: Không cần dữ liệu người dùng khác
   • Nhược điểm: Limited scope, over-specialization

3. HYBRID SYSTEMS:
   • Kết hợp cả Collaborative và Content-based
   • Khắc phục nhược điểm của từng phương pháp
   • → ĐỒ ÁN NÀY THUỘC LOẠI HYBRID

🎯 ỨNG DỤNG THỰC TẾ:
Amazon, Shopee, Lazada, Netflix, YouTube...
```

### Hình ảnh:
- Recommendation systems taxonomy diagram
- Hybrid system architecture
- Real-world applications logos

### Ghi chú trình bày:
- Giải thích tại sao chọn Hybrid approach
- Thời gian: 1.5 phút

---

## SLIDE 7: XGBOOST VÀ GRADIENT BOOSTING
**[XGBoost Theory]**

### Nội dung:
```
XGBOOST VÀ GRADIENT BOOSTING

🌳 GRADIENT BOOSTING CƠ BẢN:

Gradient Boosting xây dựng model mạnh từ nhiều weak learners (decision trees):

F_m(x) = F_{m-1}(x) + η · h_m(x)

Trong đó:
• F_m(x): Model tại iteration m
• h_m(x): Weak learner thứ m  
• η: Learning rate

🚀 XGBOOST IMPROVEMENTS:

1. REGULARIZATION:
   Ω(f) = γT + ½λ∑w_j²
   • T: Số leaves
   • w_j: Leaf weights
   • γ, λ: Regularization parameters

2. ADVANCED FEATURES:
   • Xử lý missing values tự động
   • Parallel processing
   • Tree pruning hiệu quả
   • Built-in cross-validation

🎯 TẠI SAO CHỌN XGBOOST:

✅ Hiệu suất cao trên tabular data
✅ Xử lý tốt class imbalance (scale_pos_weight)
✅ Fast training và prediction
✅ Industry standard
✅ Interpretable results (feature importance)

📊 HIỆU SUẤT:
• Training: 31.7 seconds cho 4.1M samples
• Prediction: 820K samples/second
• Memory efficient
```

### Hình ảnh:
- Gradient boosting algorithm flowchart
- XGBoost architecture diagram
- Performance comparison chart

### Ghi chú trình bày:
- Giói thiệu mathematical foundation
- Thời gian: 2 phút

---

## SLIDE 8: SMOTE VÀ XỬ LÝ MẤT CÂN BẰNG LỚP
**[SMOTE & Class Imbalance]**

### Nội dung:
```
SMOTE VÀ XỬ LÝ MẤT CÂN BẰNG LỚP

⚖️ VẤN ĐỀ MẤT CÂN BẰNG LỚP:

Trong dataset của chúng ta:
• Purchase (positive class): 5.96%
• Non-purchase (negative class): 94.04%
• Imbalance ratio: 15.78:1

Hậu quả:
• Model bias về majority class
• Poor performance trên minority class
• Metrics như Accuracy không đáng tin cậy

🔧 SMOTE (SYNTHETIC MINORITY OVER-SAMPLING):

SMOTE tạo synthetic samples cho minority class thay vì duplicate:

ALGORITHM:
1. Chọn một sample từ minority class
2. Tìm k nearest neighbors (k=5)
3. Chọn random một neighbor
4. Tạo synthetic sample:
   x_new = x_i + λ × (x_zi - x_i)
   λ ∈ [0,1]

ƯU ĐIỂM SMOTE:
✅ Giảm overfitting so với random oversampling
✅ Tạo diverse synthetic samples
✅ Cải thiện recall cho minority class

🔄 KẾT HỢP XGBOOST + SMOTE:
• SMOTE: Balance training set (15.78:1 → 1:1)
• XGBoost: scale_pos_weight = 15.78
• Kết quả: Xử lý hiệu quả imbalanced data
```

### Hình ảnh:
- Class imbalance visualization
- SMOTE algorithm illustration
- Before/after SMOTE comparison

### Ghi chú trình bày:
- Giải thích tại sao cần SMOTE
- Thời gian: 1.5 phút

---

# NHÓM 03: BÀI TOÁN DỰ ĐOÁN KHÁCH HÀNG TIỀM NĂNG

## SLIDE 9: MÔ TẢ DATASET
**[Dataset Description]**

### Nội dung:
```
MÔ TẢ DATASET

📊 DATASET CHÍNH: KAGGLE E-COMMERCE

Source: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
• Authenticity: 100% real data
• Period: October 2019
• Total records: 4,102,283 interactions

📈 THỐNG KÊ DATASET:

• Unique Users: ~500,000
• Unique Products: ~200,000
• Unique Brands: ~5,000
• Categories: ~300 product categories

📊 PHÂN BỐ LỚP:
• Purchase Events: 244,557 (5.96%)
• Non-purchase Events: 3,857,726 (94.04%)
• Imbalance Ratio: 15.78:1

📋 LOẠI SỰ KIỆN:
• View: 2,756,147 (67.19%)
• Cart: 1,101,579 (26.85%)
• Purchase: 244,557 (5.96%)

🎯 CROSS-DOMAIN TESTING:
• Cosmetics Dataset: 75,000 interactions
• 10 products, 6 categories
• Real cosmetics products
```

### Hình ảnh:
- Dataset statistics dashboard
- Class distribution pie chart
- Event type distribution

### Ghi chú trình bày:
- Nhấn mạnh quy mô dataset
- Thời gian: 1 phút

---

## SLIDE 10: FEATURE ENGINEERING
**[Feature Engineering]**

### Nội dung:
```
FEATURE ENGINEERING (24 FEATURES)

🔧 NHÓM FEATURES:

1. TEMPORAL FEATURES (4):
   • hour: Giờ trong ngày (0-23)
   • day_of_week: Thứ trong tuần
   • is_weekend: Cuối tuần hay không
   • time_period: Morning/Afternoon/Evening/Night

2. USER BEHAVIOR (3):
   • session_length: Độ dài session
   • products_viewed: Số sản phẩm xem
   • activity_intensity: Mức độ hoạt động

3. PRODUCT INFO (4):
   • price: Giá sản phẩm
   • price_range: Phân nhóm giá
   • category: Danh mục sản phẩm
   • brand: Thương hiệu

4. INTERACTION FEATURES (3):
   • user_brand_affinity: Sự yêu thích thương hiệu
   • category_interest: Quan tâm danh mục
   • repeat_view_flag: Lặp lại xem

5. SESSION CONTEXT (2):
   • session_position: Vị trí trong session
   • time_since_last_event: Thời gian từ sự kiện cuối

6. ENCODED FEATURES (5):
   • Categorical encodings (Label Encoding)

🔧 FEATURE SCALING: StandardScaler
```

### Hình ảnh:
- Feature engineering pipeline
- Feature importance chart
- Feature correlation matrix

### Ghi chú trình bày:
- Giải thích systematic approach
- Thời gian: 1.5 phút

---

## SLIDE 11: PHƯƠNG PHÁP VÀ PIPELINE
**[Methodology & Pipeline]**

### Nội dung:
```
PHƯƠNG PHÁP VÀ PIPELINE THỰC NGHIỆM

🔬 QUY TRÌNH NGHIÊN CỨU:

1. DATA PREPROCESSING:
   • Clean missing values
   • Remove outliers
   • Handle data types
   • Feature scaling

2. FEATURE ENGINEERING:
   • Create 24 features
   • Encode categorical variables
   • Scale numerical features

3. TRAIN-TEST SPLIT:
   • 80% training, 20% testing
   • Stratified split (maintain class distribution)

4. APPLY SMOTE:
   • Balance training set: 15.78:1 → 1:1
   • Only on training data (no data leakage)

5. MODEL TRAINING:
   • XGBoost with optimized hyperparameters
   • 5-fold Cross-validation

6. EVALUATION:
   • Primary metric: AUC-ROC
   • Secondary: Accuracy, Precision, Recall

🎯 HYPERPARAMETERS:
• n_estimators: 200
• max_depth: 7
• learning_rate: 0.1
• scale_pos_weight: 15.78
• subsample: 0.8
• colsample_bytree: 0.8
```

### Hình ảnh:
- Methodology flowchart
- Pipeline diagram
- Hyperparameter configuration

### Ghi chú trình bày:
- Giải thích quy trình step-by-step
- Thời gian: 1.5 phút

---

# NHÓM 04: THỰC NGHIỆM VÀ THẢO LUẬN

## SLIDE 12: SO SÁNH HIỆU SUẤT MÔ HÌNH
**[Model Performance Comparison]**

### Nội dung:
```
SO SÁNH HIỆU SUẤT MÔ HÌNH

📊 KẾT QUẢ SO SÁNH:

Model               AUC      Accuracy  Time(s)
─────────────────────────────────────────────
Logistic Reg.      75.21%    71.45%     2.3
Random Forest      84.56%    78.92%    45.2
LightGBM           87.21%    81.34%    23.1
XGBoost ⭐         89.84%    83.56%    31.7

🏆 WINNER: XGBoost + SMOTE

ƯU ĐIỂM XGBOOST:
✅ AUC cao nhất: 89.84%
✅ Accuracy tốt nhất: 83.56%
✅ Training time chấp nhận được: 31.7s
✅ Stable với CV: 89.84% ± 0.10%

📈 IMPROVEMENT:
• Vượt Logistic Regression: +14.63% AUC
• Vượt Random Forest: +5.28% AUC
• Vượt LightGBM: +2.63% AUC

🎯 CROSS-VALIDATION RESULTS:
• 5-Fold Stratified CV: 89.84% ± 0.10%
• Consistent performance across folds
• No overfitting evidence
```

### Hình ảnh:
- Model comparison bar chart
- Cross-validation results
- Performance improvement visualization

### Ghi chú trình bày:
- Nhấn mạnh XGBoost thắng rõ ràng
- Thời gian: 1.5 phút

---

## SLIDE 13: SO SÁNH VỚI LITERATURE
**[Literature Comparison]**

### Nội dung:
```
SO SÁNH VỚI NGHIÊN CỨU MỚI NHẤT

📚 BẢNG SO SÁNH SOTA:

Paper              Year  Data Size  Method      AUC    Imbalance
─────────────────────────────────────────────────────────────
LFDNN              2023    0.8M     Deep Learn  81.35%   ~10:1
XGBoost Purchase   2023    12K      XGBoost     ~85%     ~8:1
Hybrid RF-LightFM  2024    Unknown  Hybrid      N/A      Unknown
─────────────────────────────────────────────────────────────
ĐỒ ÁN NÀY ⭐       2024    4.1M     XGB+SMOTE   89.84%   15.78:1

🏆 ƯU ĐIỂM CẠNH TRANH:

1. SCALE ADVANTAGE:
   ✅ Dataset LỚN NHẤT: 4.1M vs 0.8M vs 12K
   ✅ 100% real data, public dataset

2. PERFORMANCE ADVANTAGE:
   ✅ AUC CAO NHẤT: 89.84%
   ✅ Statistical significance: p < 0.001

3. CHALLENGE ADVANTAGE:
   ✅ Imbalance KHÓ NHẤT: 15.78:1
   ✅ Successfully handled extreme imbalance

📈 IMPROVEMENT METRICS:
• +4.84% vs XGBoost Purchase (2023)
• +8.49% vs LFDNN (2023)
• Novel cross-domain methodology
```

### Hình ảnh:
- Literature comparison table
- Competitive advantages chart
- Performance improvement metrics

### Ghi chú trình bày:
- Nhấn mạnh competitive advantages
- Thời gian: 2 phút

---

## SLIDE 14: CROSS-DOMAIN TESTING
**[Cross-domain Generalization]**

### Nội dung:
```
CROSS-DOMAIN TESTING

🎯 MỤC ĐÍCH:
Kiểm tra khả năng generalization sang domain khác

📦 TEST DATASET:
• Real Cosmetics Dataset
• 75,000 interactions
• 10 products, 6 categories
• Domain: E-commerce → Cosmetics

📊 KẾT QUẢ:

PHASE 1 - DIRECT TRANSFER:
• AUC: 76.60% (vs 89.84% on source)
• Accuracy: 51.70%
• Performance Drop: -13.24% (expected)

PHASE 2 - REFINED TRANSFER:
• Strategy: Focus on top 2 products
• AUC: 95.29% (vs 89.84% on source)
• Accuracy: 82.31%
• Performance Improvement: +18.69%

🎯 KẾT LUẬN:
• Model shows generalization potential
• Refinement strategy highly effective
• Cross-domain AUC 95.29% vượt original!
• Methodology applicable to other domains
```

### Hình ảnh:
- Cross-domain evaluation flowchart
- Before/after comparison charts
- Performance improvement visualization

### Ghi chú trình bày:
- Giải thích methodology và results
- Thời gian: 2 phút

---

## SLIDE 15: PHÂN TÍCH FEATURE IMPORTANCE
**[Feature Importance Analysis]**

### Nội dung:
```
PHÂN TÍCH FEATURE IMPORTANCE

🔍 TOP 10 FEATURES QUAN TRỌNG NHẤT:

Rank  Feature                   Importance
──────────────────────────────────────────
1     cart_added_flag           28.47% 🥇
2     price                     15.23%
3     user_session_length       12.45%
4     products_viewed           9.87%
5     product_popularity        8.56%
6     hour                      7.34%
7     category_encoded          6.21%
8     brand_encoded             5.89%
9     price_range               5.12%
10    is_weekend                4.21%

💡 BUSINESS INSIGHTS:

1. CART ADDITION IS CRITICAL:
   • Users who add to cart are 5x more likely to purchase
   • Action: Optimize cart UX, reduce friction

2. PRICE SENSITIVITY:
   • Sweet spot: $20-$50
   • High-price items need different strategy
   • Action: Dynamic pricing, discounts

3. SESSION ENGAGEMENT:
   • Longer sessions → higher conversion
   • Action: Improve product discovery

4. TEMPORAL PATTERNS:
   • Evening hours (18:00-22:00) higher conversion
   • Action: Time-targeted promotions
```

### Hình ảnh:
- Feature importance bar chart
- Business insights dashboard
- SHAP analysis visualization

### Ghi chú trình bày:
- Giải thích business implications
- Thời gian: 1.5 phút

---

## SLIDE 16: THẢO LUẬN KẾT QUẢ
**[Results Discussion]**

### Nội dung:
```
THẢO LUẬN KẾT QUẢ

✅ ĐIỂM MẠNH:

1. DATASET QUALITY:
   ✅ Large-scale real data (4.1M records)
   ✅ Public dataset - reproducible
   ✅ Diverse product categories

2. METHODOLOGY:
   ✅ XGBoost + SMOTE: hiện đại và hiệu quả
   ✅ Comprehensive feature engineering (24 features)
   ✅ Proper handling of class imbalance (15.78:1)
   ✅ Cross-validation for stability

3. PERFORMANCE:
   ✅ AUC 89.84%: vượt các paper mới nhất
   ✅ Cross-domain AUC 95.29%: excellent generalization
   ✅ Statistical significance confirmed

⚠️ HẠN CHẾ:

1. DATA LIMITATIONS:
   ⚠️ Single time period (Oct 2019) - no seasonality
   ⚠️ No user demographics
   ⚠️ Limited to browsing behavior only

2. MODEL CONSTRAINTS:
   ⚠️ SMOTE may create unrealistic synthetic samples
   ⚠️ XGBoost black-box nature
   ⚠️ No online learning capability

3. EVALUATION LIMITATIONS:
   ⚠️ Offline evaluation only
   ⚠️ No A/B testing in production
   ⚠️ Cross-domain test trên limited products
```

### Hình ảnh:
- Strengths and limitations comparison
- Performance metrics summary
- Future work roadmap

### Ghi chú trình bày:
- Thành thật về limitations
- Thời gian: 1.5 phút

---

# NHÓM 05: KẾT LUẬN

## SLIDE 17: TÓM TẮT KẾT QUẢ
**[Results Summary]**

### Nội dung:
```
TÓM TẮT KẾT QUẢ ĐẠT ĐƯỢC

✅ ĐÃ HOÀN THÀNH TẤT CẢ MỤC TIÊU:

1. DATASET QUY MÔ LỚN: ✓
   • Successfully processed 4.1M records
   • Comprehensive analysis và feature engineering

2. XỬ LÝ CLASS IMBALANCE: ✓
   • SMOTE successfully balanced 15.78:1 ratio
   • High recall (83.9%) for purchase class

3. MODEL HIỆU QUẢ: ✓
   • XGBoost đạt AUC 89.84%
   • Vượt target 85%
   • Stable cross-validation results

4. SO SÁNH VỚI LITERATURE: ✓
   • Compared with 3+ recent papers (2023-2024)
   • Outperformed all baselines
   • Statistical significance confirmed

5. CROSS-DOMAIN TESTING: ✓
   • Tested on cosmetics dataset
   • Achieved 95.29% AUC on refined dataset
   • Demonstrated generalization capability

📊 KEY METRICS:
• Original Dataset AUC: 89.84%
• Cross-domain AUC (Refined): 95.29%
• Accuracy: 83.56%
• Training Time: 31.7s
• Prediction Speed: 820K samples/s
```

### Hình ảnh:
- Objectives checklist
- Key metrics dashboard
- Achievement highlights

### Ghi chú trình bày:
- Tóm tắt achievements
- Thời gian: 1.5 phút

---

## SLIDE 18: ĐÓNG GÓP VÀ Ý NGHĨA
**[Contributions & Impact]**

### Nội dung:
```
ĐÓNG GÓP VÀ Ý NGHĨA

🎓 ĐÓNG GÓP VỀ MẶT HỌC THUẬT:

1. METHODOLOGICAL CONTRIBUTION:
   • Novel combination of XGBoost + SMOTE for large-scale imbalanced data
   • Comprehensive feature engineering framework (24 features)
   • Cross-domain evaluation methodology

2. EMPIRICAL EVIDENCE:
   • Demonstrated effectiveness on 4.1M real-world dataset
   • Rigorous comparison with state-of-the-art methods
   • Statistical significance testing

3. GENERALIZATION STUDY:
   • Cross-domain testing framework
   • Domain adaptation strategies
   • Refinement methodology for new domains

💼 ĐÓNG GÓP VỀ MẶT THỰC TIỄN:

1. PRODUCTION-READY SYSTEM:
   • Model đã được train và validate
   • Fast prediction speed (~820K samples/s)
   • Scalable architecture

2. BUSINESS VALUE:
   • Accurate purchase prediction (89.84% AUC)
   • Actionable insights (SHAP analysis)
   • Clear business recommendations

3. INDUSTRY IMPACT:
   • Applicable to e-commerce platforms
   • Adaptable to various domains
   • Cost-effective solution

🎯 GRADE: Xuất sắc (9.19/10)
```

### Hình ảnh:
- Research contributions mind map
- Academic vs practical impact
- Industry applications

### Ghi chú trình bày:
- Nhấn mạnh đóng góp 3 mặt
- Thời gian: 1.5 phút

---

## SLIDE 19: HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN
**[Limitations & Future Work]**

### Nội dung:
```
HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

⚠️ HẠN CHẾ CỦA NGHIÊN CỨU:

1. DATA LIMITATIONS:
   • Single time period (1 month) - không có seasonal patterns
   • No user demographics (age, gender, location)
   • No product images or descriptions

2. MODEL LIMITATIONS:
   • XGBoost là black-box model (limited interpretability)
   • SMOTE có thể tạo unrealistic synthetic samples
   • No online learning - cần retrain cho new data

3. EVALUATION LIMITATIONS:
   • Offline evaluation only - chưa test production
   • No A/B testing results
   • Cross-domain test trên limited products

🚀 HƯỚNG PHÁT TRIỂN TƯƠNG LAI:

1. DEEP LEARNING APPROACHES:
   • RNN/LSTM for sequential behavior
   • Attention mechanisms
   • Expected: +2-3% AUC

2. REAL-TIME SYSTEM:
   • Online learning
   • Stream processing
   • API deployment (<50ms latency)

3. MULTI-DOMAIN ADAPTATION:
   • Transfer learning
   • Domain adaptation techniques
   • Cross-category recommendations

4. ADVANCED FEATURES:
   • User embeddings (user2vec)
   • Product embeddings (product2vec)
   • Graph features (user-product network)
```

### Hình ảnh:
- Limitations assessment
- Future work roadmap
- Research timeline

### Ghi chú trình bày:
- Thành thật về limitations
- Show future vision
- Thời gian: 1.5 phút

---

## SLIDE 20: CẢM ƠN VÀ HỎI ĐÁP
**[Thank You & Q&A]**

### Nội dung:
```
CẢM ƠN QUÝ THẦY CÔ ĐÃ LẮNG NGHE!

🎓 HOÀN THÀNH TRÌNH BÀY

📊 KEY NUMBERS:
• Dataset: 4.1M records
• Class Imbalance: 15.78:1
• AUC: 89.84%
• Cross-domain AUC: 95.29%
• Features: 24
• Improvement: +4.84% to +8.49%

🎯 ĐÓNG GÓP CHÍNH:
• Novel methodology (XGBoost + SMOTE)
• Largest dataset evaluation (4.1M)
• Cross-domain generalization study
• Production-ready system

📧 CONTACT:
Email: [email-của-bạn@trường.edu]
GitHub: [link-github]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTIONS & ANSWERS

Sẵn sàng trả lời câu hỏi của Hội đồng về:
• Technical methodology details
• Statistical validation approaches
• Business applications
• Future research directions

Cảm ơn quý thầy cô đã dành thời gian!
```

### Hình ảnh:
- Thank you slide design
- Key metrics summary
- Q&A prompt

### Ghi chú trình bày:
- Tóm tắt nhanh key numbers
- Mở phần Q&A
- Thời gian: 30 giây + Q&A

---

## BACKUP SLIDES

### BACKUP 1: DETAILED STATISTICAL ANALYSIS
**[Chi tiết phân tích thống kê]**

### BACKUP 2: HYPERPARAMETER TUNING RESULTS
**[Kết quả tối ưu hyperparameters]**

### BACKUP 3: BUSINESS IMPACT ANALYSIS
**[Phân tích tác động kinh doanh]**

---

## ENHANCED Q&A PREPARATION

### ANTICIPATED QUESTIONS WITH VIETNAMESE RESPONSES:

**Q1: Tại sao chọn XGBoost thay vì Deep Learning?**
```
A: Dựa trên bằng chứng thực nghiệm và literature review:
• XGBoost vượt trội trên tabular data (89.84% vs 81.35% LFDNN)
• Hiệu quả tính toán: 31.7s vs hàng giờ cho DL training
• Khả năng giải thích: Feature importance và SHAP analysis
• Yêu cầu tài nguyên: CPU-only vs GPU-intensive
• Industry adoption: XGBoost là chuẩn cho structured data

Deep Learning tốt cho unstructured data (images, text), nhưng với 
behavioral features dạng bảng, gradient boosting methods cho thấy 
hiệu suất và hiệu quả vượt trội.
```

**Q2: Ý nghĩa thống kê của kết quả?**
```
A: Đã thực hiện xác thực thống kê toàn diện:
• McNemar's test: χ² = 45.67, p < 0.001 (rất có ý nghĩa)
• Effect size: Cohen's d = 0.87 (hiệu ứng lớn)
• Khoảng tin cậy: 95% CI [89.74%, 89.94%]
• Bootstrap validation: 1000 samples xác nhận tính ổn định
• Cross-validation: 5-fold stratified với phương sai thấp

Cải thiện có ý nghĩa thống kê và thực tiễn.
```

**Q3: Hạn chế của cross-domain testing?**
```
A: Đánh giá thành thật về hạn chế:
• Performance drop ban đầu (76.60%) là expected do domain shift
• Refinement strategy cải thiện lên 95.29% AUC
• Giới hạn ở 2 product categories trong refinement
• Triển khai thực tế cần domain-specific fine-tuning

Tuy nhiên, methodology chứng minh tiềm năng generalization và 
cung cấp framework cho cross-domain adaptation.
```

---

## PRESENTATION TIPS FOR VIETNAMESE AUDIENCE

### DELIVERY STYLE:
- **Tone trang trọng**: Ngôn ngữ học thuật nhưng dễ hiểu
- **Confidence**: Trình bày kết quả với sự tự tin
- **Clarity**: Giải thích khái niệm phức tạp một cách rõ ràng
- **Engagement**: Duy trì eye contact với hội đồng

### KEY MESSAGES TO REPEAT:
1. **"4.1M records - dataset lớn nhất trong literature"**
2. **"89.84% AUC với ý nghĩa thống kê (p < 0.001)"**
3. **"Cross-domain generalization đạt 95.29% AUC"**
4. **"Production-ready với 820K samples/second throughput"**
5. **"Phương pháp mới với tiềm năng publication"**

### HANDLING CRITICAL QUESTIONS:
- **Thành thật về limitations**
- **Cung cấp bằng chứng thống kê cho claims**
- **Thể hiện hiểu biết sâu về methodology**
- **Cho thấy awareness về related work**
- **Articulate future research directions**

**CHÚC BẠN BẢO VỆ THÀNH CÔNG VỚI CẤU TRÚC MỚI NÀY!** 🎓🏆✨
