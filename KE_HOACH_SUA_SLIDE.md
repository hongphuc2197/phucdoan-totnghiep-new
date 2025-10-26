# KẾ HOẠCH SỬA FILE SLIDE_TRINH_BAY.md

## YÊU CẦU CỦA THẦY

### I. Nguyên tắc slide luận văn:
- Slide = bản đồ, chỉ chứa **từ khóa**, **sơ đồ khối**, **kiến trúc** và **biểu đồ**
- **Logic**: Problem → Related Work → Method → Evidence

### II. Yêu cầu nội dung:
1. **Đặt vấn đề**: Nêu rõ Input/Output, Gap trong SOTA
2. **Hiện trạng**: Phân nhóm công trình, chỉ ra hạn chế
3. **Giải pháp**: Kiến trúc + đối sánh SOTA (giống/khác biệt)
4. **Bằng chứng**: Kết quả định lượng + định tính (case study)
5. **Footnotes**: Bắt buộc cho luận điểm, số liệu
6. **Tài liệu tham khảo**: Uy tín (Google Scholar)

---

## NHỮNG ĐÃ SỬA ✅

1. ✅ Mục lục (SLIDE 2): Đã cập nhật theo cấu trúc Problem → Related → Method → Evidence
2. ✅ SLIDE 3 (Đặt vấn đề): Đã tối giản, nêu Input/Output và Gap
3. ✅ SLIDE 4 (Mục tiêu): Đã tối giản
4. ✅ **Thêm mới**: SLIDE 5 - Hiện trạng nghiên cứu (Related Work)
5. ✅ **Thêm mới**: SLIDE 6 - Đối sánh với SOTA (giống/khác biệt)
6. ✅ SLIDE 7 (XGBoost & SMOTE): Đã tối giản

---

## CẦN SỬA TIẾP 🔧

### 1. Tối giản tất cả slides còn lại
**Nguyên tắc**: 
- Chỉ để TỪ KHÓA
- Không dùng câu dài
- Thêm sơ đồ flowchart thay vì mô tả

### 2. Slides cần sửa chi tiết:

#### **SLIDE 8: DATASET**
**Hiện tại**: Quá chi tiết (15 dòng)
**Nên**: 
```
DATASET

Source: Kaggle E-commerce [2]
Size: 4.1M records
• 500K users, 200K products
Imbalance: 15.78:1
```

#### **SLIDE 9-11: Feature Engineering, Methodology, Pipeline**
**Hiện tại**: Quá chi tiết
**Nên**: 
- Chỉ để từ khóa cho 7 nhóm features
- Flowchart chi tiết (không cần mô tả)

#### **SLIDE 11-13: Kết quả**
**Hiện tại**: Ổn nhưng cần thêm footnotes
**Cần thêm**: References cho số liệu

#### **CẦN THÊM**:
- **SLIDE mới**: Kết quả định tính (Case study 1-2 ví dụ)
- **SLIDE cuối**: Tài liệu tham khảo (References)

---

## ĐỀ XUẤT

### Cách 1: Sửa từng phần (hiện tại)
- Tiếp tục tối giản từng slide một
- Thêm slide mới theo yêu cầu
- Ưu điểm: Kiểm soát từng phần
- Nhược điểm: Mất thời gian

### Cách 2: Viết lại có chọn lọc
- Để nguyên các slide đã sửa
- Chỉ sửa các slides quan trọng nhất (8-20)
- Tạo 2 slides mới: Case study + References
- Ưu điểm: Nhanh hơn
- Nhược điểm: Không toàn diện

### Cách 3: Template sẵn
- Em tự sửa theo template đã cung cấp
- Chỉ em hiểu rõ nội dung cần highlight

---

## CẤU TRÚC MỚI ĐỀ XUẤT

**SLIDE 1**: Trang bìa
**SLIDE 2**: Mục lục ✅
**SLIDE 3**: Đặt vấn đề ✅
**SLIDE 4**: Mục tiêu ✅
**SLIDE 5**: Hiện trạng ✅
**SLIDE 6**: Đối sánh SOTA ✅
**SLIDE 7**: XGBoost & SMOTE ✅
**SLIDE 8**: Dataset (CẦN SỬA)
**SLIDE 9**: Feature Engineering (CẦN SỬA)
**SLIDE 10**: Phương pháp (CẦN SỬA)
**SLIDE 11**: Pipeline triển khai (CẦN SỬA)
**SLIDE 12**: So sánh models (Thêm footnotes)
**SLIDE 13**: Feature importance (Thêm footnotes)
**SLIDE 14**: ROC/PR curves (Thêm footnotes)
**SLIDE 15**: Cross-domain testing (Thêm footnotes)
**SLIDE 16**: Cross-domain results
**SLIDE 17**: Model interpretability
**SLIDE 18**: **THÊM MỚI - Case study (Kết quả định tính)**
**SLIDE 19**: Kết luận
**SLIDE 20**: Hạn chế & phát triển
**SLIDE 21**: **THÊM MỚI - Tài liệu tham khảo**
**SLIDE 22**: Cảm ơn & Q&A

---

## FOOTNOTE TEMPLATE

Cần thêm cho mỗi slide có số liệu, luận điểm:

```
[1] - Reference về class imbalance trong e-commerce
[2] - Kaggle dataset source
[3] - Collaborative Filtering paper (gốc)
[4] - Content-based paper
[5] - Wide & Deep paper
[6] - DeepFM paper
[7] - Hybrid systems review
[8] - Feature engineering for e-commerce
[9] - SMOTE original paper
[10] - XGBoost paper
[11] - Cross-domain recommendation paper
[12] - LFDNN paper (2023)
[13] - Deep Interest Network (2024)
[14] - User-based CF
[15] - Content-based filtering
[20] - RS literature về cross-domain testing
```

---

## BƯỚC TIẾP THEO

Bạn muốn tôi:
1. **Tiếp tục sửa từng slide** (chi tiết hơn)?
2. **Tạo 2 slides mới** (Case study + References) trước?
3. **Đưa ra template đầy đủ** để em tự điền?

Đề xuất: **Option 2** - Tạo 2 slides mới quan trọng nhất trước, sau đó quay lại tối giản các slides còn lại.

