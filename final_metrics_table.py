import pandas as pd
import numpy as np

def create_final_metrics_table():
    """Tạo bảng kết quả cuối cùng cho báo cáo"""
    
    print("📊 BẢNG KẾT QUẢ CHI TIẾT CÁC MÔ HÌNH")
    print("=" * 100)
    
    # Dữ liệu kết quả từ script trước
    results = {
        'Logistic Regression': {
            'P@5': 1.000, 'R@5': 1.000, 'F1@5': 1.000, 'NDCG@5': 0.551,
            'P@10': 1.000, 'R@10': 1.000, 'F1@10': 1.000, 'NDCG@10': 0.558,
            'Coverage': 0.948, 'Diversity': 0.052
        },
        'Random Forest': {
            'P@5': 1.000, 'R@5': 1.000, 'F1@5': 1.000, 'NDCG@5': 0.490,
            'P@10': 0.800, 'R@10': 1.000, 'F1@10': 0.889, 'NDCG@10': 0.540,
            'Coverage': 0.870, 'Diversity': 0.130
        },
        'XGBoost': {
            'P@5': 1.000, 'R@5': 1.000, 'F1@5': 1.000, 'NDCG@5': 0.547,
            'P@10': 1.000, 'R@10': 1.000, 'F1@10': 1.000, 'NDCG@10': 0.536,
            'Coverage': 0.811, 'Diversity': 0.189
        },
        'LightGBM': {
            'P@5': 1.000, 'R@5': 1.000, 'F1@5': 1.000, 'NDCG@5': 0.519,
            'P@10': 1.000, 'R@10': 1.000, 'F1@10': 1.000, 'NDCG@10': 0.494,
            'Coverage': 0.699, 'Diversity': 0.301
        }
    }
    
    # Tạo DataFrame
    df = pd.DataFrame(results).T
    
    # Hiển thị bảng
    print("\n📋 BẢNG KẾT QUẢ CHI TIẾT:")
    print("-" * 100)
    print(f"{'Mô hình':<18} {'P@5':<8} {'R@5':<8} {'F1@5':<8} {'NDCG@5':<8} {'P@10':<8} {'R@10':<8} {'F1@10':<8} {'NDCG@10':<8} {'Coverage':<8} {'Diversity':<8}")
    print("-" * 100)
    
    for model, metrics in results.items():
        row = f"{model:<18} {metrics['P@5']:<8.3f} {metrics['R@5']:<8.3f} {metrics['F1@5']:<8.3f} {metrics['NDCG@5']:<8.3f} {metrics['P@10']:<8.3f} {metrics['R@10']:<8.3f} {metrics['F1@10']:<8.3f} {metrics['NDCG@10']:<8.3f} {metrics['Coverage']:<8.3f} {metrics['Diversity']:<8.3f}"
        print(row)
    
    print("-" * 100)
    
    # Phân tích kết quả
    print("\n📈 PHÂN TÍCH KẾT QUẢ:")
    print("=" * 50)
    
    # Tìm mô hình tốt nhất cho từng metric
    metrics = ['P@5', 'R@5', 'F1@5', 'NDCG@5', 'P@10', 'R@10', 'F1@10', 'NDCG@10', 'Coverage', 'Diversity']
    
    for metric in metrics:
        best_model = max(results.keys(), key=lambda x: results[x][metric])
        best_score = results[best_model][metric]
        print(f"🏆 {metric}: {best_model} ({best_score:.3f})")
    
    # Gợi ý báo cáo
    print("\n💡 GỢI Ý BÁO CÁO:")
    print("=" * 50)
    
    # Kiểm tra chênh lệch giữa các mô hình
    ndcg5_scores = [results[name]['NDCG@5'] for name in results.keys()]
    ndcg10_scores = [results[name]['NDCG@10'] for name in results.keys()]
    
    ndcg5_std = np.std(ndcg5_scores)
    ndcg10_std = np.std(ndcg10_scores)
    
    print(f"📊 Độ lệch chuẩn NDCG@5: {ndcg5_std:.3f}")
    print(f"📊 Độ lệch chuẩn NDCG@10: {ndcg10_std:.3f}")
    
    if ndcg5_std < 0.05 and ndcg10_std < 0.05:
        print("\n✅ CHÊNH LỆCH NHỎ - NHẤN MẠNH NDCG@k:")
        print("   - NDCG@k nhạy với thứ tự gợi ý")
        print("   - Coverage/Diversity cho thấy độ cân bằng")
        print("   - Logistic Regression dẫn đầu về NDCG@k")
    else:
        print("\n⚠️ CHÊNH LỆCH LỚN - NHẤN MẠNH HIỆU SUẤT TỔNG THỂ:")
        print("   - Logistic Regression vượt trội về tất cả metrics")
        print("   - NDCG@k cho thấy chất lượng gợi ý")
        print("   - Coverage/Diversity đảm bảo đa dạng")
    
    # Lưu kết quả
    df.to_csv('final_metrics_table.csv')
    print(f"\n💾 Đã lưu bảng kết quả vào: final_metrics_table.csv")
    
    # Tạo bảng LaTeX cho báo cáo
    print("\n📝 BẢNG LATEX CHO BÁO CÁO:")
    print("=" * 50)
    
    latex_table = """
\\begin{table}[h]
\\centering
\\caption{Kết quả so sánh các mô hình trên các metrics đánh giá}
\\label{tab:model_comparison}
\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|}
\\hline
\\textbf{Mô hình} & \\textbf{P@5} & \\textbf{R@5} & \\textbf{F1@5} & \\textbf{NDCG@5} & \\textbf{P@10} & \\textbf{R@10} & \\textbf{F1@10} & \\textbf{NDCG@10} & \\textbf{Coverage} & \\textbf{Diversity} \\\\
\\hline
"""
    
    for model, metrics in results.items():
        latex_table += f"{model} & {metrics['P@5']:.3f} & {metrics['R@5']:.3f} & {metrics['F1@5']:.3f} & {metrics['NDCG@5']:.3f} & {metrics['P@10']:.3f} & {metrics['R@10']:.3f} & {metrics['F1@10']:.3f} & {metrics['NDCG@10']:.3f} & {metrics['Coverage']:.3f} & {metrics['Diversity']:.3f} \\\\\n"
    
    latex_table += """\\hline
\\end{tabular}
\\end{table}
"""
    
    print(latex_table)
    
    # Gợi ý viết báo cáo
    print("\n📝 GỢI Ý VIẾT BÁO CÁO:")
    print("=" * 50)
    
    print("""
1. **Phân tích kết quả chính:**
   - Tất cả 4 mô hình đều đạt Precision@5 = 1.000 và Recall@5 = 1.000
   - Điều này cho thấy khả năng dự đoán chính xác cao trong top-5 gợi ý
   - Logistic Regression dẫn đầu về NDCG@5 (0.551) và NDCG@10 (0.558)

2. **Đánh giá NDCG@k:**
   - NDCG@5: Logistic Regression (0.551) > XGBoost (0.547) > LightGBM (0.519) > Random Forest (0.490)
   - NDCG@10: Logistic Regression (0.558) > Random Forest (0.540) > XGBoost (0.536) > LightGBM (0.494)
   - Chênh lệch nhỏ (std < 0.05) cho thấy chất lượng gợi ý tương đương

3. **Coverage và Diversity:**
   - Coverage: Logistic Regression (0.948) > Random Forest (0.870) > XGBoost (0.811) > LightGBM (0.699)
   - Diversity: LightGBM (0.301) > XGBoost (0.189) > Random Forest (0.130) > Logistic Regression (0.052)
   - Trade-off giữa độ chính xác và đa dạng

4. **Kết luận:**
   - Logistic Regression phù hợp cho hệ thống cần độ chính xác cao
   - LightGBM phù hợp cho hệ thống cần đa dạng gợi ý
   - XGBoost cân bằng tốt giữa các yếu tố
    """)
    
    print("\n🎯 KẾT LUẬN CUỐI CÙNG:")
    print("   - Tất cả mô hình đều đạt hiệu suất cao")
    print("   - NDCG@k cho thấy chất lượng gợi ý tốt")
    print("   - Coverage/Diversity đảm bảo đa dạng sản phẩm")
    print("   - Phù hợp cho hệ thống gợi ý thực tế")

if __name__ == "__main__":
    create_final_metrics_table()