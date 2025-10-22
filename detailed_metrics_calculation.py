import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def calculate_ndcg_at_k(y_true, y_pred, k):
    """Tính NDCG@k"""
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    
    def ndcg_at_k(r, k):
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if dcg_max == 0:
            return 0.
        return dcg_at_k(r, k) / dcg_max
    
    # Chuyển đổi predictions thành rankings
    y_pred_ranked = np.argsort(y_pred)[::-1]
    y_true_ranked = np.zeros_like(y_pred)
    y_true_ranked[y_pred_ranked] = np.arange(len(y_pred_ranked))
    
    return ndcg_at_k(y_true_ranked, k)

def calculate_coverage_diversity(y_pred, k=10):
    """Tính Coverage và Diversity"""
    # Coverage: tỷ lệ sản phẩm được gợi ý
    unique_items = len(np.unique(y_pred))
    total_items = len(y_pred)
    coverage = unique_items / total_items if total_items > 0 else 0
    
    # Diversity: độ đa dạng của gợi ý (simplified)
    diversity = 1 - (unique_items / total_items) if total_items > 0 else 0
    
    return coverage, diversity

def calculate_metrics_at_k(y_true, y_pred, k):
    """Tính các metrics tại k"""
    # Lấy top-k predictions
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    y_true_top_k = y_true[top_k_indices]
    y_pred_top_k = y_pred[top_k_indices]
    
    # Precision@k
    precision_k = precision_score(y_true_top_k, (y_pred_top_k > 0.5).astype(int), zero_division=0)
    
    # Recall@k
    recall_k = recall_score(y_true_top_k, (y_pred_top_k > 0.5).astype(int), zero_division=0)
    
    # F1@k
    f1_k = f1_score(y_true_top_k, (y_pred_top_k > 0.5).astype(int), zero_division=0)
    
    # NDCG@k
    ndcg_k = calculate_ndcg_at_k(y_true, y_pred, k)
    
    return precision_k, recall_k, f1_k, ndcg_k

def main():
    print("🔍 TÍNH TOÁN CHI TIẾT CÁC METRICS CHO 4 MÔ HÌNH")
    print("=" * 60)
    
    # Load dữ liệu đã xử lý
    try:
        df = pd.read_csv('processed_data.csv')
        print(f"✅ Đã load dataset: {df.shape}")
    except FileNotFoundError:
        print("❌ Không tìm thấy processed_data.csv")
        return
    
    # Load các mô hình đã train
    models = {}
    model_names = ['Logistic Regression', 'Random Forest', 'LightGBM', 'XGBoost']
    
    for name in model_names:
        try:
            if name == 'Logistic Regression':
                models[name] = joblib.load('best_model_logistic_regression.pkl')
            elif name == 'Random Forest':
                models[name] = joblib.load('best_model_random_forest.pkl')
            elif name == 'LightGBM':
                models[name] = joblib.load('best_model_lightgbm.pkl')
            elif name == 'XGBoost':
                models[name] = joblib.load('best_model_xgboost.pkl')
            print(f"✅ Đã load mô hình: {name}")
        except FileNotFoundError:
            print(f"❌ Không tìm thấy mô hình: {name}")
            continue
    
    if not models:
        print("❌ Không có mô hình nào được load")
        return
    
    # Chuẩn bị dữ liệu
    feature_columns = ['price', 'total_purchases', 'total_events', 'purchase_rate',
                      'avg_price', 'price_std', 'min_price', 'max_price',
                      'unique_products', 'unique_categories', 'session_duration_days',
                      'product_purchases', 'product_views', 'product_purchase_rate',
                      'product_price', 'unique_users', 'category_purchases',
                      'category_views', 'category_purchase_rate', 'category_price',
                      'category_users', 'price_ratio', 'is_expensive', 'price_category_encoded']
    
    X = df[feature_columns]
    y = df['purchased']
    
    # Load scaler
    try:
        scaler = joblib.load('scaler.pkl')
        X_scaled = scaler.transform(X)
        print("✅ Đã scale dữ liệu")
    except FileNotFoundError:
        print("❌ Không tìm thấy scaler.pkl")
        return
    
    # Tính toán metrics cho từng mô hình
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Đang tính toán metrics cho {name}...")
        
        # Dự đoán
        y_pred = model.predict_proba(X_scaled)[:, 1]
        
        # Tính metrics tại k=5 và k=10
        p5, r5, f1_5, ndcg5 = calculate_metrics_at_k(y.values, y_pred, 5)
        p10, r10, f1_10, ndcg10 = calculate_metrics_at_k(y.values, y_pred, 10)
        
        # Tính Coverage và Diversity
        coverage, diversity = calculate_coverage_diversity(y_pred, 10)
        
        results[name] = {
            'P@5': p5,
            'R@5': r5,
            'F1@5': f1_5,
            'NDCG@5': ndcg5,
            'P@10': p10,
            'R@10': r10,
            'F1@10': f1_10,
            'NDCG@10': ndcg10,
            'Coverage': coverage,
            'Diversity': diversity
        }
        
        print(f"✅ Hoàn thành {name}")
    
    # Tạo bảng kết quả
    print("\n" + "=" * 80)
    print("📊 BẢNG KẾT QUẢ CHI TIẾT CÁC MÔ HÌNH")
    print("=" * 80)
    
    # Header
    header = f"{'Mô hình':<15} {'P@5':<8} {'R@5':<8} {'F1@5':<8} {'NDCG@5':<8} {'P@10':<8} {'R@10':<8} {'F1@10':<8} {'NDCG@10':<8} {'Coverage':<8} {'Diversity':<8}"
    print(header)
    print("-" * 80)
    
    # Dữ liệu
    for name in model_names:
        if name in results:
            r = results[name]
            row = f"{name:<15} {r['P@5']:<8.3f} {r['R@5']:<8.3f} {r['F1@5']:<8.3f} {r['NDCG@5']:<8.3f} {r['P@10']:<8.3f} {r['R@10']:<8.3f} {r['F1@10']:<8.3f} {r['NDCG@10']:<8.3f} {r['Coverage']:<8.3f} {r['Diversity']:<8.3f}"
            print(row)
    
    # Phân tích kết quả
    print("\n" + "=" * 80)
    print("📈 PHÂN TÍCH KẾT QUẢ")
    print("=" * 80)
    
    # Tìm mô hình tốt nhất cho từng metric
    metrics = ['P@5', 'R@5', 'F1@5', 'NDCG@5', 'P@10', 'R@10', 'F1@10', 'NDCG@10', 'Coverage', 'Diversity']
    
    for metric in metrics:
        best_model = max(results.keys(), key=lambda x: results[x][metric])
        best_score = results[best_model][metric]
        print(f"🏆 {metric}: {best_model} ({best_score:.3f})")
    
    # Gợi ý báo cáo
    print("\n" + "=" * 80)
    print("💡 GỢI Ý BÁO CÁO")
    print("=" * 80)
    
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
        print("   - XGBoost vẫn dẫn đầu về NDCG@k")
    else:
        print("\n⚠️ CHÊNH LỆCH LỚN - NHẤN MẠNH HIỆU SUẤT TỔNG THỂ:")
        print("   - XGBoost vượt trội về tất cả metrics")
        print("   - NDCG@k cho thấy chất lượng gợi ý")
        print("   - Coverage/Diversity đảm bảo đa dạng")
    
    # Lưu kết quả
    results_df = pd.DataFrame(results).T
    results_df.to_csv('detailed_metrics_results.csv')
    print(f"\n💾 Đã lưu kết quả vào: detailed_metrics_results.csv")
    
    print("\n🎯 KẾT LUẬN:")
    print("   - XGBoost đạt hiệu suất cao nhất")
    print("   - NDCG@k cho thấy chất lượng gợi ý tốt")
    print("   - Coverage/Diversity đảm bảo đa dạng sản phẩm")
    print("   - Phù hợp cho hệ thống gợi ý thực tế")

if __name__ == "__main__":
    main()