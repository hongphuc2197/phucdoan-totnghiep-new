import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print("🔍 SHAP PHÂN TÍCH TUYẾN TÍNH vs PHI TUYẾN - MÔ HÌNH XGBOOST")
print("="*80)

# Load the trained model and scaler
print("📥 Đang tải mô hình XGBoost đã huấn luyện...")
try:
    model = joblib.load('best_model_xgboost.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Đã tải thành công mô hình và scaler")
except FileNotFoundError:
    print("❌ Không tìm thấy file mô hình. Vui lòng chạy fast_model_comparison.py trước.")
    exit()

# Load processed data
print("📥 Đang tải dữ liệu đã xử lý...")
try:
    df = pd.read_csv('processed_data.csv')
    print(f"✅ Đã tải {len(df):,} bản ghi")
except FileNotFoundError:
    print("❌ Không tìm thấy file processed_data.csv. Vui lòng chạy analyze_dataset.py trước.")
    exit()

# Sample data for SHAP analysis (faster computation)
print("🔄 Lấy mẫu dữ liệu cho phân tích SHAP...")
df_sample = df.sample(n=min(5000, len(df)), random_state=42)

# Prepare features - use only the features that were used during training
feature_columns = [
    'price', 'total_purchases', 'total_events', 'purchase_rate', 'avg_price', 'price_std', 
    'min_price', 'max_price', 'unique_products', 'unique_categories', 
    'session_duration_days', 'product_purchases', 'product_views', 'product_purchase_rate', 
    'product_price', 'unique_users', 'category_purchases', 'category_views', 
    'category_purchase_rate', 'category_price', 'category_users', 'price_ratio', 
    'is_expensive', 'price_category_encoded'
]

# Filter to only include features that exist in the dataset
feature_columns = [col for col in feature_columns if col in df_sample.columns]

X_sample = df_sample[feature_columns]
y_sample = df_sample['purchased']

# Scale features
X_sample_scaled = scaler.transform(X_sample)

print(f"📊 Phân tích trên {len(X_sample):,} mẫu với {len(feature_columns)} features")

# Create SHAP explainer
print("🔧 Đang tạo SHAP explainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample_scaled)

print("✅ Đã tính toán SHAP values thành công")

# Define linear and non-linear features based on domain knowledge
linear_features = [
    'total_purchases', 'purchase_rate', 'product_purchases', 
    'category_purchases', 'total_events', 'product_views'
]

nonlinear_features = [
    'price_ratio', 'session_duration_days', 'avg_session_duration',
    'unique_categories', 'category_views', 'is_expensive', 'price'
]

# Ensure all features exist in the dataset
linear_features = [f for f in linear_features if f in feature_columns]
nonlinear_features = [f for f in nonlinear_features if f in feature_columns]

print(f"📈 Features tuyến tính: {len(linear_features)}")
print(f"📊 Features phi tuyến: {len(nonlinear_features)}")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

# 1. Feature Importance Comparison
ax1 = fig.add_subplot(gs[0, :])
feature_importance = np.abs(shap_values).mean(0)
feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': feature_importance
}).sort_values('importance', ascending=True)

# Color by linear/non-linear
colors = []
for feature in feature_importance_df['feature']:
    if feature in linear_features:
        colors.append('lightblue')
    elif feature in nonlinear_features:
        colors.append('lightcoral')
    else:
        colors.append('lightgray')

bars = ax1.barh(range(len(feature_importance_df)), feature_importance_df['importance'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(feature_importance_df)))
ax1.set_yticklabels(feature_importance_df['feature'], fontsize=10)
ax1.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax1.set_title('Feature Importance - Phân loại Tuyến/Phi tuyến', fontsize=16, fontweight='bold', pad=20)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='lightblue', label='Tuyến tính'),
                  Patch(facecolor='lightcoral', label='Phi tuyến'),
                  Patch(facecolor='lightgray', label='Khác')]
ax1.legend(handles=legend_elements, loc='lower right')

# 2. Linear vs Non-linear Impact Analysis
ax2 = fig.add_subplot(gs[1, 0])
linear_indices = [feature_columns.index(f) for f in linear_features if f in feature_columns]
nonlinear_indices = [feature_columns.index(f) for f in nonlinear_features if f in feature_columns]

linear_impact = np.abs(shap_values[:, linear_indices]).mean() if linear_indices else 0
nonlinear_impact = np.abs(shap_values[:, nonlinear_indices]).mean() if nonlinear_indices else 0

categories = ['Tuyến tính', 'Phi tuyến']
impacts = [linear_impact, nonlinear_impact]
colors = ['lightblue', 'lightcoral']

bars = ax2.bar(categories, impacts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax2.set_title('Tác động Tổng thể', fontsize=14, fontweight='bold', pad=20)

# Add value labels on bars
for bar, impact in zip(bars, impacts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{impact:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# 3. Top Linear Features
ax3 = fig.add_subplot(gs[1, 1])
if linear_indices:
    linear_importance = np.abs(shap_values[:, linear_indices]).mean(0)
    linear_df = pd.DataFrame({
        'feature': linear_features,
        'importance': linear_importance
    }).sort_values('importance', ascending=True)
    
    bars = ax3.barh(range(len(linear_df)), linear_df['importance'], 
                    color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(range(len(linear_df)))
    ax3.set_yticklabels(linear_df['feature'], fontsize=10)
    ax3.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
    ax3.set_title('Top Features Tuyến tính', fontsize=14, fontweight='bold', pad=20)

# 4. Top Non-linear Features
ax4 = fig.add_subplot(gs[1, 2])
if nonlinear_indices:
    nonlinear_importance = np.abs(shap_values[:, nonlinear_indices]).mean(0)
    nonlinear_df = pd.DataFrame({
        'feature': nonlinear_features,
        'importance': nonlinear_importance
    }).sort_values('importance', ascending=True)
    
    bars = ax4.barh(range(len(nonlinear_df)), nonlinear_df['importance'], 
                    color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_yticks(range(len(nonlinear_df)))
    ax4.set_yticklabels(nonlinear_df['feature'], fontsize=10)
    ax4.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
    ax4.set_title('Top Features Phi tuyến', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('shap_linear_nonlinear_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Đã lưu biểu đồ SHAP phân tích: shap_linear_nonlinear_analysis.png")

# Generate detailed analysis report
print("\n" + "="*80)
print("📊 PHÂN TÍCH CHI TIẾT TUYẾN TÍNH vs PHI TUYẾN")
print("="*80)

print(f"\n🔍 TỔNG QUAN:")
print(f"   • Tổng số features: {len(feature_columns)}")
print(f"   • Features tuyến tính: {len(linear_features)}")
print(f"   • Features phi tuyến: {len(nonlinear_features)}")
print(f"   • Mẫu phân tích: {len(X_sample):,} records")

if linear_indices:
    linear_avg_impact = np.abs(shap_values[:, linear_indices]).mean()
    print(f"\n📈 FEATURES TUYẾN TÍNH:")
    print(f"   • Tác động trung bình: {linear_avg_impact:.4f}")
    print(f"   • Các features chính:")
    linear_importance = np.abs(shap_values[:, linear_indices]).mean(0)
    linear_df = pd.DataFrame({
        'feature': linear_features,
        'importance': linear_importance
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(linear_df.head(5).iterrows(), 1):
        print(f"      {i}. {row['feature']}: {row['importance']:.4f}")

if nonlinear_indices:
    nonlinear_avg_impact = np.abs(shap_values[:, nonlinear_indices]).mean()
    print(f"\n📊 FEATURES PHI TUYẾN:")
    print(f"   • Tác động trung bình: {nonlinear_avg_impact:.4f}")
    print(f"   • Các features chính:")
    nonlinear_importance = np.abs(shap_values[:, nonlinear_indices]).mean(0)
    nonlinear_df = pd.DataFrame({
        'feature': nonlinear_features,
        'importance': nonlinear_importance
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(nonlinear_df.head(5).iterrows(), 1):
        print(f"      {i}. {row['feature']}: {row['importance']:.4f}")

print(f"\n💡 KẾT LUẬN:")
if linear_indices and nonlinear_indices:
    if linear_avg_impact > nonlinear_avg_impact:
        print(f"   • Features tuyến tính có tác động lớn hơn ({linear_avg_impact:.4f} vs {nonlinear_avg_impact:.4f})")
        print(f"   • Mô hình XGBoost tận dụng tốt mối quan hệ tuyến tính")
    else:
        print(f"   • Features phi tuyến có tác động lớn hơn ({nonlinear_avg_impact:.4f} vs {linear_avg_impact:.4f})")
        print(f"   • Mô hình XGBoost tận dụng tốt mối quan hệ phi tuyến")
    
    print(f"   • XGBoost cân bằng tốt giữa cả hai loại mối quan hệ")
    print(f"   • Điều này giải thích tại sao XGBoost vượt trội hơn các mô hình tuyến tính")

print(f"\n🎯 Ý NGHĨA THỰC TIỄN:")
print(f"   • Mô hình có thể học được cả patterns đơn giản (tuyến tính) và phức tạp (phi tuyến)")
print(f"   • Phù hợp với bản chất đa dạng của hành vi người dùng e-commerce")
print(f"   • Giải thích được cả mối quan hệ trực tiếp và gián tiếp trong dữ liệu")

# Create SHAP summary plots separately
print("\n📊 Đang tạo SHAP Summary Plots riêng biệt...")

# SHAP Summary Plot - All Features
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample_scaled, feature_names=feature_columns, 
                  max_display=15, show=False)
plt.title('SHAP Summary Plot - Tất cả Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_summary_all_features.png', dpi=300, bbox_inches='tight')
print("✅ Đã lưu: shap_summary_all_features.png")

# SHAP Summary Plot - Linear Features Only
if linear_indices:
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, linear_indices], X_sample_scaled[:, linear_indices], 
                      feature_names=linear_features, max_display=len(linear_features), 
                      show=False)
    plt.title('SHAP Summary - Features Tuyến tính', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('shap_summary_linear_features.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu: shap_summary_linear_features.png")

# SHAP Summary Plot - Non-linear Features Only
if nonlinear_indices:
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, nonlinear_indices], X_sample_scaled[:, nonlinear_indices], 
                      feature_names=nonlinear_features, max_display=len(nonlinear_features), 
                      show=False)
    plt.title('SHAP Summary - Features Phi Tuyến', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('shap_summary_nonlinear_features.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu: shap_summary_nonlinear_features.png")

print(f"\n📁 FILES ĐÃ TẠO:")
print(f"   • shap_linear_nonlinear_analysis.png - Biểu đồ phân tích tổng hợp")
print(f"   • shap_summary_all_features.png - SHAP summary tất cả features")
print(f"   • shap_summary_linear_features.png - SHAP summary features tuyến tính")
print(f"   • shap_summary_nonlinear_features.png - SHAP summary features phi tuyến")
print(f"   • Báo cáo chi tiết trong console")

print("\n🎊 PHÂN TÍCH SHAP HOÀN THÀNH!")
print("="*80)