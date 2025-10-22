import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING XGBOOST MODEL ON REAL COSMETICS DATASET")
print("="*80)

# Load the trained model and scaler
try:
    model = joblib.load('best_model_xgboost.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✓ Loaded trained XGBoost model and scaler successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load the real cosmetics dataset
try:
    cosmetics_df = pd.read_csv('real_cosmetics_dataset.csv')
    print(f"✓ Loaded cosmetics dataset: {cosmetics_df.shape}")
except Exception as e:
    print(f"✗ Error loading cosmetics dataset: {e}")
    exit(1)

print(f"\nCosmetics Dataset Overview:")
print(f"- Total interactions: {len(cosmetics_df):,}")
print(f"- Unique users: {cosmetics_df['user_id'].nunique():,}")
print(f"- Products: {cosmetics_df['product_name'].nunique()}")
print(f"- Purchase rate: {cosmetics_df['purchased'].mean():.3f}")

# Create features for the cosmetics dataset (same as original model)
print(f"\nCreating features for cosmetics dataset...")

# User-level features
user_features = cosmetics_df.groupby('user_id').agg({
    'purchased': ['sum', 'count', 'mean'],
    'price': ['mean', 'std', 'min', 'max'],
    'product_name': 'nunique',
    'session_duration': 'mean',
    'pages_viewed': 'mean'
}).reset_index()

user_features.columns = ['user_id', 'total_purchases', 'total_events', 'purchase_rate', 
                        'avg_price', 'price_std', 'min_price', 'max_price', 
                        'unique_products', 'avg_session_duration', 'avg_pages_viewed']

# Product-level features
product_features = cosmetics_df.groupby('product_name').agg({
    'purchased': ['sum', 'count', 'mean'],
    'price': 'mean',
    'user_id': 'nunique'
}).reset_index()

product_features.columns = ['product_name', 'product_purchases', 'product_views', 'product_purchase_rate', 
                           'product_price', 'unique_users']

# Category-level features
category_features = cosmetics_df.groupby('category').agg({
    'purchased': ['sum', 'count', 'mean'],
    'price': 'mean',
    'user_id': 'nunique'
}).reset_index()

category_features.columns = ['category', 'category_purchases', 'category_views', 'category_purchase_rate', 
                            'category_price', 'category_users']

# Merge features
cosmetics_main = cosmetics_df[['user_id', 'product_name', 'category', 'price', 'purchased', 
                              'age', 'gender', 'income_level', 'beauty_enthusiast', 'skin_type']].copy()

cosmetics_main = cosmetics_main.merge(user_features, on='user_id', how='left')
cosmetics_main = cosmetics_main.merge(product_features, on='product_name', how='left')
cosmetics_main = cosmetics_main.merge(category_features, on='category', how='left')

# Create additional features
cosmetics_main['price_ratio'] = cosmetics_main['price'] / (cosmetics_main['avg_price'] + 1e-8)
cosmetics_main['is_expensive'] = (cosmetics_main['price'] > cosmetics_main['avg_price']).astype(int)

# Encode categorical variables
cosmetics_main['gender_encoded'] = cosmetics_main['gender'].map({'F': 1, 'M': 0})
cosmetics_main['income_encoded'] = cosmetics_main['income_level'].map({'low': 0, 'medium': 1, 'high': 2})

# Create price categories
cosmetics_main['price_category'] = pd.cut(cosmetics_main['price'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
cosmetics_main['price_category_encoded'] = cosmetics_main['price_category'].cat.codes

# Fill missing values for numeric columns only
numeric_columns = cosmetics_main.select_dtypes(include=[np.number]).columns
cosmetics_main[numeric_columns] = cosmetics_main[numeric_columns].fillna(0)

# Fill missing values for categorical columns
categorical_columns = cosmetics_main.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    if col != 'price_category':  # Skip the categorical price category
        cosmetics_main[col] = cosmetics_main[col].fillna('unknown')

# Prepare features for model testing (exactly same features as original model)
# We need to map cosmetics features to original model features
feature_columns = ['price', 'total_purchases', 'total_events', 'purchase_rate', 
                  'avg_price', 'price_std', 'min_price', 'max_price', 
                  'unique_products', 'unique_categories', 'session_duration_days',
                  'product_purchases', 'product_views', 'product_purchase_rate', 
                  'product_price', 'unique_users', 'category_purchases', 
                  'category_views', 'category_purchase_rate', 'category_price', 
                  'category_users', 'price_ratio', 'is_expensive', 'price_category_encoded']

# Map cosmetics features to original model features
cosmetics_main['unique_categories'] = cosmetics_main['category'].nunique()  # Use category count
cosmetics_main['session_duration_days'] = cosmetics_main['avg_session_duration'] / (24 * 3600)  # Convert to days

# Fill missing values for additional features
cosmetics_main['age'] = cosmetics_main['age'].fillna(cosmetics_main['age'].mean())
cosmetics_main['beauty_enthusiast'] = cosmetics_main['beauty_enthusiast'].fillna(0)
cosmetics_main['gender_encoded'] = cosmetics_main['gender_encoded'].fillna(1)
cosmetics_main['income_encoded'] = cosmetics_main['income_encoded'].fillna(1)

X_cosmetics = cosmetics_main[feature_columns]
y_cosmetics = cosmetics_main['purchased']

print(f"✓ Features prepared:")
print(f"  - Feature matrix shape: {X_cosmetics.shape}")
print(f"  - Target distribution: {y_cosmetics.value_counts().to_dict()}")

# Scale features using the same scaler from training
X_cosmetics_scaled = scaler.transform(X_cosmetics)

# Make predictions
print(f"\nMaking predictions on cosmetics dataset...")
y_pred = model.predict(X_cosmetics_scaled)
y_pred_proba = model.predict_proba(X_cosmetics_scaled)[:, 1]

# Calculate metrics
auc_score = roc_auc_score(y_cosmetics, y_pred_proba)

print(f"\n" + "="*80)
print("MODEL PERFORMANCE ON COSMETICS DATASET")
print("="*80)

print(f"AUC Score: {auc_score:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_cosmetics, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_cosmetics, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Analyze performance by product
print(f"\n" + "="*80)
print("PERFORMANCE ANALYSIS BY PRODUCT")
print("="*80)

cosmetics_main['predicted'] = y_pred
cosmetics_main['predicted_proba'] = y_pred_proba

product_analysis = cosmetics_main.groupby('product_name').agg({
    'purchased': ['count', 'sum', 'mean'],
    'predicted': 'sum',
    'predicted_proba': 'mean',
    'price': 'mean'
}).round(3)

product_analysis.columns = ['total_interactions', 'actual_purchases', 'actual_purchase_rate', 
                           'predicted_purchases', 'avg_prediction_prob', 'avg_price']

# Calculate accuracy for each product
def calculate_accuracy(group):
    return (group['purchased'] == group['predicted']).mean()

product_accuracy = cosmetics_main.groupby('product_name').apply(calculate_accuracy)
product_analysis['prediction_accuracy'] = product_accuracy.round(3)

product_analysis = product_analysis.sort_values('actual_purchases', ascending=False)

print("Top 5 Products Performance:")
for idx, (product, row) in enumerate(product_analysis.head(5).iterrows(), 1):
    print(f"\n{idx}. {product}")
    print(f"   - Actual purchases: {row['actual_purchases']:.0f}")
    print(f"   - Predicted purchases: {row['predicted_purchases']:.0f}")
    print(f"   - Actual purchase rate: {row['actual_purchase_rate']:.3f}")
    print(f"   - Avg prediction probability: {row['avg_prediction_prob']:.3f}")
    print(f"   - Prediction accuracy: {row['prediction_accuracy']:.3f}")
    print(f"   - Avg price: ${row['avg_price']:.2f}")

# Analyze performance by category
print(f"\n" + "="*80)
print("PERFORMANCE ANALYSIS BY CATEGORY")
print("="*80)

category_analysis = cosmetics_main.groupby('category').agg({
    'purchased': ['count', 'sum', 'mean'],
    'predicted': 'sum',
    'predicted_proba': 'mean'
}).round(3)

category_analysis.columns = ['total_interactions', 'actual_purchases', 'actual_purchase_rate', 
                            'predicted_purchases', 'avg_prediction_prob']

print("Category Performance:")
for category, row in category_analysis.iterrows():
    print(f"\n{category}:")
    print(f"   - Actual purchases: {row['actual_purchases']:.0f}")
    print(f"   - Predicted purchases: {row['predicted_purchases']:.0f}")
    print(f"   - Actual purchase rate: {row['actual_purchase_rate']:.3f}")
    print(f"   - Avg prediction probability: {row['avg_prediction_prob']:.3f}")

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# ROC Curve
fpr, tpr, _ = roc_curve(y_cosmetics, y_pred_proba)
ax1.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve - Cosmetics Dataset')
ax1.legend()
ax1.grid(True)

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix')

# Product Performance Comparison
top_5_products = product_analysis.head(5)
x_pos = np.arange(len(top_5_products))
width = 0.35

ax3.bar(x_pos - width/2, top_5_products['actual_purchases'], width, label='Actual', alpha=0.7)
ax3.bar(x_pos + width/2, top_5_products['predicted_purchases'], width, label='Predicted', alpha=0.7)
ax3.set_xlabel('Products')
ax3.set_ylabel('Number of Purchases')
ax3.set_title('Actual vs Predicted Purchases (Top 5 Products)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in top_5_products.index], rotation=45)
ax3.legend()

# Category Performance
category_performance = category_analysis.sort_values('actual_purchases', ascending=True)
ax4.barh(range(len(category_performance)), category_performance['actual_purchases'], alpha=0.7, label='Actual')
ax4.barh(range(len(category_performance)), category_performance['predicted_purchases'], alpha=0.7, label='Predicted')
ax4.set_yticks(range(len(category_performance)))
ax4.set_yticklabels(category_performance.index)
ax4.set_xlabel('Number of Purchases')
ax4.set_title('Category Performance')
ax4.legend()

plt.tight_layout()
plt.savefig('cosmetics_model_test_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Model compatibility assessment
print(f"\n" + "="*80)
print("MODEL COMPATIBILITY ASSESSMENT")
print("="*80)

# Calculate compatibility metrics
actual_purchase_rate = y_cosmetics.mean()
predicted_purchase_rate = y_pred.mean()
prediction_accuracy = (y_cosmetics == y_pred).mean()

print(f"Actual purchase rate: {actual_purchase_rate:.3f}")
print(f"Predicted purchase rate: {predicted_purchase_rate:.3f}")
print(f"Overall prediction accuracy: {prediction_accuracy:.3f}")

# Determine if model is suitable
if auc_score > 0.7 and prediction_accuracy > 0.8:
    compatibility = "HIGH"
    recommendation = "Model performs well on cosmetics dataset"
elif auc_score > 0.6 and prediction_accuracy > 0.7:
    compatibility = "MEDIUM"
    recommendation = "Model shows moderate performance, consider fine-tuning"
else:
    compatibility = "LOW"
    recommendation = "Model needs significant improvement for cosmetics domain"

print(f"\nCompatibility Level: {compatibility}")
print(f"Recommendation: {recommendation}")

# Save results
results_summary = {
    'dataset': 'Real Cosmetics Dataset',
    'auc_score': auc_score,
    'prediction_accuracy': prediction_accuracy,
    'actual_purchase_rate': actual_purchase_rate,
    'predicted_purchase_rate': predicted_purchase_rate,
    'compatibility': compatibility,
    'recommendation': recommendation
}

import json
with open('cosmetics_test_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✓ Results saved to 'cosmetics_test_results.json'")
print(f"✓ Visualization saved as 'cosmetics_model_test_results.png'")
print(f"\nModel testing on cosmetics dataset completed!")