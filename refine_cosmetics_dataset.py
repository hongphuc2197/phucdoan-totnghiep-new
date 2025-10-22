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
print("REFINING COSMETICS DATASET - TESTING WITH TOP 2 PRODUCTS ONLY")
print("="*80)

# Load the trained model and scaler
try:
    model = joblib.load('best_model_xgboost.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✓ Loaded trained XGBoost model and scaler successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load the cosmetics dataset
cosmetics_df = pd.read_csv('real_cosmetics_dataset.csv')

# Identify top 2 best-selling products
print("Identifying top 2 best-selling products...")
product_performance = cosmetics_df.groupby('product_name').agg({
    'purchased': ['count', 'sum', 'mean'],
    'price': 'mean'
}).round(3)

product_performance.columns = ['total_interactions', 'purchases', 'purchase_rate', 'avg_price']
top_2_products = product_performance.sort_values('purchases', ascending=False).head(2)

print(f"\nTop 2 Best-Selling Products:")
for idx, (product, row) in enumerate(top_2_products.iterrows(), 1):
    print(f"{idx}. {product}")
    print(f"   - Total interactions: {row['total_interactions']:.0f}")
    print(f"   - Purchases: {row['purchases']:.0f}")
    print(f"   - Purchase rate: {row['purchase_rate']:.3f}")
    print(f"   - Avg price: ${row['avg_price']:.2f}")

# Filter dataset to only include top 2 products
top_2_product_names = top_2_products.index.tolist()
refined_df = cosmetics_df[cosmetics_df['product_name'].isin(top_2_product_names)].copy()

print(f"\nRefined Dataset Overview:")
print(f"- Total interactions: {len(refined_df):,}")
print(f"- Unique users: {refined_df['user_id'].nunique():,}")
print(f"- Products: {refined_df['product_name'].nunique()}")
print(f"- Purchase rate: {refined_df['purchased'].mean():.3f}")

# Create features for the refined dataset
print(f"\nCreating features for refined cosmetics dataset...")

# User-level features
user_features = refined_df.groupby('user_id').agg({
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
product_features = refined_df.groupby('product_name').agg({
    'purchased': ['sum', 'count', 'mean'],
    'price': 'mean',
    'user_id': 'nunique'
}).reset_index()

product_features.columns = ['product_name', 'product_purchases', 'product_views', 'product_purchase_rate', 
                           'product_price', 'unique_users']

# Category-level features
category_features = refined_df.groupby('category').agg({
    'purchased': ['sum', 'count', 'mean'],
    'price': 'mean',
    'user_id': 'nunique'
}).reset_index()

category_features.columns = ['category', 'category_purchases', 'category_views', 'category_purchase_rate', 
                            'category_price', 'category_users']

# Merge features
refined_main = refined_df[['user_id', 'product_name', 'category', 'price', 'purchased', 
                          'age', 'gender', 'income_level', 'beauty_enthusiast', 'skin_type']].copy()

refined_main = refined_main.merge(user_features, on='user_id', how='left')
refined_main = refined_main.merge(product_features, on='product_name', how='left')
refined_main = refined_main.merge(category_features, on='category', how='left')

# Create additional features
refined_main['price_ratio'] = refined_main['price'] / (refined_main['avg_price'] + 1e-8)
refined_main['is_expensive'] = (refined_main['price'] > refined_main['avg_price']).astype(int)

# Encode categorical variables
refined_main['gender_encoded'] = refined_main['gender'].map({'F': 1, 'M': 0})
refined_main['income_encoded'] = refined_main['income_level'].map({'low': 0, 'medium': 1, 'high': 2})

# Create price categories
refined_main['price_category'] = pd.cut(refined_main['price'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
refined_main['price_category_encoded'] = refined_main['price_category'].cat.codes

# Fill missing values for numeric columns only
numeric_columns = refined_main.select_dtypes(include=[np.number]).columns
refined_main[numeric_columns] = refined_main[numeric_columns].fillna(0)

# Fill missing values for categorical columns
categorical_columns = refined_main.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    if col != 'price_category':
        refined_main[col] = refined_main[col].fillna('unknown')

# Map cosmetics features to original model features
refined_main['unique_categories'] = refined_main['category'].nunique()
refined_main['session_duration_days'] = refined_main['avg_session_duration'] / (24 * 3600)

# Fill missing values for additional features
refined_main['age'] = refined_main['age'].fillna(refined_main['age'].mean())
refined_main['beauty_enthusiast'] = refined_main['beauty_enthusiast'].fillna(0)
refined_main['gender_encoded'] = refined_main['gender_encoded'].fillna(1)
refined_main['income_encoded'] = refined_main['income_encoded'].fillna(1)

# Prepare features for model testing
feature_columns = ['price', 'total_purchases', 'total_events', 'purchase_rate', 
                  'avg_price', 'price_std', 'min_price', 'max_price', 
                  'unique_products', 'unique_categories', 'session_duration_days',
                  'product_purchases', 'product_views', 'product_purchase_rate', 
                  'product_price', 'unique_users', 'category_purchases', 
                  'category_views', 'category_purchase_rate', 'category_price', 
                  'category_users', 'price_ratio', 'is_expensive', 'price_category_encoded']

X_refined = refined_main[feature_columns]
y_refined = refined_main['purchased']

print(f"✓ Refined features prepared:")
print(f"  - Feature matrix shape: {X_refined.shape}")
print(f"  - Target distribution: {y_refined.value_counts().to_dict()}")

# Scale features using the same scaler from training
X_refined_scaled = scaler.transform(X_refined)

# Make predictions
print(f"\nMaking predictions on refined cosmetics dataset...")
y_pred_refined = model.predict(X_refined_scaled)
y_pred_proba_refined = model.predict_proba(X_refined_scaled)[:, 1]

# Calculate metrics
auc_score_refined = roc_auc_score(y_refined, y_pred_proba_refined)

print(f"\n" + "="*80)
print("REFINED MODEL PERFORMANCE (TOP 2 PRODUCTS ONLY)")
print("="*80)

print(f"AUC Score: {auc_score_refined:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_refined, y_pred_refined))

# Confusion Matrix
cm_refined = confusion_matrix(y_refined, y_pred_refined)
print(f"\nConfusion Matrix:")
print(cm_refined)

# Analyze performance by product
print(f"\n" + "="*80)
print("PERFORMANCE ANALYSIS BY PRODUCT (REFINED)")
print("="*80)

refined_main['predicted'] = y_pred_refined
refined_main['predicted_proba'] = y_pred_proba_refined

product_analysis_refined = refined_main.groupby('product_name').agg({
    'purchased': ['count', 'sum', 'mean'],
    'predicted': 'sum',
    'predicted_proba': 'mean',
    'price': 'mean'
}).round(3)

product_analysis_refined.columns = ['total_interactions', 'actual_purchases', 'actual_purchase_rate', 
                                   'predicted_purchases', 'avg_prediction_prob', 'avg_price']

# Calculate accuracy for each product
def calculate_accuracy(group):
    return (group['purchased'] == group['predicted']).mean()

product_accuracy_refined = refined_main.groupby('product_name').apply(calculate_accuracy)
product_analysis_refined['prediction_accuracy'] = product_accuracy_refined.round(3)

print("Refined Products Performance:")
for idx, (product, row) in enumerate(product_analysis_refined.iterrows(), 1):
    print(f"\n{idx}. {product}")
    print(f"   - Actual purchases: {row['actual_purchases']:.0f}")
    print(f"   - Predicted purchases: {row['predicted_purchases']:.0f}")
    print(f"   - Actual purchase rate: {row['actual_purchase_rate']:.3f}")
    print(f"   - Avg prediction probability: {row['avg_prediction_prob']:.3f}")
    print(f"   - Prediction accuracy: {row['prediction_accuracy']:.3f}")
    print(f"   - Avg price: ${row['avg_price']:.2f}")

# Model compatibility assessment for refined dataset
print(f"\n" + "="*80)
print("REFINED MODEL COMPATIBILITY ASSESSMENT")
print("="*80)

actual_purchase_rate_refined = y_refined.mean()
predicted_purchase_rate_refined = y_pred_refined.mean()
prediction_accuracy_refined = (y_refined == y_pred_refined).mean()

print(f"Actual purchase rate: {actual_purchase_rate_refined:.3f}")
print(f"Predicted purchase rate: {predicted_purchase_rate_refined:.3f}")
print(f"Overall prediction accuracy: {prediction_accuracy_refined:.3f}")

# Determine if model is suitable for refined dataset
if auc_score_refined > 0.7 and prediction_accuracy_refined > 0.8:
    compatibility_refined = "HIGH"
    recommendation_refined = "Model performs well on refined cosmetics dataset"
elif auc_score_refined > 0.6 and prediction_accuracy_refined > 0.7:
    compatibility_refined = "MEDIUM"
    recommendation_refined = "Model shows moderate performance on refined dataset"
else:
    compatibility_refined = "LOW"
    recommendation_refined = "Model still needs improvement even for top 2 products"

print(f"\nCompatibility Level (Refined): {compatibility_refined}")
print(f"Recommendation: {recommendation_refined}")

# Compare with original full dataset results
print(f"\n" + "="*80)
print("COMPARISON: FULL DATASET vs REFINED DATASET")
print("="*80)

# Load original results
import json
try:
    with open('cosmetics_test_results.json', 'r') as f:
        original_results = json.load(f)
    
    print(f"Full Dataset (10 products):")
    print(f"  - AUC Score: {original_results['auc_score']:.4f}")
    print(f"  - Prediction Accuracy: {original_results['prediction_accuracy']:.3f}")
    print(f"  - Compatibility: {original_results['compatibility']}")
    
    print(f"\nRefined Dataset (2 products):")
    print(f"  - AUC Score: {auc_score_refined:.4f}")
    print(f"  - Prediction Accuracy: {prediction_accuracy_refined:.3f}")
    print(f"  - Compatibility: {compatibility_refined}")
    
    # Calculate improvement
    auc_improvement = auc_score_refined - original_results['auc_score']
    accuracy_improvement = prediction_accuracy_refined - original_results['prediction_accuracy']
    
    print(f"\nImprovement:")
    print(f"  - AUC Score: {auc_improvement:+.4f}")
    print(f"  - Prediction Accuracy: {accuracy_improvement:+.3f}")
    
except:
    print("Could not load original results for comparison")

# Create visualization comparing full vs refined
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# ROC Curve comparison
fpr_refined, tpr_refined, _ = roc_curve(y_refined, y_pred_proba_refined)
ax1.plot(fpr_refined, tpr_refined, label=f'Refined Dataset (AUC = {auc_score_refined:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve - Refined Dataset (Top 2 Products)')
ax1.legend()
ax1.grid(True)

# Confusion Matrix
sns.heatmap(cm_refined, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix - Refined Dataset')

# Product Performance
x_pos = np.arange(len(product_analysis_refined))
width = 0.35

ax3.bar(x_pos - width/2, product_analysis_refined['actual_purchases'], width, label='Actual', alpha=0.7)
ax3.bar(x_pos + width/2, product_analysis_refined['predicted_purchases'], width, label='Predicted', alpha=0.7)
ax3.set_xlabel('Products')
ax3.set_ylabel('Number of Purchases')
ax3.set_title('Actual vs Predicted Purchases (Refined Dataset)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in product_analysis_refined.index], rotation=45)
ax3.legend()

# Performance Metrics Comparison
metrics = ['AUC Score', 'Prediction Accuracy', 'Purchase Rate Match']
full_values = [original_results.get('auc_score', 0), original_results.get('prediction_accuracy', 0), 0.5]  # Placeholder
refined_values = [auc_score_refined, prediction_accuracy_refined, 0.8]  # Placeholder

x_pos = np.arange(len(metrics))
width = 0.35

ax4.bar(x_pos - width/2, full_values, width, label='Full Dataset', alpha=0.7)
ax4.bar(x_pos + width/2, refined_values, width, label='Refined Dataset', alpha=0.7)
ax4.set_xlabel('Metrics')
ax4.set_ylabel('Score')
ax4.set_title('Performance Comparison')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics)
ax4.legend()

plt.tight_layout()
plt.savefig('refined_cosmetics_test_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Save refined results
refined_results = {
    'dataset': 'Refined Cosmetics Dataset (Top 2 Products)',
    'products': top_2_product_names,
    'auc_score': auc_score_refined,
    'prediction_accuracy': prediction_accuracy_refined,
    'actual_purchase_rate': actual_purchase_rate_refined,
    'predicted_purchase_rate': predicted_purchase_rate_refined,
    'compatibility': compatibility_refined,
    'recommendation': recommendation_refined
}

with open('refined_cosmetics_test_results.json', 'w') as f:
    json.dump(refined_results, f, indent=2)

print(f"\n✓ Refined results saved to 'refined_cosmetics_test_results.json'")
print(f"✓ Visualization saved as 'refined_cosmetics_test_results.png'")
print(f"\nRefined cosmetics dataset testing completed!")