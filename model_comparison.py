import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load processed data
print("Loading processed data...")
df = pd.read_csv('processed_data.csv')

# Prepare features and target
feature_columns = ['price', 'total_purchases', 'total_events', 'purchase_rate', 
                  'avg_price', 'price_std', 'min_price', 'max_price', 
                  'unique_products', 'unique_categories', 'session_duration_days',
                  'product_purchases', 'product_views', 'product_purchase_rate', 
                  'product_price', 'unique_users', 'category_purchases', 
                  'category_views', 'category_purchase_rate', 'category_price', 
                  'category_users', 'price_ratio', 'is_expensive', 'price_category_encoded']

X = df[feature_columns]
y = df['purchased']

print(f"Dataset shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")
print(f"Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
print("Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE - Training set shape: {X_train_balanced.shape}")
print(f"After SMOTE - Target distribution: {np.bincount(y_train_balanced)}")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
}

# Hyperparameter tuning for each model
print("\nPerforming hyperparameter tuning...")

# Logistic Regression parameters
lr_params = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Random Forest parameters
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# XGBoost parameters
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# LightGBM parameters
lgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 50, 100]
}

param_grids = {
    'Logistic Regression': lr_params,
    'Random Forest': rf_params,
    'XGBoost': xgb_params,
    'LightGBM': lgb_params
}

# Train and evaluate models
results = {}
best_models = {}

print("\nTraining and evaluating models...")

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        model, 
        param_grids[name], 
        cv=3, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_balanced, y_train_balanced)
    best_models[name] = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_models[name].predict(X_test_scaled)
    y_pred_proba = best_models[name].predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation score
    cv_scores = cross_val_score(best_models[name], X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')
    
    results[name] = {
        'best_params': grid_search.best_params_,
        'auc_score': auc_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"{name} - Best params: {grid_search.best_params_}")
    print(f"{name} - AUC: {auc_score:.4f}")
    print(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Create comparison table
print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'AUC Score': [results[name]['auc_score'] for name in results.keys()],
    'CV Mean': [results[name]['cv_mean'] for name in results.keys()],
    'CV Std': [results[name]['cv_std'] for name in results.keys()]
}).sort_values('AUC Score', ascending=False)

print(comparison_df.to_string(index=False))

# Detailed classification reports
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS")
print("="*80)

for name in results.keys():
    print(f"\n{name}:")
    print(classification_report(y_test, results[name]['y_pred']))

# Plot ROC curves
plt.figure(figsize=(12, 8))

for name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["auc_score"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot Precision-Recall curves
plt.figure(figsize=(12, 8))

for name in results.keys():
    precision, recall, _ = precision_recall_curve(y_test, results[name]['y_pred_proba'])
    plt.plot(recall, precision, label=f'{name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance for tree-based models
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

tree_models = ['Random Forest', 'XGBoost', 'LightGBM']
for i, name in enumerate(tree_models):
    if name in best_models:
        if hasattr(best_models[name], 'feature_importances_'):
            importances = best_models[name].feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            axes[i].bar(range(10), importances[indices])
            axes[i].set_title(f'{name} - Top 10 Feature Importances')
            axes[i].set_xticks(range(10))
            axes[i].set_xticklabels([feature_columns[j] for j in indices], rotation=45, ha='right')

# Remove empty subplot
axes[3].remove()

plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
plt.show()

# Save best model
best_model_name = comparison_df.iloc[0]['Model']
print(f"\nBest performing model: {best_model_name}")
print(f"Best AUC Score: {comparison_df.iloc[0]['AUC Score']:.4f}")

# Save the best model
import joblib
model_filename = f'best_model_{best_model_name.lower().replace(" ", "_")}.pkl'
joblib.dump(best_models[best_model_name], model_filename)
joblib.dump(scaler, 'scaler.pkl')

print(f"\nBest model saved as '{model_filename}'")
print("Scaler saved as 'scaler.pkl'")

print("\nModel comparison completed!")