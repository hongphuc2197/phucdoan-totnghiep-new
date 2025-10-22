import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data_and_model():
    """Load dữ liệu và model đã train"""
    print("🔄 Loading data and model...")
    
    # Load data
    df = pd.read_csv('processed_data.csv')
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Load model and scaler
    model = joblib.load('best_model_xgboost.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model and scaler loaded")
    
    # Prepare features
    feature_columns = ['price', 'total_purchases', 'total_events', 'purchase_rate',
                      'avg_price', 'price_std', 'min_price', 'max_price',
                      'unique_products', 'unique_categories', 'session_duration_days',
                      'product_purchases', 'product_views', 'product_purchase_rate',
                      'product_price', 'unique_users', 'category_purchases',
                      'category_views', 'category_purchase_rate', 'category_price',
                      'category_users', 'price_ratio', 'is_expensive', 'price_category_encoded']
    
    X = df[feature_columns]
    y = df['purchased']
    
    return X, y, model, scaler, feature_columns

def evaluate_model(X, y, model, scaler, test_size=0.2):
    """Evaluate model performance"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Train model
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def feature_ablation_study(X, y, model, scaler, feature_columns):
    """Ablation study by removing individual features"""
    print("\n🔍 FEATURE ABLATION STUDY")
    print("=" * 50)
    
    # Baseline performance (all features)
    print("📊 Calculating baseline performance...")
    baseline_results = evaluate_model(X, y, model, scaler)
    print(f"Baseline AUC: {baseline_results['auc']:.4f}")
    print(f"Baseline Accuracy: {baseline_results['accuracy']:.4f}")
    
    # Test removing each feature
    ablation_results = []
    
    for i, feature in enumerate(feature_columns):
        print(f"\n🔄 Testing without feature {i+1}/{len(feature_columns)}: {feature}")
        
        # Remove feature
        X_ablated = X.drop(columns=[feature])
        
        # Evaluate
        results = evaluate_model(X_ablated, y, model, scaler)
        
        # Calculate impact
        auc_impact = baseline_results['auc'] - results['auc']
        accuracy_impact = baseline_results['accuracy'] - results['accuracy']
        
        ablation_results.append({
            'feature': feature,
            'auc_without': results['auc'],
            'accuracy_without': results['accuracy'],
            'auc_impact': auc_impact,
            'accuracy_impact': accuracy_impact,
            'precision_without': results['precision'],
            'recall_without': results['recall'],
            'f1_without': results['f1']
        })
        
        print(f"   AUC without {feature}: {results['auc']:.4f} (impact: {auc_impact:+.4f})")
    
    return pd.DataFrame(ablation_results), baseline_results

def component_ablation_study(X, y, feature_columns):
    """Ablation study by removing components (SMOTE, scaling, etc.)"""
    print("\n🔍 COMPONENT ABLATION STUDY")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    component_results = []
    
    # 1. Full pipeline (baseline)
    print("📊 Testing full pipeline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X_train_balanced, y_train_balanced)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc_full = roc_auc_score(y_test, y_pred_proba)
    accuracy_full = accuracy_score(y_test, model.predict(X_test_scaled))
    
    component_results.append({
        'component': 'Full Pipeline',
        'auc': auc_full,
        'accuracy': accuracy_full,
        'description': 'All components included'
    })
    
    # 2. Without SMOTE
    print("📊 Testing without SMOTE...")
    model_no_smote = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100, max_depth=6, learning_rate=0.1)
    model_no_smote.fit(X_train_scaled, y_train)
    
    y_pred_proba = model_no_smote.predict_proba(X_test_scaled)[:, 1]
    auc_no_smote = roc_auc_score(y_test, y_pred_proba)
    accuracy_no_smote = accuracy_score(y_test, model_no_smote.predict(X_test_scaled))
    
    component_results.append({
        'component': 'Without SMOTE',
        'auc': auc_no_smote,
        'accuracy': accuracy_no_smote,
        'description': 'No class imbalance handling'
    })
    
    # 3. Without Scaling
    print("📊 Testing without scaling...")
    model_no_scale = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100, max_depth=6, learning_rate=0.1)
    model_no_scale.fit(X_train, y_train)
    
    y_pred_proba = model_no_scale.predict_proba(X_test)[:, 1]
    auc_no_scale = roc_auc_score(y_test, y_pred_proba)
    accuracy_no_scale = accuracy_score(y_test, model_no_scale.predict(X_test))
    
    component_results.append({
        'component': 'Without Scaling',
        'auc': auc_no_scale,
        'accuracy': accuracy_no_scale,
        'description': 'No feature scaling'
    })
    
    # 4. Without both SMOTE and Scaling
    print("📊 Testing without SMOTE and scaling...")
    model_minimal = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100, max_depth=6, learning_rate=0.1)
    model_minimal.fit(X_train, y_train)
    
    y_pred_proba = model_minimal.predict_proba(X_test)[:, 1]
    auc_minimal = roc_auc_score(y_test, y_pred_proba)
    accuracy_minimal = accuracy_score(y_test, model_minimal.predict(X_test))
    
    component_results.append({
        'component': 'Minimal Pipeline',
        'auc': auc_minimal,
        'accuracy': accuracy_minimal,
        'description': 'No SMOTE, no scaling'
    })
    
    return pd.DataFrame(component_results)

def create_ablation_visualizations(feature_results, component_results, baseline_results):
    """Create visualization for ablation study results"""
    print("\n📊 Creating ablation study visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
    
    # 1. Feature Impact on AUC
    ax1 = axes[0, 0]
    feature_impact = feature_results.nlargest(10, 'auc_impact')
    bars1 = ax1.barh(range(len(feature_impact)), feature_impact['auc_impact'], color='skyblue')
    ax1.set_yticks(range(len(feature_impact)))
    ax1.set_yticklabels(feature_impact['feature'], fontsize=8)
    ax1.set_xlabel('AUC Impact (Baseline - Without Feature)')
    ax1.set_title('Top 10 Features by AUC Impact')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                ha='left', va='center', fontsize=8)
    
    # 2. Feature Impact on Accuracy
    ax2 = axes[0, 1]
    feature_impact_acc = feature_results.nlargest(10, 'accuracy_impact')
    bars2 = ax2.barh(range(len(feature_impact_acc)), feature_impact_acc['accuracy_impact'], color='lightcoral')
    ax2.set_yticks(range(len(feature_impact_acc)))
    ax2.set_yticklabels(feature_impact_acc['feature'], fontsize=8)
    ax2.set_xlabel('Accuracy Impact (Baseline - Without Feature)')
    ax2.set_title('Top 10 Features by Accuracy Impact')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                ha='left', va='center', fontsize=8)
    
    # 3. Component Comparison
    ax3 = axes[1, 0]
    x_pos = np.arange(len(component_results))
    width = 0.35
    
    bars3_auc = ax3.bar(x_pos - width/2, component_results['auc'], width, label='AUC', color='lightgreen')
    ax3_twin = ax3.twinx()
    bars3_acc = ax3_twin.bar(x_pos + width/2, component_results['accuracy'], width, label='Accuracy', color='orange')
    
    ax3.set_xlabel('Components')
    ax3.set_ylabel('AUC Score', color='green')
    ax3_twin.set_ylabel('Accuracy Score', color='orange')
    ax3.set_title('Component Ablation: AUC vs Accuracy')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(component_results['component'], rotation=45, ha='right')
    
    # Add value labels
    for i, (bar_auc, bar_acc) in enumerate(zip(bars3_auc, bars3_acc)):
        ax3.text(bar_auc.get_x() + bar_auc.get_width()/2, bar_auc.get_height() + 0.001, 
                f'{bar_auc.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        ax3_twin.text(bar_acc.get_x() + bar_acc.get_width()/2, bar_acc.get_height() + 0.001, 
                     f'{bar_acc.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Performance Degradation
    ax4 = axes[1, 1]
    baseline_auc = baseline_results['auc']
    baseline_acc = baseline_results['accuracy']
    
    auc_degradation = [(baseline_auc - row['auc']) / baseline_auc * 100 for _, row in component_results.iterrows()]
    acc_degradation = [(baseline_acc - row['accuracy']) / baseline_acc * 100 for _, row in component_results.iterrows()]
    
    x_pos = np.arange(len(component_results))
    bars4_auc = ax4.bar(x_pos - width/2, auc_degradation, width, label='AUC Degradation %', color='red', alpha=0.7)
    bars4_acc = ax4.bar(x_pos + width/2, acc_degradation, width, label='Accuracy Degradation %', color='darkred', alpha=0.7)
    
    ax4.set_xlabel('Components')
    ax4.set_ylabel('Performance Degradation (%)')
    ax4.set_title('Performance Degradation by Component Removal')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(component_results['component'], rotation=45, ha='right')
    ax4.legend()
    
    # Add value labels
    for i, (bar_auc, bar_acc) in enumerate(zip(bars4_auc, bars4_acc)):
        ax4.text(bar_auc.get_x() + bar_auc.get_width()/2, bar_auc.get_height() + 0.1, 
                f'{bar_auc.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
        ax4.text(bar_acc.get_x() + bar_acc.get_width()/2, bar_acc.get_height() + 0.1, 
                f'{bar_acc.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: ablation_study_results.png")

def generate_ablation_report(feature_results, component_results, baseline_results):
    """Generate detailed ablation study report"""
    print("\n📝 GENERATING ABLATION STUDY REPORT")
    print("=" * 60)
    
    # Feature analysis
    print("\n🏆 TOP 10 MOST CRITICAL FEATURES (by AUC impact):")
    print("-" * 60)
    top_features = feature_results.nlargest(10, 'auc_impact')
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25} AUC impact: {row['auc_impact']:+.4f}")
    
    print("\n⚠️ TOP 10 LEAST CRITICAL FEATURES (by AUC impact):")
    print("-" * 60)
    bottom_features = feature_results.nsmallest(10, 'auc_impact')
    for i, (_, row) in enumerate(bottom_features.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25} AUC impact: {row['auc_impact']:+.4f}")
    
    # Component analysis
    print("\n📊 COMPONENT IMPACT ANALYSIS:")
    print("-" * 60)
    baseline_auc = baseline_results['auc']
    baseline_acc = baseline_results['accuracy']
    
    for _, row in component_results.iterrows():
        auc_degradation = (baseline_auc - row['auc']) / baseline_auc * 100
        acc_degradation = (baseline_acc - row['accuracy']) / baseline_acc * 100
        print(f"• {row['component']:<20} AUC: {row['auc']:.4f} ({auc_degradation:+.1f}%) | Acc: {row['accuracy']:.4f} ({acc_degradation:+.1f}%)")
    
    # Key insights
    print("\n💡 KEY INSIGHTS:")
    print("-" * 60)
    
    # Most critical feature
    most_critical = feature_results.loc[feature_results['auc_impact'].idxmax()]
    print(f"🎯 Most Critical Feature: {most_critical['feature']} (AUC impact: {most_critical['auc_impact']:+.4f})")
    
    # SMOTE impact
    smote_impact = component_results[component_results['component'] == 'Without SMOTE']
    if not smote_impact.empty:
        smote_auc_degradation = (baseline_auc - smote_impact.iloc[0]['auc']) / baseline_auc * 100
        print(f"⚖️ SMOTE Impact: {smote_auc_degradation:.1f}% AUC degradation when removed")
    
    # Scaling impact
    scale_impact = component_results[component_results['component'] == 'Without Scaling']
    if not scale_impact.empty:
        scale_auc_degradation = (baseline_auc - scale_impact.iloc[0]['auc']) / baseline_auc * 100
        print(f"📏 Scaling Impact: {scale_auc_degradation:.1f}% AUC degradation when removed")
    
    # Save results
    feature_results.to_csv('feature_ablation_results.csv', index=False)
    component_results.to_csv('component_ablation_results.csv', index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   - feature_ablation_results.csv")
    print(f"   - component_ablation_results.csv")
    print(f"   - ablation_study_results.png")

def main():
    print("🔬 ABLATION STUDY FOR RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Load data and model
    X, y, model, scaler, feature_columns = load_data_and_model()
    
    # Sample data for faster computation
    sample_size = min(100000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    print(f"📊 Using sample size: {X_sample.shape[0]} for faster computation")
    
    # Feature ablation study
    feature_results, baseline_results = feature_ablation_study(X_sample, y_sample, model, scaler, feature_columns)
    
    # Component ablation study
    component_results = component_ablation_study(X_sample, y_sample, feature_columns)
    
    # Create visualizations
    create_ablation_visualizations(feature_results, component_results, baseline_results)
    
    # Generate report
    generate_ablation_report(feature_results, component_results, baseline_results)
    
    print("\n🎉 ABLATION STUDY COMPLETED!")
    print("=" * 60)
    print("📊 This study provides:")
    print("   • Feature importance ranking")
    print("   • Component impact analysis")
    print("   • Performance degradation metrics")
    print("   • Visualizations for reporting")
    print("\n💡 Use these results to strengthen your thesis!")

if __name__ == "__main__":
    main()