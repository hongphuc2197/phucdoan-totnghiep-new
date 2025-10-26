#!/usr/bin/env python3
"""
Generate Confusion Matrix - Simple Version (No XGBoost)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def main():
    print('='*60)
    print('GENERATING CONFUSION MATRIX - SIMPLE VERSION')
    print('='*60)
    
    # Load data
    print('Loading data...')
    try:
        df = pd.read_csv('processed_data.csv')
        print(f'Dataset loaded: {df.shape}')
        
        # Feature columns
        feature_columns = ['price', 'total_purchases', 'total_events', 'purchase_rate', 
                          'avg_price', 'price_std', 'min_price', 'max_price', 
                          'unique_products', 'unique_categories', 'session_duration_days',
                          'product_purchases', 'product_views', 'product_purchase_rate', 
                          'product_price', 'unique_users', 'category_purchases', 
                          'category_views', 'category_purchase_rate', 'category_price', 
                          'category_users', 'price_ratio', 'is_expensive', 'price_category_encoded']
        
        X = df[feature_columns]
        y = df['purchased']
        
        print(f'Features: {X.shape[1]}')
        print(f'Target distribution: {y.value_counts()}')
        print(f'Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f'Train set: {X_train.shape}, Test set: {X_test.shape}')
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest model (simpler than XGBoost)
        print('Training Random Forest...')
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print('\n' + '='*50)
        print('CONFUSION MATRIX RESULTS:')
        print('='*50)
        print(f'Confusion Matrix:')
        print(cm)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        print(f'\nDetailed Results:')
        print(f'True Negatives (TN):  {tn:,}')
        print(f'False Positives (FP): {fp:,}')
        print(f'False Negatives (FN): {fn:,}')
        print(f'True Positives (TP):  {tp:,}')
        
        # Metrics
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f'\nPerformance Metrics:')
        print(f'Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)')
        print(f'Precision: {precision:.4f} ({precision*100:.2f}%)')
        print(f'Recall:    {recall:.4f} ({recall*100:.2f}%)')
        print(f'F1-Score:  {f1:.4f} ({f1*100:.2f}%)')
        print(f'Specificity: {specificity:.4f} ({specificity*100:.2f}%)')
        print(f'AUC:       {auc:.4f} ({auc*100:.2f}%)')
        
        # Create confusion matrix visualization
        plt.figure(figsize=(12, 8))
        
        # Confusion Matrix Heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Purchase', 'Purchase'],
                   yticklabels=['No Purchase', 'Purchase'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Metrics Bar Chart
        plt.subplot(2, 2, 2)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC']
        values = [accuracy, precision, recall, f1, specificity, auc]
        bars = plt.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#95E1D3', '#FFD93D'])
        plt.title('Performance Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Confusion Matrix Values Table
        plt.subplot(2, 2, 3)
        plt.axis('off')
        table_data = [
            ['', 'Predicted No Purchase', 'Predicted Purchase'],
            ['Actual No Purchase', f'{tn:,}', f'{fp:,}'],
            ['Actual Purchase', f'{fn:,}', f'{tp:,}']
        ]
        
        table = plt.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        plt.title('Confusion Matrix Values', pad=20)
        
        # Performance Summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary_text = f"""
PERFORMANCE SUMMARY:

Dataset: {df.shape[0]:,} records
Test Set: {len(y_test):,} samples
Class Imbalance: {y.value_counts()[0] / y.value_counts()[1]:.2f}:1

Key Metrics:
• AUC Score: {auc:.4f} ({auc*100:.2f}%)
• Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
• Precision: {precision:.4f} ({precision*100:.2f}%)
• Recall: {recall:.4f} ({recall*100:.2f}%)
• F1-Score: {f1:.4f} ({f1*100:.2f}%)

Model Performance:
• Detects {recall*100:.1f}% of actual buyers
• {precision*100:.1f}% of predictions are correct
• {specificity*100:.1f}% correctly identify non-buyers
        """
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f'\nConfusion matrix visualization saved as: confusion_matrix_results.png')
        
        # Save results to file
        results = {
            'confusion_matrix': cm.tolist(),
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'auc': auc
            },
            'detailed_counts': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
        
        import json
        with open('confusion_matrix_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'Results saved to: confusion_matrix_results.json')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

