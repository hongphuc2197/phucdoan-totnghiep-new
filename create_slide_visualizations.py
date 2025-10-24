"""
Script to create missing visualizations for presentation slides
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory if needed
import os
os.makedirs('slide_images', exist_ok=True)

print("Creating slide visualizations...")

# ============================================================================
# SLIDE 3: Class Distribution Pie Chart
# ============================================================================
print("1. Creating class distribution chart...")

fig, ax = plt.subplots(figsize=(10, 8))

# Data
sizes = [5.96, 94.04]
labels = ['Purchase\n(5.96%)', 'Non-Purchase\n(94.04%)']
colors = ['#FF6B6B', '#4ECDC4']
explode = (0.1, 0)

# Create pie chart
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                    colors=colors, autopct='%1.1f%%',
                                    shadow=True, startangle=90,
                                    textprops={'fontsize': 14, 'weight': 'bold'})

# Add title
ax.set_title('Class Distribution - E-commerce Dataset\n4.1M Records', 
             fontsize=18, weight='bold', pad=20)

# Add imbalance ratio annotation
ax.text(0, -1.4, 'Imbalance Ratio: 15.78:1', 
        ha='center', fontsize=16, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('slide_images/slide03_class_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide03_class_distribution.png")
plt.close()

# ============================================================================
# SLIDE 4: Input-Output System Diagram
# ============================================================================
print("2. Creating input-output diagram...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Input box
input_box = FancyBboxPatch((0.5, 2), 2, 2, boxstyle="round,pad=0.1", 
                           edgecolor='#2E86AB', facecolor='#A9E5FF', linewidth=3)
ax.add_patch(input_box)
ax.text(1.5, 3.5, 'INPUT', ha='center', va='center', fontsize=14, weight='bold')
ax.text(1.5, 3.0, '• User behavior', ha='center', va='center', fontsize=10)
ax.text(1.5, 2.7, '• Product info', ha='center', va='center', fontsize=10)
ax.text(1.5, 2.4, '• Temporal data', ha='center', va='center', fontsize=10)

# System box
system_box = FancyBboxPatch((4, 1.5), 2, 3, boxstyle="round,pad=0.1",
                            edgecolor='#F77F00', facecolor='#FFD19A', linewidth=3)
ax.add_patch(system_box)
ax.text(5, 4.0, 'SYSTEM', ha='center', va='center', fontsize=14, weight='bold')
ax.text(5, 3.5, 'XGBoost', ha='center', va='center', fontsize=11, weight='bold')
ax.text(5, 3.2, '+', ha='center', va='center', fontsize=11)
ax.text(5, 2.9, 'SMOTE', ha='center', va='center', fontsize=11, weight='bold')
ax.text(5, 2.5, '24 Features', ha='center', va='center', fontsize=10, style='italic')

# Output box
output_box = FancyBboxPatch((7.5, 2), 2, 2, boxstyle="round,pad=0.1",
                            edgecolor='#06A77D', facecolor='#B5E8D5', linewidth=3)
ax.add_patch(output_box)
ax.text(8.5, 3.5, 'OUTPUT', ha='center', va='center', fontsize=14, weight='bold')
ax.text(8.5, 3.0, 'Purchase: 0/1', ha='center', va='center', fontsize=10)
ax.text(8.5, 2.7, 'Probability', ha='center', va='center', fontsize=10)
ax.text(8.5, 2.4, 'AUC: 89.84%', ha='center', va='center', fontsize=10, weight='bold')

# Arrows
arrow1 = FancyArrowPatch((2.5, 3), (4, 3), arrowstyle='->', 
                         mutation_scale=30, linewidth=3, color='#2E86AB')
arrow2 = FancyArrowPatch((6, 3), (7.5, 3), arrowstyle='->', 
                         mutation_scale=30, linewidth=3, color='#06A77D')
ax.add_patch(arrow1)
ax.add_patch(arrow2)

ax.set_title('Recommendation System Architecture', fontsize=18, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('slide_images/slide04_system_diagram.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide04_system_diagram.png")
plt.close()

# ============================================================================
# SLIDE 5: Recommendation System Types
# ============================================================================
print("3. Creating recommendation system types diagram...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Collaborative Filtering
collab_box = FancyBboxPatch((0.5, 6), 3.5, 2.5, boxstyle="round,pad=0.15",
                            edgecolor='#E63946', facecolor='#FFB4BA', linewidth=2)
ax.add_patch(collab_box)
ax.text(2.25, 8.0, 'Collaborative Filtering', ha='center', fontsize=12, weight='bold')
ax.text(2.25, 7.5, '• User similarity', ha='left', fontsize=9)
ax.text(2.25, 7.2, '• Item similarity', ha='left', fontsize=9)
ax.text(2.25, 6.9, '❌ Cold start', ha='left', fontsize=9)

# Content-based Filtering
content_box = FancyBboxPatch((6, 6), 3.5, 2.5, boxstyle="round,pad=0.15",
                             edgecolor='#457B9D', facecolor='#A8DADC', linewidth=2)
ax.add_patch(content_box)
ax.text(7.75, 8.0, 'Content-based Filtering', ha='center', fontsize=12, weight='bold')
ax.text(7.75, 7.5, '• Product features', ha='left', fontsize=9)
ax.text(7.75, 7.2, '• User preferences', ha='left', fontsize=9)
ax.text(7.75, 6.9, '❌ Over-specialization', ha='left', fontsize=9)

# Hybrid System
hybrid_box = FancyBboxPatch((2.5, 2), 5, 3, boxstyle="round,pad=0.2",
                            edgecolor='#06A77D', facecolor='#90EE90', linewidth=3)
ax.add_patch(hybrid_box)
ax.text(5, 4.5, '⭐ HYBRID SYSTEM ⭐', ha='center', fontsize=14, weight='bold')
ax.text(5, 4.0, 'Collaborative + Content-based', ha='center', fontsize=11)
ax.text(5, 3.5, '✅ Best of both worlds', ha='center', fontsize=10, weight='bold')
ax.text(5, 3.1, '✅ Overcomes limitations', ha='center', fontsize=10)
ax.text(5, 2.7, '✅ Better performance', ha='center', fontsize=10)
ax.text(5, 2.3, '→ ĐỒ ÁN NÀY', ha='center', fontsize=11, weight='bold', 
        style='italic', color='red')

# Arrows pointing to hybrid
arrow1 = FancyArrowPatch((2.25, 6), (4, 5), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='#E63946', linestyle='--')
arrow2 = FancyArrowPatch((7.75, 6), (6, 5), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='#457B9D', linestyle='--')
ax.add_patch(arrow1)
ax.add_patch(arrow2)

ax.set_title('Recommendation System Types', fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('slide_images/slide05_recommendation_types.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide05_recommendation_types.png")
plt.close()

# ============================================================================
# SLIDE 6: XGBoost + SMOTE Illustration
# ============================================================================
print("4. Creating XGBoost + SMOTE illustration...")

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# XGBoost illustration (top)
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 3)
ax1.axis('off')
ax1.set_title('XGBoost: Gradient Boosting Ensemble', fontsize=14, weight='bold')

# Draw trees
tree_positions = [1, 3, 5, 7, 9]
for i, x in enumerate(tree_positions):
    color = plt.cm.Blues(0.3 + i * 0.15)
    # Tree trunk
    rect = Rectangle((x-0.3, 0.5), 0.6, 0.8, facecolor=color, edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    # Tree top (triangle)
    triangle = plt.Polygon([[x, 2.5], [x-0.5, 1.3], [x+0.5, 1.3]], 
                           facecolor=color, edgecolor='black', linewidth=2)
    ax1.add_patch(triangle)
    ax1.text(x, 0.2, f'Tree {i+1}', ha='center', fontsize=10, weight='bold')
    
    # Add plus sign
    if i < len(tree_positions) - 1:
        ax1.text(x+1, 1.5, '+', ha='center', va='center', fontsize=20, weight='bold')

# Final prediction arrow
arrow = FancyArrowPatch((9.5, 1.5), (9.5, 0.3), arrowstyle='->', 
                        mutation_scale=30, linewidth=3, color='green')
ax1.add_patch(arrow)
ax1.text(9.5, 0.05, 'Final\nPrediction', ha='center', fontsize=10, weight='bold')

# SMOTE illustration (bottom left)
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Before SMOTE: Imbalanced (15.78:1)', fontsize=12, weight='bold')

# Minority class (few points)
np.random.seed(42)
minority_x = np.random.randn(20) * 0.5 + 2
minority_y = np.random.randn(20) * 0.5 + 2
ax2.scatter(minority_x, minority_y, c='red', s=100, label='Purchase (5.96%)', 
            alpha=0.7, edgecolors='black')

# Majority class (many points)
majority_x = np.random.randn(300) * 1.5 + 1
majority_y = np.random.randn(300) * 1.5 + 1
ax2.scatter(majority_x, majority_y, c='blue', s=50, label='Non-Purchase (94.04%)', 
            alpha=0.3, edgecolors='none')

ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Feature 1', fontsize=10)
ax2.set_ylabel('Feature 2', fontsize=10)

# After SMOTE (bottom right)
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('After SMOTE: Balanced (1:1)', fontsize=12, weight='bold')

# Minority class (original + synthetic)
minority_synth_x = np.random.randn(300) * 0.8 + 2
minority_synth_y = np.random.randn(300) * 0.8 + 2
ax3.scatter(minority_synth_x, minority_synth_y, c='red', s=50, 
            label='Purchase (synthetic)', alpha=0.5, edgecolors='none')
ax3.scatter(minority_x, minority_y, c='darkred', s=100, 
            label='Purchase (original)', alpha=0.8, edgecolors='black', linewidth=1)

# Majority class (same)
ax3.scatter(majority_x, majority_y, c='blue', s=50, label='Non-Purchase', 
            alpha=0.3, edgecolors='none')

ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('Feature 1', fontsize=10)
ax3.set_ylabel('Feature 2', fontsize=10)

plt.tight_layout()
plt.savefig('slide_images/slide06_xgboost_smote.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide06_xgboost_smote.png")
plt.close()

# ============================================================================
# SLIDE 7: Dataset Overview
# ============================================================================
print("5. Creating dataset overview charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E-commerce Dataset Overview - 4.1M Records', fontsize=16, weight='bold')

# Event type distribution
ax1 = axes[0, 0]
events = ['view', 'cart', 'purchase']
counts = [2756147, 1101579, 244557]
colors_events = ['#3498db', '#f39c12', '#e74c3c']
bars1 = ax1.bar(events, counts, color=colors_events, edgecolor='black', linewidth=2)
ax1.set_ylabel('Count', fontsize=12, weight='bold')
ax1.set_title('Event Type Distribution', fontsize=13, weight='bold')
ax1.set_ylim(0, 3000000)
# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=10, weight='bold')
ax1.grid(axis='y', alpha=0.3)

# Class distribution (bar)
ax2 = axes[0, 1]
classes = ['Purchase', 'Non-Purchase']
class_counts = [244557, 3857726]
colors_class = ['#e74c3c', '#3498db']
bars2 = ax2.bar(classes, class_counts, color=colors_class, edgecolor='black', linewidth=2)
ax2.set_ylabel('Count', fontsize=12, weight='bold')
ax2.set_title('Class Distribution (Imbalance: 15.78:1)', fontsize=13, weight='bold')
# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=10, weight='bold')
ax2.grid(axis='y', alpha=0.3)

# Dataset statistics (text)
ax3 = axes[1, 0]
ax3.axis('off')
stats_text = """
DATASET STATISTICS

📊 Total Records: 4,102,283
👥 Unique Users: ~500,000
🛍️  Unique Products: ~200,000
🏷️  Unique Brands: ~5,000
📁 Categories: ~300

📅 Period: October 2019
✅ Authenticity: 100% REAL DATA

🔗 Source: Kaggle
   E-commerce Behavior Data
"""
ax3.text(0.5, 0.5, stats_text, ha='center', va='center', 
         fontsize=12, family='monospace',
         bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))

# Class percentage pie
ax4 = axes[1, 1]
sizes_percent = [5.96, 94.04]
labels_percent = ['Purchase\n5.96%', 'Non-Purchase\n94.04%']
colors_pie = ['#e74c3c', '#3498db']
explode_pie = (0.1, 0)
wedges, texts, autotexts = ax4.pie(sizes_percent, explode=explode_pie, 
                                     labels=labels_percent, colors=colors_pie,
                                     autopct='%1.1f%%', shadow=True, startangle=90,
                                     textprops={'fontsize': 11, 'weight': 'bold'})
ax4.set_title('Class Percentage', fontsize=13, weight='bold')

plt.tight_layout()
plt.savefig('slide_images/slide07_dataset_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide07_dataset_overview.png")
plt.close()

# ============================================================================
# SLIDE 8: Feature Engineering
# ============================================================================
print("6. Creating feature engineering infographic...")

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11, 'Feature Engineering: 24 Features', ha='center', 
        fontsize=18, weight='bold')

# Feature groups
feature_groups = [
    ('Temporal (4)', ['hour', 'day_of_week', 'is_weekend', 'time_period'], '#FF6B6B', 9.5),
    ('User Behavior (3)', ['session_length', 'products_viewed', 'activity_intensity'], '#4ECDC4', 8),
    ('Product Info (4)', ['price', 'price_range', 'category', 'brand'], '#45B7D1', 6.5),
    ('Product Metrics (3)', ['popularity', 'view_count', 'cart_rate'], '#FFA07A', 5),
    ('Interaction (3)', ['user_brand_affinity', 'category_interest', 'repeat_view'], '#98D8C8', 3.5),
    ('Session Context (2)', ['session_position', 'time_since_last_event'], '#F7B731', 2),
    ('Encoded Features (5)', ['categorical encodings', '(Label Encoding)'], '#6C5CE7', 0.5),
]

for i, (title, features, color, y_pos) in enumerate(feature_groups):
    # Draw box
    box = FancyBboxPatch((0.5, y_pos-0.3), 9, len(features)*0.25 + 0.4,
                         boxstyle="round,pad=0.1", 
                         edgecolor=color, facecolor=color, 
                         alpha=0.3, linewidth=2)
    ax.add_patch(box)
    
    # Title
    ax.text(1, y_pos + len(features)*0.25, title, 
            fontsize=12, weight='bold', color=color)
    
    # Features
    for j, feat in enumerate(features):
        ax.text(1.5, y_pos + (len(features)-j-1)*0.25, f'• {feat}', 
                fontsize=10)

plt.tight_layout()
plt.savefig('slide_images/slide08_feature_engineering.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide08_feature_engineering.png")
plt.close()

# ============================================================================
# SLIDE 9: Methodology Flowchart
# ============================================================================
print("7. Creating methodology flowchart...")

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Title
ax.text(5, 13, 'Research Methodology Pipeline', ha='center', 
        fontsize=18, weight='bold')

# Steps
steps = [
    ('1. Data Preprocessing', '• Clean missing values\n• Remove outliers\n• Handle data types', '#FFB6C1', 11.5),
    ('2. Feature Engineering', '• Create 24 features\n• Encode categorical\n• Scale numerical', '#ADD8E6', 9.5),
    ('3. Train-Test Split', '• 80% train, 20% test\n• Stratified split', '#90EE90', 7.8),
    ('4. Apply SMOTE', '• Balance 15.78:1 → 1:1\n• Training set only', '#FFD700', 6.3),
    ('5. Model Training', '• XGBoost\n• Hyperparameter tuning\n• 5-fold CV', '#FFA07A', 4.5),
    ('6. Evaluation', '• AUC-ROC\n• Accuracy, Precision, Recall', '#DDA0DD', 2.5),
]

prev_y = 13
prev_spacing = 1.5
for title, content, color, y_pos in steps:
    # Draw box
    box = FancyBboxPatch((1.5, y_pos-0.5), 7, 1.3,
                         boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    
    # Title
    ax.text(2, y_pos + 0.6, title, fontsize=12, weight='bold')
    
    # Content
    for i, line in enumerate(content.split('\n')):
        ax.text(2.2, y_pos + 0.3 - i*0.3, line, fontsize=9)
    
    # Arrow to next step
    if prev_y != 13:  # Not first iteration
        arrow = FancyArrowPatch((5, prev_y-0.5), (5, y_pos+0.8),
                                arrowstyle='->', mutation_scale=30,
                                linewidth=3, color='#2E86AB')
        ax.add_patch(arrow)
    
    prev_y = y_pos

# Final result
result_box = FancyBboxPatch((2, 0.5), 6, 1,
                            boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='lightgreen',
                            linewidth=3)
ax.add_patch(result_box)
ax.text(5, 1.2, '✅ AUC: 89.84%', ha='center', fontsize=14, weight='bold')
ax.text(5, 0.8, 'Best Model: XGBoost + SMOTE', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('slide_images/slide09_methodology_flowchart.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide09_methodology_flowchart.png")
plt.close()

# ============================================================================
# SLIDE 12: Cross-Validation Box Plot
# ============================================================================
print("8. Creating cross-validation chart...")

fig, ax = plt.subplots(figsize=(10, 8))

# CV scores
cv_scores = [0.8967, 0.8991, 0.8978, 0.8995, 0.8989]
folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']

# Bar plot
bars = ax.bar(folds, cv_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#95E1D3'],
              edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (fold, score) in enumerate(zip(folds, cv_scores)):
    ax.text(i, score + 0.0005, f'{score:.4f}', 
            ha='center', va='bottom', fontsize=11, weight='bold')

# Mean line
mean_score = np.mean(cv_scores)
ax.axhline(y=mean_score, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_score:.4f}')

# Std annotation
std_score = np.std(cv_scores)
ax.text(2, mean_score + 0.001, f'Std Dev: ±{std_score:.4f}', 
        ha='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax.set_ylabel('AUC Score', fontsize=14, weight='bold')
ax.set_xlabel('Cross-Validation Folds', fontsize=14, weight='bold')
ax.set_title('5-Fold Stratified Cross-Validation Results', fontsize=16, weight='bold')
ax.set_ylim(0.895, 0.901)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('slide_images/slide12_cross_validation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide12_cross_validation.png")
plt.close()

# ============================================================================
# SLIDE 19: Future Work Roadmap
# ============================================================================
print("9. Creating future work roadmap...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7.5, 9, 'Future Development Roadmap', ha='center', 
        fontsize=18, weight='bold')

# Timeline
timeline_y = 7
ax.plot([1, 14], [timeline_y, timeline_y], 'k-', linewidth=3)

# Milestones
milestones = [
    ('Current\nState', 2, '✅ XGBoost\n89.84% AUC', '#90EE90'),
    ('Phase 1\nQ1 2025', 5, '🧠 Deep Learning\nRNN/LSTM\n+2-3% AUC', '#87CEEB'),
    ('Phase 2\nQ2 2025', 8, '⚡ Real-time\nOnline Learning\n<50ms latency', '#FFB6C1'),
    ('Phase 3\nQ3 2025', 11, '🌐 Multi-domain\nTransfer Learning\nGeneral Model', '#DDA0DD'),
]

for title, x, content, color in milestones:
    # Circle on timeline
    circle = plt.Circle((x, timeline_y), 0.3, color=color, ec='black', linewidth=2, zorder=5)
    ax.add_patch(circle)
    
    # Title above
    ax.text(x, timeline_y + 0.7, title, ha='center', fontsize=11, weight='bold')
    
    # Content box below
    box = FancyBboxPatch((x-1, timeline_y-3.5), 2, 2,
                         boxstyle="round,pad=0.15",
                         edgecolor=color, facecolor=color,
                         alpha=0.3, linewidth=2)
    ax.add_patch(box)
    
    # Content text
    for i, line in enumerate(content.split('\n')):
        ax.text(x, timeline_y - 1.8 - i*0.4, line, ha='center', fontsize=9, weight='bold')

# Bottom features
features_y = 1.5
feature_items = [
    ('📊 Ensemble Methods', 2),
    ('🎯 User Embeddings', 5),
    ('📱 Mobile API', 8),
    ('☁️ Cloud Deploy', 11),
]

for feature, x in feature_items:
    ax.text(x, features_y, feature, ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                     edgecolor='orange', linewidth=1))

plt.tight_layout()
plt.savefig('slide_images/slide19_future_roadmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide19_future_roadmap.png")
plt.close()

print("\n" + "="*60)
print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*60)
print(f"\nTotal images created: 9")
print(f"Saved in: slide_images/")
print("\nFiles created:")
print("  1. slide03_class_distribution.png")
print("  2. slide04_system_diagram.png")
print("  3. slide05_recommendation_types.png")
print("  4. slide06_xgboost_smote.png")
print("  5. slide07_dataset_overview.png")
print("  6. slide08_feature_engineering.png")
print("  7. slide09_methodology_flowchart.png")
print("  8. slide12_cross_validation.png")
print("  9. slide19_future_roadmap.png")
print("\n✨ Ready to use in your presentation!")

