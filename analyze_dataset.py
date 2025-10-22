import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('dataset/2019-Oct.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

print("\nEvent types distribution:")
print(df['event_type'].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# Data preprocessing
print("\nPreprocessing data...")

# Convert event_time to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Create binary target: 1 for purchase, 0 for not purchase
df['purchased'] = (df['event_type'] == 'purchase').astype(int)

# Feature engineering
print("Creating features...")

# User-level features
user_features = df.groupby('user_id').agg({
    'purchased': ['sum', 'count', 'mean'],
    'price': ['mean', 'std', 'min', 'max'],
    'product_id': 'nunique',
    'category_id': 'nunique',
    'event_time': ['min', 'max']
}).reset_index()

# Flatten column names
user_features.columns = ['user_id', 'total_purchases', 'total_events', 'purchase_rate', 
                        'avg_price', 'price_std', 'min_price', 'max_price', 
                        'unique_products', 'unique_categories', 'first_event', 'last_event']

# Calculate session duration
user_features['session_duration_days'] = (user_features['last_event'] - user_features['first_event']).dt.days

# Product-level features
product_features = df.groupby('product_id').agg({
    'purchased': ['sum', 'count', 'mean'],
    'price': 'mean',
    'user_id': 'nunique'
}).reset_index()

product_features.columns = ['product_id', 'product_purchases', 'product_views', 'product_purchase_rate', 
                           'product_price', 'unique_users']

# Category-level features
category_features = df.groupby('category_id').agg({
    'purchased': ['sum', 'count', 'mean'],
    'price': 'mean',
    'user_id': 'nunique'
}).reset_index()

category_features.columns = ['category_id', 'category_purchases', 'category_views', 'category_purchase_rate', 
                            'category_price', 'category_users']

# Merge features
print("Merging features...")

# Create main dataset with user-product interactions
main_df = df[['user_id', 'product_id', 'category_id', 'price', 'purchased']].copy()

# Merge user features
main_df = main_df.merge(user_features, on='user_id', how='left')

# Merge product features
main_df = main_df.merge(product_features, on='product_id', how='left')

# Merge category features
main_df = main_df.merge(category_features, on='category_id', how='left')

# Fill missing values
main_df = main_df.fillna(0)

# Create additional features
main_df['price_ratio'] = main_df['price'] / (main_df['avg_price'] + 1e-8)
main_df['is_expensive'] = (main_df['price'] > main_df['avg_price']).astype(int)
main_df['price_category'] = pd.cut(main_df['price'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])

# Encode categorical variables
le_category = LabelEncoder()
main_df['price_category_encoded'] = le_category.fit_transform(main_df['price_category'].astype(str))

# Prepare features for modeling
feature_columns = ['price', 'total_purchases', 'total_events', 'purchase_rate', 
                  'avg_price', 'price_std', 'min_price', 'max_price', 
                  'unique_products', 'unique_categories', 'session_duration_days',
                  'product_purchases', 'product_views', 'product_purchase_rate', 
                  'product_price', 'unique_users', 'category_purchases', 
                  'category_views', 'category_purchase_rate', 'category_price', 
                  'category_users', 'price_ratio', 'is_expensive', 'price_category_encoded']

X = main_df[feature_columns]
y = main_df['purchased']

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# Save processed data
main_df.to_csv('processed_data.csv', index=False)
print("Processed data saved to 'processed_data.csv'")

print("\nDataset analysis completed!")