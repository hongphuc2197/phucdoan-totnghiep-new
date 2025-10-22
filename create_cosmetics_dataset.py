import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import joblib

# Create a realistic cosmetics dataset for testing our trained model
print("Creating cosmetics dataset for testing...")

# Load our trained model and scaler
try:
    model = joblib.load('best_model_xgboost.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Loaded trained XGBoost model and scaler successfully!")
except:
    print("Warning: Could not load trained model. Will create dataset anyway.")

# Define top 5 best-selling cosmetics products (based on real market data)
top_cosmetics = {
    'product_1': {
        'name': 'L\'Oréal Paris True Match Foundation',
        'category': 'Foundation',
        'price_range': (15, 25),
        'popularity_score': 0.95,
        'brand': 'L\'Oréal',
        'target_demographic': 'Women 18-45'
    },
    'product_2': {
        'name': 'Maybelline Fit Me Concealer',
        'category': 'Concealer',
        'price_range': (8, 15),
        'popularity_score': 0.92,
        'brand': 'Maybelline',
        'target_demographic': 'Women 16-50'
    },
    'product_3': {
        'name': 'MAC Ruby Woo Lipstick',
        'category': 'Lipstick',
        'price_range': (20, 30),
        'popularity_score': 0.88,
        'brand': 'MAC',
        'target_demographic': 'Women 20-40'
    },
    'product_4': {
        'name': 'Urban Decay Naked Eyeshadow Palette',
        'category': 'Eyeshadow',
        'price_range': (35, 50),
        'popularity_score': 0.85,
        'brand': 'Urban Decay',
        'target_demographic': 'Women 18-35'
    },
    'product_5': {
        'name': 'Fenty Beauty Pro Filt\'r Foundation',
        'category': 'Foundation',
        'price_range': (30, 40),
        'popularity_score': 0.90,
        'brand': 'Fenty Beauty',
        'target_demographic': 'Women 18-45'
    }
}

print(f"\nTop 5 Best-Selling Cosmetics Products:")
for i, (key, product) in enumerate(top_cosmetics.items(), 1):
    print(f"{i}. {product['name']} - {product['brand']} (${product['price_range'][0]}-${product['price_range'][1]})")

# Generate synthetic user behavior data for cosmetics
np.random.seed(42)
n_users = 10000
n_interactions = 50000

# Create user profiles
user_data = []
for user_id in range(1, n_users + 1):
    age = np.random.choice([18, 25, 35, 45, 55], p=[0.2, 0.3, 0.25, 0.15, 0.1])
    gender = np.random.choice(['F', 'M'], p=[0.8, 0.2])  # 80% female for cosmetics
    income_level = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
    beauty_enthusiast = np.random.choice([0, 1], p=[0.6, 0.4])
    
    user_data.append({
        'user_id': user_id,
        'age': age,
        'gender': gender,
        'income_level': income_level,
        'beauty_enthusiast': beauty_enthusiast
    })

user_df = pd.DataFrame(user_data)

# Generate interaction data
interactions = []
products = list(top_cosmetics.keys())

for _ in range(n_interactions):
    user_id = np.random.randint(1, n_users + 1)
    product_id = np.random.choice(products)
    product_info = top_cosmetics[product_id]
    
    # Get user profile
    user_profile = user_df[user_df['user_id'] == user_id].iloc[0]
    
    # Calculate purchase probability based on user profile and product
    base_prob = product_info['popularity_score']
    
    # Adjust based on user characteristics
    if user_profile['beauty_enthusiast']:
        base_prob *= 1.3
    
    if user_profile['income_level'] == 'high':
        base_prob *= 1.2
    elif user_profile['income_level'] == 'low':
        base_prob *= 0.8
    
    if user_profile['age'] in [18, 25, 35]:  # Target demographic
        base_prob *= 1.1
    
    # Generate price within range
    price = np.random.uniform(product_info['price_range'][0], product_info['price_range'][1])
    
    # Determine event type based on probability
    if np.random.random() < base_prob * 0.3:  # 30% of high probability interactions become purchases
        event_type = 'purchase'
        purchased = 1
    elif np.random.random() < 0.6:  # 60% become views
        event_type = 'view'
        purchased = 0
    elif np.random.random() < 0.8:  # 20% become cart additions
        event_type = 'cart'
        purchased = 0
    else:  # 20% become removals
        event_type = 'remove_from_cart'
        purchased = 0
    
    # Generate session data
    session_duration = np.random.exponential(300)  # Average 5 minutes
    pages_viewed = np.random.poisson(5)
    
    interactions.append({
        'user_id': user_id,
        'product_id': product_id,
        'product_name': product_info['name'],
        'category': product_info['category'],
        'brand': product_info['brand'],
        'price': round(price, 2),
        'event_type': event_type,
        'purchased': purchased,
        'session_duration': session_duration,
        'pages_viewed': pages_viewed,
        'age': user_profile['age'],
        'gender': user_profile['gender'],
        'income_level': user_profile['income_level'],
        'beauty_enthusiast': user_profile['beauty_enthusiast']
    })

cosmetics_df = pd.DataFrame(interactions)

print(f"\nGenerated cosmetics dataset:")
print(f"- Total interactions: {len(cosmetics_df)}")
print(f"- Unique users: {cosmetics_df['user_id'].nunique()}")
print(f"- Products: {cosmetics_df['product_id'].nunique()}")
print(f"- Purchase rate: {cosmetics_df['purchased'].mean():.3f}")

# Analyze product performance
print(f"\nProduct Performance Analysis:")
product_performance = cosmetics_df.groupby('product_id').agg({
    'purchased': ['count', 'sum', 'mean'],
    'price': 'mean'
}).round(3)

product_performance.columns = ['total_interactions', 'purchases', 'purchase_rate', 'avg_price']
product_performance = product_performance.sort_values('purchases', ascending=False)

for idx, (product_id, row) in enumerate(product_performance.iterrows(), 1):
    product_name = top_cosmetics[product_id]['name']
    print(f"{idx}. {product_name}")
    print(f"   - Total interactions: {row['total_interactions']}")
    print(f"   - Purchases: {row['purchases']}")
    print(f"   - Purchase rate: {row['purchase_rate']:.3f}")
    print(f"   - Avg price: ${row['avg_price']:.2f}")

# Save the dataset
cosmetics_df.to_csv('cosmetics_dataset.csv', index=False)
print(f"\nCosmetics dataset saved as 'cosmetics_dataset.csv'")

# Create feature engineering for the cosmetics dataset
print(f"\nCreating features for model testing...")

# User-level features
user_features = cosmetics_df.groupby('user_id').agg({
    'purchased': ['sum', 'count', 'mean'],
    'price': ['mean', 'std', 'min', 'max'],
    'product_id': 'nunique',
    'session_duration': 'mean',
    'pages_viewed': 'mean'
}).reset_index()

user_features.columns = ['user_id', 'total_purchases', 'total_events', 'purchase_rate', 
                        'avg_price', 'price_std', 'min_price', 'max_price', 
                        'unique_products', 'avg_session_duration', 'avg_pages_viewed']

# Product-level features
product_features = cosmetics_df.groupby('product_id').agg({
    'purchased': ['sum', 'count', 'mean'],
    'price': 'mean',
    'user_id': 'nunique'
}).reset_index()

product_features.columns = ['product_id', 'product_purchases', 'product_views', 'product_purchase_rate', 
                           'product_price', 'unique_users']

# Merge features
cosmetics_main = cosmetics_df[['user_id', 'product_id', 'price', 'purchased', 'age', 'gender', 'income_level', 'beauty_enthusiast']].copy()

cosmetics_main = cosmetics_main.merge(user_features, on='user_id', how='left')
cosmetics_main = cosmetics_main.merge(product_features, on='product_id', how='left')

# Create additional features
cosmetics_main['price_ratio'] = cosmetics_main['price'] / (cosmetics_main['avg_price'] + 1e-8)
cosmetics_main['is_expensive'] = (cosmetics_main['price'] > cosmetics_main['avg_price']).astype(int)

# Encode categorical variables
cosmetics_main['gender_encoded'] = cosmetics_main['gender'].map({'F': 1, 'M': 0})
cosmetics_main['income_encoded'] = cosmetics_main['income_level'].map({'low': 0, 'medium': 1, 'high': 2})

# Fill missing values
cosmetics_main = cosmetics_main.fillna(0)

# Prepare features for model testing
feature_columns = ['price', 'total_purchases', 'total_events', 'purchase_rate', 
                  'avg_price', 'price_std', 'min_price', 'max_price', 
                  'unique_products', 'avg_session_duration', 'avg_pages_viewed',
                  'product_purchases', 'product_views', 'product_purchase_rate', 
                  'product_price', 'unique_users', 'price_ratio', 'is_expensive',
                  'age', 'gender_encoded', 'income_encoded', 'beauty_enthusiast']

X_cosmetics = cosmetics_main[feature_columns]
y_cosmetics = cosmetics_main['purchased']

print(f"Cosmetics dataset features prepared:")
print(f"- Feature matrix shape: {X_cosmetics.shape}")
print(f"- Target distribution: {y_cosmetics.value_counts().to_dict()}")

# Save processed cosmetics data
cosmetics_main.to_csv('cosmetics_processed.csv', index=False)
print(f"Processed cosmetics data saved as 'cosmetics_processed.csv'")

print(f"\nCosmetics dataset creation completed!")