"""
Train Linear Regression Model for Performance Prediction

Uses engineered behavioral features (17 total) to predict performance index.
This model runs ONLY on data validated as "human" by the Isolation Forest.
"""

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Import feature engineering
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from performance.ml_models.feature_engineering import engineer_features, ML_FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train():
    """Train Linear Regression model using engineered behavioral features."""
    
    print("\n" + "="*70)
    print("TRAINING LINEAR REGRESSION - PERFORMANCE PREDICTION MODEL")
    print("="*70)
    
    # Load dataset
    dataset_path = os.path.join(BASE_DIR, 'performance', 'dataset.csv')
    df = pd.read_csv(dataset_path)
    
    print(f"\n📊 Loaded {len(df)} records from dataset")
    
    # Encode categorical column
    le = LabelEncoder()
    df['Extracurricular Activities'] = le.fit_transform(
        df['Extracurricular Activities']
    )
    
    # Filter for realistic data only (same as Isolation Forest training)
    df_filtered = df[
        (df['Hours Studied'] >= 0) &
        (df['Hours Studied'] <= 24) &
        (df['Sleep Hours'] >= 0) &
        (df['Sleep Hours'] <= 24) &
        (df['Hours Studied'] + df['Sleep Hours'] <= 24) &
        (df['Previous Scores'] >= 0) &
        (df['Previous Scores'] <= 100) &
        (df['Sample Question Papers Practiced'] >= 0) &
        (df['Sample Question Papers Practiced'] <= 50)
    ].copy()
    
    print(f"📋 Filtered to {len(df_filtered)} valid human records")
    
    # Engineer features for all records
    print("\n🔧 Engineering behavioral features...")
    engineered_data = []
    targets = []
    
    for _, row in df_filtered.iterrows():
        data_dict = {
            'hours_studied': row['Hours Studied'],
            'previous_scores': row['Previous Scores'],
            'extracurricular': bool(row['Extracurricular Activities']),
            'sleep_hours': row['Sleep Hours'],
            'sample_papers': int(row['Sample Question Papers Practiced'])
        }
        
        features = engineer_features(data_dict)
        engineered_data.append(features)
        targets.append(row['Performance Index'])
    
    # Convert to DataFrame
    features_df = pd.DataFrame(engineered_data)
    X = features_df[ML_FEATURE_NAMES].values
    y = np.array(targets)
    
    print(f"\n📈 Training data shape: {X.shape}")
    print(f"   Features: {len(ML_FEATURE_NAMES)}")
    print(f"   Samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n🔀 Train/Test Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Train Linear Regression
    print("\n🤖 Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    train_predictions = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    
    # Evaluate on test set
    test_predictions = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print("\n📊 Model Performance:")
    print("\n   Training Set:")
    print(f"     - R² Score: {train_r2:.4f}")
    print(f"     - RMSE: {train_rmse:.4f}")
    print(f"     - MAE: {train_mae:.4f}")
    
    print("\n   Test Set:")
    print(f"     - R² Score: {test_r2:.4f}")
    print(f"     - RMSE: {test_rmse:.4f}")
    print(f"     - MAE: {test_mae:.4f}")
    
    # Feature importance (coefficients)
    print("\n📌 Feature Importance (Top 10 by coefficient magnitude):")
    feature_importance = pd.DataFrame({
        'feature': ML_FEATURE_NAMES,
        'coefficient': model.coef_
    })
    feature_importance['abs_coef'] = np.abs(feature_importance['coefficient'])
    feature_importance = feature_importance.sort_values('abs_coef', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s}: {row['coefficient']:8.4f}")
    
    # Save model
    model_path = os.path.join(BASE_DIR, 'performance', 'model.pkl')
    
    # Save both model and feature names for consistency
    model_data = {
        'model': model,
        'feature_names': ML_FEATURE_NAMES,
        'model_type': 'LinearRegression'
    }
    
    joblib.dump(model_data, model_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    print("="*70)
    
    return model


if __name__ == "__main__":
    train()
