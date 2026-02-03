"""
Train Human-Likeness Anomaly Detection Model

Uses Isolation Forest to learn what "normal" human student data looks like.
Trained ONLY on valid, realistic human data from the dataset.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from .feature_engineering import engineer_features, ML_FEATURE_NAMES


def load_and_prepare_data(dataset_path: str) -> pd.DataFrame:
    """
    Load dataset and prepare features for training.
    
    Args:
        dataset_path: Path to dataset.csv
    
    Returns:
        DataFrame with engineered features
    """
    # Load raw dataset
    df = pd.read_csv(dataset_path)
    
    print(f"Loaded {len(df)} records from dataset")
    
    # Encode categorical column
    le = LabelEncoder()
    df['Extracurricular Activities'] = le.fit_transform(
        df['Extracurricular Activities']
    )
    
    # Filter for realistic data only (basic sanity checks)
    # This ensures we train ONLY on valid human data
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
    
    print(f"Filtered to {len(df_filtered)} valid records")
    
    # Engineer features for all records
    engineered_data = []
    
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
    
    # Convert to DataFrame
    features_df = pd.DataFrame(engineered_data)
    
    print(f"Engineered {len(features_df.columns)} features")
    print(f"Features: {list(features_df.columns)}")
    
    return features_df


def train_human_likeness_model(
    dataset_path: str,
    output_path: str,
    contamination: float = 0.05,
    n_estimators: int = 100,
    max_samples: str = 'auto',
    random_state: int = 42
) -> IsolationForest:
    """
    Train Isolation Forest for human-likeness detection.
    
    Args:
        dataset_path: Path to dataset.csv
        output_path: Path to save trained model
        contamination: Expected proportion of outliers (default: 5%)
        n_estimators: Number of trees in the forest
        max_samples: Number of samples to draw for training each tree
        random_state: Random seed for reproducibility
    
    Returns:
        Trained IsolationForest model
    """
    print("\n" + "="*70)
    print("TRAINING HUMAN-LIKENESS ANOMALY DETECTION MODEL")
    print("="*70)
    
    # Load and prepare data
    features_df = load_and_prepare_data(dataset_path)
    
    # Select features for training (consistent order)
    X = features_df[ML_FEATURE_NAMES].values
    
    print(f"\nTraining data shape: {X.shape}")
    print(f"Features used: {len(ML_FEATURE_NAMES)}")
    
    # Initialize Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    print("\nTraining Isolation Forest...")
    print(f"  - contamination: {contamination}")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - max_samples: {max_samples}")
    
    # Train model
    model.fit(X)
    
    # Evaluate on training data
    predictions = model.predict(X)
    scores = model.score_samples(X)
    
    n_inliers = np.sum(predictions == 1)
    n_outliers = np.sum(predictions == -1)
    
    print("\nTraining Results:")
    print(f"  - Total samples: {len(predictions)}")
    print(f"  - Classified as human-like: {n_inliers} ({n_inliers/len(predictions)*100:.1f}%)")
    print(f"  - Classified as anomalous: {n_outliers} ({n_outliers/len(predictions)*100:.1f}%)")
    print(f"  - Mean anomaly score: {np.mean(scores):.4f}")
    print(f"  - Std anomaly score: {np.std(scores):.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    
    print(f"\n✅ Model saved to: {output_path}")
    print("="*70)
    
    return model


def test_model_on_samples(model: IsolationForest):
    """
    Test trained model on sample inputs.
    """
    print("\n" + "="*70)
    print("TESTING MODEL ON SAMPLE INPUTS")
    print("="*70)
    
    test_cases = [
        {
            'name': 'Normal student',
            'data': {
                'hours_studied': 6,
                'previous_scores': 75,
                'extracurricular': True,
                'sleep_hours': 7,
                'sample_papers': 4
            }
        },
        {
            'name': 'High achiever',
            'data': {
                'hours_studied': 8,
                'previous_scores': 92,
                'extracurricular': True,
                'sleep_hours': 6.5,
                'sample_papers': 9
            }
        },
        {
            'name': 'Struggling student',
            'data': {
                'hours_studied': 2,
                'previous_scores': 45,
                'extracurricular': False,
                'sleep_hours': 8,
                'sample_papers': 1
            }
        },
        {
            'name': 'Extreme values (suspicious)',
            'data': {
                'hours_studied': 0.1,
                'previous_scores': 100,
                'extracurricular': False,
                'sleep_hours': 23,
                'sample_papers': 0
            }
        },
        {
            'name': 'All multiples of 10 (bot)',
            'data': {
                'hours_studied': 10,
                'previous_scores': 80,
                'extracurricular': True,
                'sleep_hours': 10,
                'sample_papers': 10
            }
        }
    ]
    
    for test in test_cases:
        features = engineer_features(test['data'])
        feature_array = np.array([features[name] for name in ML_FEATURE_NAMES]).reshape(1, -1)
        
        prediction = model.predict(feature_array)[0]
        score = model.score_samples(feature_array)[0]
        
        result = "✅ HUMAN-LIKE" if prediction == 1 else "🚨 ANOMALOUS"
        
        print(f"\n{test['name']}:")
        print(f"  Input: {test['data']}")
        print(f"  Result: {result}")
        print(f"  Anomaly Score: {score:.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, 'performance', 'dataset.csv')
    MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, 'performance', 'ml_models', 'human_likeness_model.pkl')
    
    # Train model
    model = train_human_likeness_model(
        dataset_path=DATASET_PATH,
        output_path=MODEL_OUTPUT_PATH,
        contamination=0.05,  # Expect 5% of training data to be outliers
        n_estimators=150,
        random_state=42
    )
    
    # Test model
    test_model_on_samples(model)
