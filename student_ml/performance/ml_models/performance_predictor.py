"""
PerformancePredictor - Linear Regression Performance Prediction

Clean separation of concerns:
- NO validation logic
- ONLY runs on pre-validated "human" data
- Uses engineered behavioral features (17 features)
- Focused solely on predicting performance index
"""

import os
import joblib
import numpy as np
from typing import Dict
from .feature_engineering import engineer_features, features_to_array, ML_FEATURE_NAMES


class PerformancePredictor:
    """
    Predicts student performance index using Linear Regression.
    
    IMPORTANT: This class assumes input has already been validated
    by HumanGatekeeper. It performs NO validation checks.
    
    Uses the same 17 engineered features as the Isolation Forest.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with trained regression model.
        
        Args:
            model_path: Path to trained performance prediction model (.pkl)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Performance model not found: {model_path}")
        
        # Load model data (contains model + metadata)
        model_data = joblib.load(model_path)
        
        # Handle both old format (just model) and new format (dict)
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', ML_FEATURE_NAMES)
            self.model_type = model_data.get('model_type', 'LinearRegression')
        else:
            # Old format - just the model
            self.model = model_data
            self.feature_names = ML_FEATURE_NAMES
            self.model_type = 'Unknown'
        
        print(f"✅ Loaded {self.model_type} performance model from {model_path}")
        print(f"   Features: {len(self.feature_names)}")
    
    def predict(self, data: Dict) -> float:
        """
        Predict performance index for validated human data.
        
        Args:
            data: Dictionary with keys:
                - hours_studied: float
                - previous_scores: float
                - extracurricular: bool
                - sleep_hours: float
                - sample_papers: int
        
        Returns:
            Predicted performance index (float)
        
        Note:
            Assumes data has been validated by HumanGatekeeper.
            Does NOT perform any validation or safety checks.
        """
        # Engineer behavioral features (same as Isolation Forest)
        features = engineer_features(data)
        
        # Convert to array in correct order
        feature_array = features_to_array(features, self.feature_names)
        
        # Reshape for single prediction
        feature_array = feature_array.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(feature_array)[0]
        
        return float(prediction)
    
    def predict_with_confidence(self, data: Dict) -> Dict:
        """
        Predict with confidence interval estimation.
        
        For Linear Regression, we estimate uncertainty using residual
        standard error from training (if available).
        
        Args:
            data: Validated student data dictionary
        
        Returns:
            Dictionary with:
                - prediction: float
                - std_dev: float (estimated, if available)
                - confidence_interval: tuple (approximate 95% CI)
        """
        prediction = self.predict(data)
        
        result = {
            'prediction': prediction,
            'std_dev': None,
            'confidence_interval': None
        }
        
        # For Linear Regression, we can't compute confidence intervals
        # without the training data variance. Return prediction only.
        # Could be extended by saving training metrics during training.
        
        return result
