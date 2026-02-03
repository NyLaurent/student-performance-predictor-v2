"""
HumanGatekeeper - Rule-Based Validation + ML Anomaly Detection

Separates concerns:
1. Physical/logical constraint enforcement (deterministic)
2. Human-likeness detection (ML-based Isolation Forest)
3. Bot pattern detection (heuristic)

Does NOT perform performance prediction.
"""

import numpy as np
from typing import Dict, Tuple
from .feature_engineering import engineer_features, features_to_array, ML_FEATURE_NAMES


class HumanGatekeeper:
    """
    Validates student data through multiple layers:
    
    1. Data type validation (NaN, infinity)
    2. Physical constraints (non-negotiable rules)
    3. ML-based human-likeness (Isolation Forest)
    4. Bot pattern detection (heuristic)
    
    Returns classification and reason for decision.
    """
    
    def __init__(self, human_likeness_model=None):
        """
        Initialize gatekeeper.
        
        Args:
            human_likeness_model: Trained Isolation Forest model (optional)
                                 If None, ML layer is skipped
        """
        self.human_likeness_model = human_likeness_model
    
    def classify(self, data: Dict) -> Tuple[str, str]:
        """
        Classify input through all validation layers.
        
        Returns:
            Tuple of (classification, reason)
            Classifications: 'invalid', 'unrealistic', 'suspicious', 'human'
        """
        # Layer 1: Data type validation
        validation_result = self._validate_data_types(data)
        if validation_result != "valid":
            return ("invalid", validation_result)
        
        # Extract values for further checks
        hours_studied = data.get('hours_studied', 0)
        previous_scores = data.get('previous_scores', 0)
        sleep_hours = data.get('sleep_hours', 0)
        sample_papers = data.get('sample_papers', 0)
        
        # Layer 2: Physical constraints (DETERMINISTIC)
        physics_check = self._check_physical_constraints(
            hours_studied, previous_scores, sleep_hours, sample_papers
        )
        if physics_check != "valid":
            return ("unrealistic", physics_check)
        
        # Layer 3: ML-based human-likeness detection
        if self.human_likeness_model is not None:
            ml_check = self._check_ml_human_likeness(data)
            if ml_check != "valid":
                return ("suspicious", ml_check)
        
        # Layer 4: Heuristic bot detection (weak, does not override ML)
        bot_check = self._check_bot_patterns(
            hours_studied, previous_scores, sleep_hours, sample_papers
        )
        if bot_check != "valid":
            return ("suspicious", bot_check)
        
        return ("human", "Data appears realistic and human-like")
    
    def _validate_data_types(self, data: Dict) -> str:
        """
        Layer 1: Validate data types and format.
        
        Checks for:
        - NaN values
        - Infinity values
        - Type conversion errors
        """
        try:
            hours_studied = float(data.get('hours_studied', 0))
            previous_scores = float(data.get('previous_scores', 0))
            sleep_hours = float(data.get('sleep_hours', 0))
            sample_papers = float(data.get('sample_papers', 0))
            
            # Check for NaN or infinity
            values = [hours_studied, previous_scores, sleep_hours, sample_papers]
            for val in values:
                if np.isnan(val):
                    return "Contains NaN (Not a Number) values"
                if np.isinf(val):
                    return "Contains infinite values"
            
            return "valid"
        except (ValueError, TypeError) as e:
            return f"Invalid data type: {str(e)}"
    
    def _check_physical_constraints(
        self, hours_studied: float, previous_scores: float,
        sleep_hours: float, sample_papers: float
    ) -> str:
        """
        Layer 2: Enforce non-negotiable physical and domain constraints.
        
        These rules are DETERMINISTIC and MUST NOT be replaced by ML.
        """
        # === TIME CONSTRAINTS ===
        
        if hours_studied < 0:
            return "Hours studied cannot be negative"
        
        if hours_studied > 24:
            return "Hours studied cannot exceed 24 hours per day"
        
        if sleep_hours < 0:
            return "Sleep hours cannot be negative"
        
        if sleep_hours > 24:
            return "Sleep hours cannot exceed 24 hours per day"
        
        # Critical: Total time cannot exceed 24 hours
        total_hours = hours_studied + sleep_hours
        if total_hours > 24:
            return f"Hours studied ({hours_studied}) + Sleep hours ({sleep_hours}) = {total_hours} exceeds 24 hours per day"
        
        # Medical constraint: Maximum realistic sleep
        if sleep_hours > 16:
            return "Sleeping more than 16 hours per day is medically unrealistic"
        
        # === SCORE CONSTRAINTS ===
        
        if previous_scores < 0:
            return "Previous scores cannot be negative"
        
        if previous_scores > 100:
            return "Previous scores cannot exceed 100"
        
        # === PRACTICE CONSTRAINTS ===
        
        if sample_papers < 0:
            return "Sample papers cannot be negative"
        
        if sample_papers > 50:
            return "Sample papers practiced exceeds reasonable limit (>50)"
        
        # === LOGICAL CONSISTENCY ===
        
        # Extreme imbalance: studying most of the day with minimal sleep
        if hours_studied > 20 and sleep_hours < 2:
            return "Studying >20 hours with <2 hours sleep is not sustainable"
        
        # Cannot practice many papers without any study time
        if hours_studied < 0.5 and sample_papers > 10:
            return "Cannot practice >10 sample papers with <0.5 hours of study"
        
        # Suspicious perfect scores with zero effort
        if previous_scores >= 95 and hours_studied < 0.5 and sample_papers == 0:
            return "Near-perfect scores with zero effort is suspicious"
        
        return "valid"
    
    def _check_ml_human_likeness(self, data: Dict) -> str:
        """
        Layer 3: ML-based anomaly detection using Isolation Forest.
        
        Uses engineered features to detect:
        - Statistical outliers
        - Unusual behavioral patterns
        - Non-human data distributions
        
        Returns "valid" if human-like, or reason if anomalous.
        """
        try:
            # Engineer behavioral features
            features = engineer_features(data)
            feature_array = features_to_array(features, ML_FEATURE_NAMES)
            
            # Reshape for single prediction
            feature_array = feature_array.reshape(1, -1)
            
            # Predict: 1 = human-like, -1 = anomaly
            prediction = self.human_likeness_model.predict(feature_array)[0]
            
            if prediction == -1:
                # Get anomaly score for explanation
                score = self.human_likeness_model.score_samples(feature_array)[0]
                return f"ML anomaly detection flagged unusual pattern (score: {score:.3f})"
            
            return "valid"
            
        except Exception as e:
            # If ML fails, don't block - log and continue
            print(f"Warning: ML human-likeness check failed: {e}")
            return "valid"
    
    def _check_bot_patterns(
        self, hours_studied: float, previous_scores: float,
        sleep_hours: float, sample_papers: float
    ) -> str:
        """
        Layer 4: Heuristic bot detection (WEAK - does not override ML).
        
        Detects obvious automation patterns:
        - All values are perfect multiples
        - Repeating digits
        - Sequential patterns
        
        This is SUPPLEMENTARY to ML detection.
        """
        values = [hours_studied, previous_scores, sleep_hours, sample_papers]
        
        # Check for all values being multiples of 10 (very suspicious)
        non_zero_values = [v for v in values if v != 0]
        if non_zero_values and all(v % 10 == 0 for v in non_zero_values):
            return "All non-zero values are multiples of 10 - bot-like pattern"
        
        # Check for repeating digits (e.g., 11, 22, 77, 99)
        rounded = [int(v) for v in values]
        repeating_count = sum(1 for v in rounded if v > 10 and v % 11 == 0)
        if repeating_count >= 3:
            return "Multiple repeating digit values detected - bot-like pattern"
        
        # Check for perfect sequential pattern
        sorted_rounded = sorted(rounded)
        if len(set(sorted_rounded)) == len(sorted_rounded):  # All unique
            if sorted_rounded[-1] - sorted_rounded[0] == len(sorted_rounded) - 1:
                return "Sequential number pattern detected - bot-like behavior"
        
        return "valid"
