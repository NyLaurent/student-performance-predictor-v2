"""
Feature Engineering for Student Performance Prediction

Provides derived behavioral features that capture relationships
between raw inputs, improving model performance and interpretability.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union


def engineer_features(data: Dict[str, Union[float, int, bool]]) -> Dict[str, float]:
    """
    Create derived behavioral features from raw student data.
    
    These features capture:
    - Study efficiency patterns
    - Work-life balance indicators
    - Effort-outcome relationships
    - Time allocation patterns
    
    Args:
        data: Dictionary with keys:
            - hours_studied: float (0-24)
            - previous_scores: float (0-100)
            - extracurricular: bool
            - sleep_hours: float (0-24)
            - sample_papers: int (>=0)
    
    Returns:
        Dictionary of engineered features
    """
    hours_studied = float(data.get('hours_studied', 0))
    previous_scores = float(data.get('previous_scores', 0))
    extracurricular = 1 if data.get('extracurricular', False) else 0
    sleep_hours = float(data.get('sleep_hours', 0))
    sample_papers = int(data.get('sample_papers', 0))
    
    # ===== BEHAVIORAL FEATURES =====
    
    # Study-sleep balance ratio
    # High ratio = study-focused; Low ratio = sleep-focused
    study_sleep_ratio = hours_studied / (sleep_hours + 0.1)  # Avoid division by zero
    
    # Overall effort score (weighted combination)
    # Combines study hours with practice papers
    effort_score = hours_studied + 0.5 * sample_papers
    
    # Score-effort gap (performance relative to effort)
    # Positive = efficient; Negative = struggling
    score_effort_gap = previous_scores - (effort_score * 5)  # Normalized to ~0-100 scale
    
    # Total daily time commitment
    # Measures overall busy-ness
    total_daily_load = hours_studied + sleep_hours
    
    # Normalized study time (as fraction of day)
    normalized_study = hours_studied / 24.0
    
    # Normalized sleep time (as fraction of day)
    normalized_sleep = sleep_hours / 24.0
    
    # Free time remaining (hours available for other activities)
    free_time = 24.0 - total_daily_load
    
    # Study intensity (papers per study hour)
    # High = efficient practice; Low = less structured study
    study_intensity = sample_papers / (hours_studied + 0.1)
    
    # Score efficiency (score per hour studied)
    # Measures learning effectiveness
    score_efficiency = previous_scores / (hours_studied + 0.1)
    
    # Work-life balance indicator
    # 1 = balanced, <1 = overworked, >1 = under-utilizing time
    work_life_balance = (sleep_hours * 2) / (hours_studied + 0.1)  # Sleep should be ~2x study
    
    # Extracurricular engagement score
    # Interaction between activities and time management
    extracurricular_engagement = extracurricular * (free_time / 24.0)
    
    # Study consistency indicator
    # High when study and practice are proportional
    study_consistency = min(hours_studied, sample_papers) / (max(hours_studied, sample_papers) + 0.1)
    
    return {
        # Raw features
        'hours_studied': hours_studied,
        'previous_scores': previous_scores,
        'extracurricular': extracurricular,
        'sleep_hours': sleep_hours,
        'sample_papers': sample_papers,
        
        # Engineered features
        'study_sleep_ratio': study_sleep_ratio,
        'effort_score': effort_score,
        'score_effort_gap': score_effort_gap,
        'total_daily_load': total_daily_load,
        'normalized_study': normalized_study,
        'normalized_sleep': normalized_sleep,
        'free_time': free_time,
        'study_intensity': study_intensity,
        'score_efficiency': score_efficiency,
        'work_life_balance': work_life_balance,
        'extracurricular_engagement': extracurricular_engagement,
        'study_consistency': study_consistency,
    }


def features_to_array(features: Dict[str, float], feature_names: list = None) -> np.ndarray:
    """
    Convert feature dictionary to numpy array in consistent order.
    
    Args:
        features: Dictionary of features from engineer_features()
        feature_names: Optional list of feature names to use (for consistency)
    
    Returns:
        Numpy array of feature values
    """
    if feature_names is None:
        # Default feature order for ML models
        feature_names = [
            'hours_studied',
            'previous_scores',
            'extracurricular',
            'sleep_hours',
            'sample_papers',
            'study_sleep_ratio',
            'effort_score',
            'score_effort_gap',
            'total_daily_load',
            'normalized_study',
            'normalized_sleep',
            'free_time',
            'study_intensity',
            'score_efficiency',
            'work_life_balance',
            'extracurricular_engagement',
            'study_consistency',
        ]
    
    return np.array([features.get(name, 0.0) for name in feature_names])


def features_to_dataframe(features: Dict[str, float]) -> pd.DataFrame:
    """
    Convert feature dictionary to pandas DataFrame (single row).
    
    Args:
        features: Dictionary of features from engineer_features()
    
    Returns:
        DataFrame with one row containing all features
    """
    return pd.DataFrame([features])


# Feature names used by ML models
ML_FEATURE_NAMES = [
    'hours_studied',
    'previous_scores',
    'extracurricular',
    'sleep_hours',
    'sample_papers',
    'study_sleep_ratio',
    'effort_score',
    'score_effort_gap',
    'total_daily_load',
    'normalized_study',
    'normalized_sleep',
    'free_time',
    'study_intensity',
    'score_efficiency',
    'work_life_balance',
    'extracurricular_engagement',
    'study_consistency',
]

# Feature names used by performance prediction model (original features only)
PERFORMANCE_FEATURE_NAMES = [
    'Hours Studied',
    'Previous Scores',
    'Extracurricular Activities',
    'Sleep Hours',
    'Sample Question Papers Practiced'
]
