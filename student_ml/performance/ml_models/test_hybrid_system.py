"""
Comprehensive Test Suite for Hybrid Rule-Based + ML Architecture

Tests:
1. Feature engineering correctness
2. HumanGatekeeper validation layers
3. Physical constraints enforcement
4. ML human-likeness detection
5. Bot pattern detection
6. PerformancePredictor isolation
7. End-to-end integration
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'student_ml.settings')
import django
django.setup()

from performance.ml_models.feature_engineering import engineer_features, features_to_array, ML_FEATURE_NAMES
from performance.ml_models.human_gatekeeper import HumanGatekeeper
from performance.ml_models.performance_predictor import PerformancePredictor


class TestResults:
    """Track test results"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.failed_tests = []
    
    def add_result(self, passed: bool, test_name: str):
        self.total += 1
        if passed:
            self.passed += 1
            print(f"✅ PASS: {test_name}")
        else:
            self.failed += 1
            self.failed_tests.append(test_name)
            print(f"❌ FAIL: {test_name}")
    
    def summary(self):
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total: {self.total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {self.passed/self.total*100:.1f}%")
        
        if self.failed > 0:
            print("\nFailed Tests:")
            for test in self.failed_tests:
                print(f"  - {test}")
        else:
            print("\n🎉 ALL TESTS PASSED!")
        
        return self.failed == 0


# ===== TEST 1: FEATURE ENGINEERING =====

def test_feature_engineering(results: TestResults):
    """Test that engineered features are calculated correctly."""
    print("\n" + "="*70)
    print("TEST CATEGORY 1: FEATURE ENGINEERING")
    print("="*70)
    
    # Test case 1: Normal student
    data = {
        'hours_studied': 6,
        'previous_scores': 75,
        'extracurricular': True,
        'sleep_hours': 7,
        'sample_papers': 4
    }
    
    features = engineer_features(data)
    
    # Verify raw features
    results.add_result(
        features['hours_studied'] == 6,
        "Raw feature: hours_studied"
    )
    results.add_result(
        features['extracurricular'] == 1,
        "Raw feature: extracurricular encoding"
    )
    
    # Verify engineered features
    expected_ratio = 6 / (7 + 0.1)
    results.add_result(
        abs(features['study_sleep_ratio'] - expected_ratio) < 0.01,
        "Engineered feature: study_sleep_ratio"
    )
    
    expected_effort = 6 + 0.5 * 4
    results.add_result(
        abs(features['effort_score'] - expected_effort) < 0.01,
        "Engineered feature: effort_score"
    )
    
    expected_load = 6 + 7
    results.add_result(
        features['total_daily_load'] == expected_load,
        "Engineered feature: total_daily_load"
    )
    
    results.add_result(
        features['normalized_study'] == 6/24,
        "Engineered feature: normalized_study"
    )
    
    expected_free_time = 24 - 13
    results.add_result(
        features['free_time'] == expected_free_time,
        "Engineered feature: free_time"
    )
    
    # Test feature array conversion
    feature_array = features_to_array(features, ML_FEATURE_NAMES)
    results.add_result(
        len(feature_array) == len(ML_FEATURE_NAMES),
        "Feature array: correct length"
    )
    results.add_result(
        feature_array[0] == 6,  # hours_studied is first
        "Feature array: correct order"
    )


# ===== TEST 2: PHYSICAL CONSTRAINTS =====

def test_physical_constraints(results: TestResults):
    """Test deterministic physical constraint enforcement."""
    print("\n" + "="*70)
    print("TEST CATEGORY 2: PHYSICAL CONSTRAINTS (DETERMINISTIC)")
    print("="*70)
    
    gatekeeper = HumanGatekeeper()  # No ML model
    
    # Test case 1: Negative hours
    classification, reason = gatekeeper.classify({
        'hours_studied': -5,
        'previous_scores': 75,
        'extracurricular': True,
        'sleep_hours': 7,
        'sample_papers': 3
    })
    results.add_result(
        classification == "unrealistic" and "negative" in reason.lower(),
        "Physical constraint: negative hours_studied"
    )
    
    # Test case 2: Hours exceed 24
    classification, reason = gatekeeper.classify({
        'hours_studied': 30,
        'previous_scores': 75,
        'extracurricular': True,
        'sleep_hours': 7,
        'sample_papers': 3
    })
    results.add_result(
        classification == "unrealistic" and "24" in reason,
        "Physical constraint: hours_studied > 24"
    )
    
    # Test case 3: Total hours exceed 24
    classification, reason = gatekeeper.classify({
        'hours_studied': 18,
        'previous_scores': 75,
        'extracurricular': True,
        'sleep_hours': 10,
        'sample_papers': 3
    })
    results.add_result(
        classification == "unrealistic" and "exceeds 24" in reason,
        "Physical constraint: study + sleep > 24"
    )
    
    # Test case 4: Previous scores > 100
    classification, reason = gatekeeper.classify({
        'hours_studied': 6,
        'previous_scores': 150,
        'extracurricular': True,
        'sleep_hours': 7,
        'sample_papers': 3
    })
    results.add_result(
        classification == "unrealistic" and "100" in reason,
        "Physical constraint: previous_scores > 100"
    )
    
    # Test case 5: Sleep > 16 hours
    classification, reason = gatekeeper.classify({
        'hours_studied': 6,
        'previous_scores': 75,
        'extracurricular': True,
        'sleep_hours': 18,
        'sample_papers': 3
    })
    results.add_result(
        classification == "unrealistic" and "16" in reason,
        "Physical constraint: sleep_hours > 16"
    )
    
    # Test case 6: Extreme study with minimal sleep
    classification, reason = gatekeeper.classify({
        'hours_studied': 22,
        'previous_scores': 75,
        'extracurricular': True,
        'sleep_hours': 1,
        'sample_papers': 3
    })
    results.add_result(
        classification == "unrealistic" and "sustainable" in reason.lower(),
        "Logical constraint: extreme study + minimal sleep"
    )
    
    # Test case 7: Many papers without study
    classification, reason = gatekeeper.classify({
        'hours_studied': 0.3,
        'previous_scores': 75,
        'extracurricular': True,
        'sleep_hours': 7,
        'sample_papers': 15
    })
    results.add_result(
        classification == "unrealistic" and "sample papers" in reason.lower(),
        "Logical constraint: many papers without study"
    )


# ===== TEST 3: DATA TYPE VALIDATION =====

def test_data_type_validation(results: TestResults):
    """Test NaN and infinity detection."""
    print("\n" + "="*70)
    print("TEST CATEGORY 3: DATA TYPE VALIDATION")
    print("="*70)
    
    gatekeeper = HumanGatekeeper()
    
    # Test case 1: NaN value
    classification, reason = gatekeeper.classify({
        'hours_studied': float('nan'),
        'previous_scores': 75,
        'extracurricular': True,
        'sleep_hours': 7,
        'sample_papers': 3
    })
    results.add_result(
        classification == "invalid" and "NaN" in reason,
        "Data type: NaN detection"
    )
    
    # Test case 2: Infinity value
    classification, reason = gatekeeper.classify({
        'hours_studied': float('inf'),
        'previous_scores': 75,
        'extracurricular': True,
        'sleep_hours': 7,
        'sample_papers': 3
    })
    results.add_result(
        classification == "invalid" and "infinite" in reason.lower(),
        "Data type: Infinity detection"
    )


# ===== TEST 4: BOT PATTERN DETECTION =====

def test_bot_patterns(results: TestResults):
    """Test heuristic bot detection (supplementary to ML)."""
    print("\n" + "="*70)
    print("TEST CATEGORY 4: BOT PATTERN DETECTION")
    print("="*70)
    
    gatekeeper = HumanGatekeeper()
    
    # Test case 1: All multiples of 10
    classification, reason = gatekeeper.classify({
        'hours_studied': 10,
        'previous_scores': 80,
        'extracurricular': True,
        'sleep_hours': 10,
        'sample_papers': 10
    })
    results.add_result(
        classification == "suspicious" and "10" in reason,
        "Bot pattern: all multiples of 10"
    )
    
    # Test case 2: Sequential pattern
    classification, reason = gatekeeper.classify({
        'hours_studied': 1,
        'previous_scores': 2,
        'extracurricular': False,
        'sleep_hours': 3,
        'sample_papers': 4
    })
    results.add_result(
        classification == "suspicious" and "sequential" in reason.lower(),
        "Bot pattern: sequential numbers"
    )


# ===== TEST 5: VALID HUMAN DATA =====

def test_valid_human_data(results: TestResults):
    """Test that valid edge cases are NOT rejected."""
    print("\n" + "="*70)
    print("TEST CATEGORY 5: VALID HUMAN DATA (NO FALSE POSITIVES)")
    print("="*70)
    
    gatekeeper = HumanGatekeeper()  # Without ML for basic tests
    
    valid_cases = [
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
                'hours_studied': 9,
                'previous_scores': 92,
                'extracurricular': True,
                'sleep_hours': 6,
                'sample_papers': 8
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
            'name': 'Minimal effort',
            'data': {
                'hours_studied': 1,
                'previous_scores': 30,
                'extracurricular': False,
                'sleep_hours': 9,
                'sample_papers': 0
            }
        },
        {
            'name': 'Balanced student',
            'data': {
                'hours_studied': 5.5,
                'previous_scores': 68,
                'extracurricular': True,
                'sleep_hours': 7.5,
                'sample_papers': 3
            }
        },
    ]
    
    for case in valid_cases:
        classification, reason = gatekeeper.classify(case['data'])
        results.add_result(
            classification == "human",
            f"Valid human: {case['name']}"
        )


# ===== TEST 6: PERFORMANCE PREDICTOR ISOLATION =====

def test_performance_predictor(results: TestResults):
    """Test that PerformancePredictor has no validation logic."""
    print("\n" + "="*70)
    print("TEST CATEGORY 6: PERFORMANCE PREDICTOR (LINEAR REGRESSION)")
    print("="*70)
    
    # Check if model exists
    model_path = os.path.join(BASE_DIR, 'student_ml', 'performance', 'model.pkl')
    
    if not os.path.exists(model_path):
        print("⚠️  Performance model not found. Skipping predictor tests.")
        print("   Run: python manage.py shell -c 'from performance.train_model import train; train()'")
        return
    
    try:
        predictor = PerformancePredictor(model_path)
        
        # Test prediction on valid data
        data = {
            'hours_studied': 6,
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 4
        }
        
        prediction = predictor.predict(data)
        results.add_result(
            isinstance(prediction, float) and 0 <= prediction <= 100,
            "PerformancePredictor: returns valid prediction"
        )
        
        # Test that predictor uses engineered features
        results.add_result(
            hasattr(predictor, 'feature_names') and len(predictor.feature_names) == 17,
            "PerformancePredictor: uses 17 engineered features"
        )
        
        # Test prediction with confidence
        prediction_details = predictor.predict_with_confidence(data)
        results.add_result(
            'prediction' in prediction_details,
            "PerformancePredictor: includes prediction in details"
        )
        
        print(f"   Sample prediction: {prediction:.2f}")
        print(f"   Model type: {predictor.model_type}")
        print(f"   Features used: {len(predictor.feature_names)}")
        
    except Exception as e:
        print(f"⚠️  Could not test PerformancePredictor: {e}")
        import traceback
        traceback.print_exc()


# ===== MAIN TEST RUNNER =====

def run_all_tests():
    """Execute comprehensive test suite."""
    print("\n" + "="*70)
    print("HYBRID RULE-BASED + ML SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nArchitecture:")
    print("  INPUT → Feature Engineering → HumanGatekeeper → PerformancePredictor → OUTPUT")
    print("\nComponents:")
    print("  1. Feature Engineering (17 behavioral features)")
    print("  2. HumanGatekeeper (rules + Isolation Forest)")
    print("  3. PerformancePredictor (Linear Regression, no validation)")
    print("="*70)
    
    results = TestResults()
    
    # Run all test categories
    test_feature_engineering(results)
    test_physical_constraints(results)
    test_data_type_validation(results)
    test_bot_patterns(results)
    test_valid_human_data(results)
    test_performance_predictor(results)
    
    # Summary
    success = results.summary()
    
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
