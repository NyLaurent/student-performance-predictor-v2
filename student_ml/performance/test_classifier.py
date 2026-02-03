"""
Test cases for the Enhanced Human Classifier
Demonstrates all validation layers and edge cases
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'student_ml.settings')
import django
django.setup()

from performance.human_classifier import HumanClassifier

# Initialize classifier
DATASET_PATH = os.path.join(BASE_DIR, 'performance', 'dataset.csv')
classifier = HumanClassifier(DATASET_PATH)

def test_case(description, data, expected_classification):
    """Run a single test case and print results"""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"Input: {data}")
    classification, reason = classifier.classify(data)
    status = "✅ PASS" if classification == expected_classification else "❌ FAIL"
    print(f"Result: {classification} - {reason}")
    print(f"{status} (Expected: {expected_classification})")
    return classification == expected_classification

def run_all_tests():
    """Run comprehensive test suite"""
    print("\n" + "="*70)
    print("ENHANCED HUMAN CLASSIFIER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = []
    
    # ====== VALID HUMAN DATA ======
    print("\n\n📋 CATEGORY 1: VALID HUMAN DATA")
    results.append(test_case(
        "Normal student data",
        {
            'hours_studied': 5.5,
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 3
        },
        "human"
    ))
    
    results.append(test_case(
        "High achiever",
        {
            'hours_studied': 8,
            'previous_scores': 90,
            'extracurricular': True,
            'sleep_hours': 6.5,
            'sample_papers': 8
        },
        "human"
    ))
    
    # ====== INVALID DATA (TYPE/FORMAT ERRORS) ======
    print("\n\n❌ CATEGORY 2: INVALID DATA (Type/Format Errors)")
    results.append(test_case(
        "NaN value",
        {
            'hours_studied': float('nan'),
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 3
        },
        "invalid"
    ))
    
    results.append(test_case(
        "Infinity value",
        {
            'hours_studied': float('inf'),
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 3
        },
        "invalid"
    ))
    
    # ====== UNREALISTIC DATA (RANGE VIOLATIONS) ======
    print("\n\n⚠️  CATEGORY 3: UNREALISTIC DATA (Range Violations)")
    results.append(test_case(
        "Negative hours studied",
        {
            'hours_studied': -5,
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 3
        },
        "unrealistic"
    ))
    
    results.append(test_case(
        "Hours studied > 24",
        {
            'hours_studied': 30,
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 3
        },
        "unrealistic"
    ))
    
    results.append(test_case(
        "Previous scores > 100",
        {
            'hours_studied': 5,
            'previous_scores': 150,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 3
        },
        "unrealistic"
    ))
    
    results.append(test_case(
        "Negative previous scores",
        {
            'hours_studied': 5,
            'previous_scores': -20,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 3
        },
        "unrealistic"
    ))
    
    results.append(test_case(
        "Excessive sample papers",
        {
            'hours_studied': 5,
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 100
        },
        "unrealistic"
    ))
    
    # ====== UNREALISTIC DATA (LOGICAL INCONSISTENCIES) ======
    print("\n\n⚠️  CATEGORY 4: UNREALISTIC DATA (Logical Inconsistencies)")
    results.append(test_case(
        "Study + Sleep > 24 hours",
        {
            'hours_studied': 18,
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 10,
            'sample_papers': 3
        },
        "unrealistic"
    ))
    
    results.append(test_case(
        "Extreme study hours with minimal sleep",
        {
            'hours_studied': 22,
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 1,
            'sample_papers': 3
        },
        "unrealistic"
    ))
    
    results.append(test_case(
        "Excessive sleep hours",
        {
            'hours_studied': 5,
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 18,
            'sample_papers': 3
        },
        "unrealistic"
    ))
    
    results.append(test_case(
        "Many papers without studying",
        {
            'hours_studied': 0.5,
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 15
        },
        "unrealistic"
    ))
    
    results.append(test_case(
        "Perfect score with zero effort",
        {
            'hours_studied': 0,
            'previous_scores': 98,
            'extracurricular': False,
            'sleep_hours': 8,
            'sample_papers': 0
        },
        "unrealistic"
    ))
    
    results.append(test_case(
        "Very low score despite extreme effort",
        {
            'hours_studied': 18,
            'previous_scores': 15,
            'extracurricular': True,
            'sleep_hours': 5,
            'sample_papers': 15
        },
        "unrealistic"
    ))
    
    # ====== SUSPICIOUS DATA (BOT PATTERNS) ======
    print("\n\n🤖 CATEGORY 5: SUSPICIOUS DATA (Bot Patterns)")
    results.append(test_case(
        "All values multiples of 10",
        {
            'hours_studied': 10,
            'previous_scores': 80,
            'extracurricular': True,
            'sleep_hours': 10,
            'sample_papers': 10
        },
        "suspicious"
    ))
    
    results.append(test_case(
        "Repeating digit pattern",
        {
            'hours_studied': 11,
            'previous_scores': 77,
            'extracurricular': True,
            'sleep_hours': 11,
            'sample_papers': 5
        },
        "suspicious"
    ))
    
    results.append(test_case(
        "Sequential numbers pattern",
        {
            'hours_studied': 1,
            'previous_scores': 2,
            'extracurricular': False,
            'sleep_hours': 3,
            'sample_papers': 4
        },
        "suspicious"
    ))
    
    # ====== EDGE CASES ======
    print("\n\n🔍 CATEGORY 6: EDGE CASES")
    results.append(test_case(
        "All zeros",
        {
            'hours_studied': 0,
            'previous_scores': 0,
            'extracurricular': False,
            'sleep_hours': 0,
            'sample_papers': 0
        },
        "suspicious"  # This should be caught as unrealistic or suspicious
    ))
    
    results.append(test_case(
        "Maximum valid values",
        {
            'hours_studied': 12,
            'previous_scores': 100,
            'extracurricular': True,
            'sleep_hours': 8,
            'sample_papers': 10
        },
        "human"
    ))
    
    results.append(test_case(
        "Minimal but valid values",
        {
            'hours_studied': 1,
            'previous_scores': 30,
            'extracurricular': False,
            'sleep_hours': 6,
            'sample_papers': 0
        },
        "human"
    ))
    
    # ====== SUMMARY ======
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100
    print(f"Passed: {passed}/{total} ({percentage:.1f}%)")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()
