"""
Setup Script for Hybrid Rule-Based + ML System

Automates:
1. Training Isolation Forest for human-likeness detection
2. Training Random Forest for performance prediction
3. Running comprehensive tests
4. Validating system readiness
"""

import os
import sys
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'student_ml.settings')
import django
django.setup()

from performance.train_model import train as train_performance_model


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text)
    print("="*70)


def check_dataset():
    """Verify dataset exists"""
    print_header("STEP 1: CHECKING DATASET")
    
    dataset_path = os.path.join(BASE_DIR, 'performance', 'dataset.csv')
    
    if os.path.exists(dataset_path):
        import pandas as pd
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset found: {dataset_path}")
        print(f"   Records: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        return True
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Please ensure dataset.csv is in the performance folder")
        return False


def train_human_likeness():
    """Train Isolation Forest for human-likeness detection"""
    print_header("STEP 2: TRAINING HUMAN-LIKENESS MODEL (ISOLATION FOREST)")
    
    try:
        # Import and run training
        from performance.ml_models.train_human_likeness import train_human_likeness_model
        
        dataset_path = os.path.join(BASE_DIR, 'performance', 'dataset.csv')
        output_path = os.path.join(BASE_DIR, 'performance', 'ml_models', 'human_likeness_model.pkl')
        
        model = train_human_likeness_model(
            dataset_path=dataset_path,
            output_path=output_path,
            contamination=0.05,
            n_estimators=150,
            random_state=42
        )
        
        print("\n✅ Human-likeness model trained successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to train human-likeness model: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_performance():
    """Train Linear Regression for performance prediction"""
    print_header("STEP 3: TRAINING PERFORMANCE PREDICTION MODEL (LINEAR REGRESSION)")
    
    try:
        train_performance_model()
        print("\n✅ Linear Regression performance model trained successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to train performance model: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tests():
    """Run comprehensive test suite"""
    print_header("STEP 4: RUNNING COMPREHENSIVE TESTS")
    
    try:
        from performance.ml_models.test_hybrid_system import run_all_tests
        
        success = run_all_tests()
        
        if success:
            print("\n✅ All tests passed")
        else:
            print("\n⚠️  Some tests failed - review output above")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Failed to run tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_models():
    """Verify both models are saved correctly"""
    print_header("STEP 5: VERIFYING MODEL FILES")
    
    model_path = os.path.join(BASE_DIR, 'performance', 'model.pkl')
    ml_model_path = os.path.join(BASE_DIR, 'performance', 'ml_models', 'human_likeness_model.pkl')
    
    models_ok = True
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"✅ Performance model: {model_path}")
        print(f"   Size: {size:.2f} MB")
    else:
        print(f"❌ Performance model not found: {model_path}")
        models_ok = False
    
    if os.path.exists(ml_model_path):
        size = os.path.getsize(ml_model_path) / (1024 * 1024)  # MB
        print(f"✅ Human-likeness model: {ml_model_path}")
        print(f"   Size: {size:.2f} MB")
    else:
        print(f"❌ Human-likeness model not found: {ml_model_path}")
        models_ok = False
    
    return models_ok


def test_api_integration():
    """Test that API can load models"""
    print_header("STEP 6: TESTING API INTEGRATION")
    
    try:
        from performance.ml_models.human_gatekeeper import HumanGatekeeper
        from performance.ml_models.performance_predictor import PerformancePredictor
        import joblib
        
        # Load models
        ml_model_path = os.path.join(BASE_DIR, 'performance', 'ml_models', 'human_likeness_model.pkl')
        model_path = os.path.join(BASE_DIR, 'performance', 'model.pkl')
        
        ml_model = joblib.load(ml_model_path)
        gatekeeper = HumanGatekeeper(human_likeness_model=ml_model)
        predictor = PerformancePredictor(model_path)
        
        # Test sample prediction
        test_data = {
            'hours_studied': 6,
            'previous_scores': 75,
            'extracurricular': True,
            'sleep_hours': 7,
            'sample_papers': 4
        }
        
        classification, reason = gatekeeper.classify(test_data)
        
        if classification == "human":
            prediction = predictor.predict(test_data)
            print(f"✅ API integration test passed")
            print(f"   Test input: {test_data}")
            print(f"   Classification: {classification}")
            print(f"   Prediction: {prediction:.2f}")
            return True
        else:
            print(f"⚠️  Test data classified as: {classification}")
            print(f"   Reason: {reason}")
            return False
            
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete setup process"""
    print("\n" + "="*70)
    print("HYBRID RULE-BASED + ML SYSTEM - SETUP SCRIPT")
    print("="*70)
    print("\nThis script will:")
    print("  1. Check dataset availability")
    print("  2. Train Isolation Forest (human-likeness - 17 features)")
    print("  3. Train Linear Regression (performance prediction - 17 features)")
    print("  4. Run comprehensive tests")
    print("  5. Verify model files")
    print("  6. Test API integration")
    print("\n" + "="*70)
    
    # Run setup steps
    results = []
    
    results.append(("Dataset Check", check_dataset()))
    
    if results[-1][1]:  # Only continue if dataset exists
        results.append(("Human-Likeness Training", train_human_likeness()))
        results.append(("Performance Training", train_performance()))
        results.append(("Comprehensive Tests", run_tests()))
        results.append(("Model Verification", verify_models()))
        results.append(("API Integration", test_api_integration()))
    
    # Summary
    print_header("SETUP SUMMARY")
    
    for step, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {step}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 SETUP COMPLETE - SYSTEM READY FOR PRODUCTION")
        print("="*70)
        print("\nNext steps:")
        print("  1. Start server: python manage.py runserver")
        print("  2. Access form: http://127.0.0.1:8000/api/form/")
        print("  3. Test API: curl -X POST http://127.0.0.1:8000/api/predict/ ...")
        print("\nDocumentation:")
        print("  - Architecture: HYBRID_ARCHITECTURE.md")
        print("  - Tests: python ml_models/test_hybrid_system.py")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  SETUP INCOMPLETE - REVIEW ERRORS ABOVE")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
