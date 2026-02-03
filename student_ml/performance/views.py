from django.shortcuts import render
import os
import joblib
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Import new hybrid architecture components
from .ml_models.human_gatekeeper import HumanGatekeeper
from .ml_models.performance_predictor import PerformancePredictor

# Paths
MODEL_PATH = os.path.join(settings.BASE_DIR, 'performance', 'model.pkl')
HUMAN_LIKENESS_MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'performance', 'ml_models', 'human_likeness_model.pkl'
)

# Initialize components
def load_human_likeness_model():
    """Load trained Isolation Forest model for human-likeness detection."""
    if os.path.exists(HUMAN_LIKENESS_MODEL_PATH):
        try:
            return joblib.load(HUMAN_LIKENESS_MODEL_PATH)
        except Exception as e:
            print(f"Warning: Could not load human-likeness model: {e}")
            return None
    return None

# Initialize gatekeeper with ML model (if available)
human_likeness_model = load_human_likeness_model()
gatekeeper = HumanGatekeeper(human_likeness_model=human_likeness_model)

# Initialize performance predictor (lazy load)
performance_predictor = None


def home(request):
    """Render the home page with guidelines."""
    return render(request, 'performance/home.html')


def performance_form(request):
    """Render the prediction form."""
    return render(request, 'performance/form.html')



# @api_view(['POST'])
# def predict_performance(request):
#     model = load_model()
#     if model is None:
#         return Response(
#             {"error": "Model not trained yet"},
#             status=400
#         )

#     data = request.data

#     # Input validation
#     try:
#         hours_studied = float(data.get('hours_studied', 0))
#         previous_scores = float(data.get('previous_scores', 0))
#         extracurricular = bool(data.get('extracurricular', False))
#         sleep_hours = float(data.get('sleep_hours', 0))
#         sample_papers = int(data.get('sample_papers', 0))
#     except (ValueError, TypeError):
#         return Response(
#             {"error": "Invalid data types provided"},
#             status=400
#         )

#     errors = []
#     if not 0 <= hours_studied <= 24:
#         errors.append("hours_studied must be between 0 and 24")
#     if not 0 <= sleep_hours <= 24:
#         errors.append("sleep_hours must be between 0 and 24")
#     if not 0 <= previous_scores <= 100:
#         errors.append("previous_scores must be between 0 and 100")
#     if sample_papers < 0:
#         errors.append("sample_papers cannot be negative")

#     if errors:
#         return Response({"errors": errors}, status=400)

#     features = np.array([[  
#         hours_studied,
#         previous_scores,
#         1 if extracurricular else 0,
#         sleep_hours,
#         sample_papers
#     ]])

#     prediction = model.predict(features)[0]

#     return Response({
#         "predicted_performance_index": round(float(prediction), 2)
#     })



@api_view(['POST'])
def classify_and_predict(request):
    """
    Hybrid Rule-Based + ML Classification & Prediction Endpoint.
    
    Architecture:
        INPUT → Feature Engineering → HumanGatekeeper → PerformancePredictor → OUTPUT
        
    HumanGatekeeper performs:
        1. Data type validation
        2. Physical constraints enforcement
        3. ML-based human-likeness detection (Isolation Forest - 17 features)
        4. Bot pattern detection
    
    PerformancePredictor:
        - Runs ONLY if input classified as "human"
        - Uses Linear Regression with same 17 engineered features
        - No validation logic - pure performance prediction
    """
    global performance_predictor
    
    # Lazy load performance predictor
    if performance_predictor is None:
        if not os.path.exists(MODEL_PATH):
            return Response({
                "error": "Performance prediction model not trained yet. Run train_model.py first."
            }, status=400)
        
        try:
            performance_predictor = PerformancePredictor(MODEL_PATH)
        except Exception as e:
            return Response({
                "error": f"Failed to load performance model: {str(e)}"
            }, status=500)
    
    data = request.data
    
    # Parse and normalize input data
    try:
        features_data = {
            'hours_studied': float(data.get('hours_studied', 0)),
            'previous_scores': float(data.get('previous_scores', 0)),
            'extracurricular': bool(data.get('extracurricular', False)),
            'sleep_hours': float(data.get('sleep_hours', 0)),
            'sample_papers': int(data.get('sample_papers', 0))
        }
    except (ValueError, TypeError) as e:
        return Response({
            "status": "error",
            "classification": "invalid",
            "reason": f"Invalid data types provided: {str(e)}",
            "message": "Please provide valid numeric values for all fields"
        }, status=400)
    
    # === STEP 1: GATEKEEPER VALIDATION ===
    # Run through all validation layers:
    # - Data type validation
    # - Physical constraints (deterministic)
    # - ML human-likeness (Isolation Forest)
    # - Bot pattern detection (heuristic)
    
    classification, reason = gatekeeper.classify(features_data)
    
    # === STEP 2: HANDLE CLASSIFICATION RESULT ===
    
    if classification == "invalid":
        # Data type or format errors
        return Response({
            "status": "error",
            "classification": "invalid",
            "reason": reason,
            "message": "Your input contains invalid or malformed data"
        }, status=400)
    
    elif classification == "unrealistic":
        # Violates physical or logical constraints
        return Response({
            "status": "rejected",
            "classification": "unrealistic",
            "reason": reason,
            "message": "Your input violates basic physical or logical constraints"
        }, status=200)
    
    elif classification == "suspicious":
        # ML flagged as anomaly OR bot pattern detected
        return Response({
            "status": "flagged",
            "classification": "suspicious",
            "reason": reason,
            "message": "Your input shows unusual patterns or statistical anomalies"
        }, status=200)
    
    elif classification == "human":
        # === STEP 3: PERFORMANCE PREDICTION ===
        # Input passed all validation - safe to predict
        
        try:
            # Predict performance index using Linear Regression
            prediction = performance_predictor.predict(features_data)
            
            response_data = {
                "status": "success",
                "classification": "human",
                "reason": reason,
                "predicted_performance_index": round(float(prediction), 2),
                "message": "Prediction completed successfully"
            }
            
            return Response(response_data)
            
        except Exception as e:
            return Response({
                "status": "error",
                "classification": "error",
                "reason": f"Prediction failed: {str(e)}",
                "message": "An error occurred during prediction"
            }, status=500)
    
    else:
        # Unexpected classification (should never happen)
        return Response({
            "status": "error",
            "classification": "unknown",
            "reason": f"Unexpected classification result: {classification}",
            "message": "An internal error occurred"
        }, status=500)
