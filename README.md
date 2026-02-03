# Student Performance Prediction System

A machine learning-powered Django REST API that predicts student performance index based on various academic and lifestyle factors.

## Overview

This project implements a predictive model to estimate student performance using machine learning. The system analyzes factors such as study hours, previous scores, extracurricular activities, sleep patterns, and practice question papers to predict a performance index.

## Features

- **Machine Learning Model**: Random Forest Regressor for accurate performance prediction
- **Enhanced Data Validation**: 5-layer validation system to prevent bias and detect unrealistic data
- **Human vs Bot Classification**: Detects automated submissions and suspicious patterns
- **REST API**: Django REST Framework-based API for easy integration
- **Modern Web Interface**: Responsive form with real-time feedback
- **Data Management**: Automated data loading from CSV to Django models
- **Model Training**: Script to train and save ML models
- **Database Integration**: SQLite database with Django ORM

## Technology Stack

- **Backend**: Django 6.0+
- **API Framework**: Django REST Framework
- **Machine Learning**: Scikit-learn (Random Forest)
- **Data Processing**: Pandas, NumPy
- **Database**: SQLite
- **Serialization**: Joblib for model persistence

## Dataset

The system uses a student performance dataset with the following features:

- Hours Studied
- Previous Scores
- Extracurricular Activities (Yes/No)
- Sleep Hours
- Sample Question Papers Practiced
- Performance Index (target variable)

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd "Student Performance Index"
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install django djangorestframework scikit-learn pandas numpy joblib
   ```

4. **Navigate to project directory**:

   ```bash
   cd student_ml
   ```

5. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

## Usage

### 1. Load Data into Database

Load the student performance data into the Django database:

```bash
python manage.py shell -c "from performance.load_data import run; run()"
```

### 2. Train the Machine Learning Model

Train the Random Forest model:

```bash
python manage.py shell -c "from performance.train_model import train; train()"
```

This will create `performance/model.pkl` file containing the trained model.

### 3. Run the Development Server

Start the Django development server:

```bash
python manage.py runserver
```

The API will be available at `http://127.0.0.1:8000/`

### 4. Access the Web Interface

Navigate to the prediction form:

```
http://127.0.0.1:8000/api/form/
```

### 5. Run Tests (Optional)

Test the enhanced classifier with comprehensive validation scenarios:

```bash
cd performance
python test_classifier.py
```

## 🏗️ Hybrid Rule-Based + ML Architecture (v2.0)

This system uses a **sophisticated hybrid architecture** that combines deterministic rules with machine learning:

### Architecture Overview

```
INPUT → HumanGatekeeper → PerformancePredictor → OUTPUT
        (Validation)      (Prediction)
```

### Key Components

1. **HumanGatekeeper** - 4-layer validation system:
   - Data type validation (NaN, infinity)
   - Physical constraints (deterministic rules)
   - ML human-likeness (Isolation Forest)
   - Bot pattern detection (heuristic)

2. **Feature Engineering** - 17 behavioral features:
   - Raw features (5): hours_studied, previous_scores, etc.
   - Derived features (12): study_sleep_ratio, effort_score, work_life_balance, etc.

3. **Human-Likeness Model** - Isolation Forest trained on valid data:
   - Replaces z-score thresholds with learned anomaly detection
   - Adapts to data distribution
   - Reduces bias from hand-tuned parameters

4. **PerformancePredictor** - Pure prediction (no validation):
   - Runs ONLY on validated "human" data
   - Returns prediction + confidence interval
   - Clean separation of concerns

### Classification Types

- ✅ **HUMAN** - Valid data, proceeds to prediction
- ❌ **INVALID** - Data type/format errors
- ⚠️ **UNREALISTIC** - Violates physical/logical constraints  
- 🤖 **SUSPICIOUS** - ML anomaly detection or bot patterns

### Why This Architecture?

✅ **Reduces bias** - ML learns from data vs hard-coded thresholds  
✅ **Separates concerns** - Validation ⊥ Prediction  
✅ **Deterministic where needed** - Physics rules remain strict  
✅ **Probabilistic where appropriate** - Human-likeness is learned  
✅ **Adaptable** - Retrain on new data to evolve

For complete architecture documentation, see [HYBRID_ARCHITECTURE.md](HYBRID_ARCHITECTURE.md).

## API Documentation

### Predict Student Performance

**Endpoint**: `POST /api/predict/`

**Request Body**:

```json
{
  "hours_studied": 7,
  "previous_scores": 85,
  "extracurricular": true,
  "sleep_hours": 8,
  "sample_papers": 3
}
```

**Success Response** (Human Data):

```json
{
  "status": "success",
  "classification": "human",
  "reason": "Data appears realistic and human-like",
  "predicted_performance_index": 78.45,
  "message": "Prediction completed successfully"
}
```

**Error Response** (Unrealistic Data):

```json
{
  "status": "rejected",
  "classification": "unrealistic",
  "reason": "Hours studied (18) + Sleep hours (10) = 28 hours exceeds 24 hours per day",
  "message": "Your input violates basic physical or logical constraints"
}
```

**Warning Response** (Suspicious Pattern):

```json
{
  "status": "flagged",
  "classification": "suspicious",
  "reason": "All values are multiples of 10 - suspicious pattern",
  "message": "Your input shows unusual patterns or is a statistical outlier"
}
```

**Field Descriptions**:

- `hours_studied`: Float (0-24) - Number of hours studied per day
- `previous_scores`: Float (0-100) - Previous academic scores
- `extracurricular`: Boolean - Whether student participates in extracurricular activities
- `sleep_hours`: Float (0-24) - Hours of sleep per day
- `sample_papers`: Integer (0-50) - Number of sample question papers practiced

**Validation Rules**:

- Hours studied + Sleep hours must not exceed 24 hours per day
- All values must be within realistic ranges
- Logical consistency checks are applied (e.g., can't practice many papers without studying)
- Statistical outlier detection flags extreme values
- Bot pattern detection prevents automated spam submissions

## Project Structure

```
student_ml/
├── dataset.csv                    # Student performance dataset
├── db.sqlite3                     # SQLite database
├── manage.py                      # Django management script
├── performance/                   # Main Django app
│   ├── models.py                  # StudentPerformance model
│   ├── views.py                   # API views with enhanced validation
│   ├── urls.py                    # App URL patterns
│   ├── serializers.py             # Data serializers
│   ├── human_classifier.py        # Enhanced 5-layer validation system
│   ├── train_model.py             # ML model training script
│   ├── load_data.py               # Data loading script
│   ├── test_classifier.py         # Comprehensive validation tests
│   ├── templates/                 # Frontend templates
│   │   └── performance/
│   │       └── form.html          # Web interface
│   ├── migrations/                # Database migrations
│   └── tests.py                   # Unit tests
└── student_ml/                    # Project settings
    ├── settings.py                # Django settings
    ├── urls.py                    # Main URL patterns
    ├── wsgi.py                    # WSGI configuration
    └── asgi.py                    # ASGI configuration
```

## Model Details

The system uses a Random Forest Regressor with:

- **n_estimators**: 200
- **random_state**: 42 for reproducibility

The model is trained on 80% of the data and validated on 20%.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Testing

### Run Validation Tests

The system includes comprehensive tests for all validation layers:

```bash
cd student_ml/performance
python test_classifier.py
```

This will test:
- Valid human data scenarios
- Invalid data detection (NaN, infinity)
- Range violation detection
- Logical inconsistency detection
- Bot pattern detection
- Edge cases

Expected output: 25+ test cases covering all validation scenarios.

## Future Enhancements

- Add more ML algorithms (SVM, Neural Networks)
- Implement model comparison and selection
- Add data visualization dashboard
- Deploy to cloud platform
- Add authentication and user management
- Implement model retraining pipeline
- Machine learning-based anomaly detection
- Rate limiting and CAPTCHA integration
- Adaptive validation thresholds
# student-performance-predictor-v2
