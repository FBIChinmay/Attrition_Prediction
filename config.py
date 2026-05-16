import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
DATA_PATH = os.path.join(BASE_DIR, 'HR-Employee-Attrition.csv')

# Files to save
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.joblib')
ONEHOT_ENCODER_PATH = os.path.join(MODEL_DIR, 'onehot_encoder.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.joblib')

# Feature Configuration
# Columns to drop
COLUMNS_TO_DROP = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']

# Categorical columns that need Label Encoding (binary/ordinal)
LABEL_ENCODE_COLS = ['Attrition', 'OverTime']

# Categorical columns that need One-Hot Encoding
ONEHOT_ENCODE_COLS = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']

# Numeric columns (derived by exclusion from the original data)
NUMERIC_COLS = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# Model hyperparameters for Gradient Boosting
GB_PARAMS = {
    'learning_rate': 0.1,
    'max_depth': 4,
    'n_estimators': 100,
    'random_state': 42
}

def check_keys() -> bool:
    """
    Validates the configuration by checking if essential keys and directories exist.
    
    Returns:
        bool: True if configuration is valid, False otherwise.
        
    Example:
        >>> check_keys()
        True
    """
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        # Check if basic settings are present
        if not isinstance(COLUMNS_TO_DROP, list):
            return False
        if not isinstance(GB_PARAMS, dict):
            return False
            
        return True
    except Exception as e:
        print(f"Error checking config: {e}")
        return False
