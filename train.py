import os
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys

import config
from utils.logger import setup_logger
from utils.data_processing import preprocess_data

logger = setup_logger("model_trainer")

def train_model() -> None:
    """
    Reads the dataset, preprocesses it, applies SMOTE for class balancing,
    trains a Gradient Boosting Classifier, and saves the model.
    
    Returns:
        None
        
    Example:
        >>> train_model()
    """
    try:
        logger.info("Starting model training pipeline.")
        
        # 1. Load Data
        if not os.path.exists(config.DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {config.DATA_PATH}. Please provide 'HR-Employee-Attrition.csv' in the project root.")
            
        data = pd.read_csv(config.DATA_PATH)
        logger.info(f"Loaded dataset with shape {data.shape}")
        
        # 2. Preprocess Data
        # is_training=True fits and saves scalers and encoders
        X, y = preprocess_data(data, is_training=True)
        
        if y is None:
            raise ValueError("Target variable 'Attrition' is missing from the dataset.")
            
        # 3. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # 4. Apply SMOTE for class imbalance
        logger.info("Applying SMOTE to balance the training data.")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"Resampled Train set shape: {X_train_resampled.shape}")
        
        # 5. Model Training
        logger.info("Training Gradient Boosting Classifier with optimal parameters.")
        model = GradientBoostingClassifier(**config.GB_PARAMS)
        model.fit(X_train_resampled, y_train_resampled)
        
        # 6. Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Model trained successfully. Test Accuracy: {acc:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # 7. Save Model
        if not os.path.exists(config.MODEL_DIR):
            os.makedirs(config.MODEL_DIR)
        joblib.dump(model, config.MODEL_PATH)
        logger.info(f"Model saved to {config.MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Error during model training pipeline: {e}")
        raise

if __name__ == "__main__":
    train_model()
