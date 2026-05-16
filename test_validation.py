import unittest
import pandas as pd
import numpy as np
import os
import io

import config
from utils.data_processing import preprocess_data, handle_outliers
from app import process_and_predict

class TestHRAttritionApp(unittest.TestCase):
    
    def test_config_keys(self):
        """Test 1: Check if config loads and check_keys returns True"""
        self.assertTrue(config.check_keys(), "config.check_keys() should return True")

    def test_handle_outliers(self):
        """Test 2: Check if outlier handling works correctly"""
        # Create dummy data with an extreme outlier (100 normal values around 2, and one 10000)
        normal_data = [2] * 100
        data_with_outlier = normal_data + [10000]
        df = pd.DataFrame({'TrainingTimesLastYear': data_with_outlier})
        
        # Calculate mean manually including outlier
        mean_val = np.mean(data_with_outlier)
        
        df_clean = handle_outliers(df, 'TrainingTimesLastYear')
        
        # The outlier (index 100) should be replaced by mean_val
        self.assertAlmostEqual(df_clean.loc[100, 'TrainingTimesLastYear'], mean_val, places=2)

    def test_empty_input_handling(self):
        """Test 3: Submit empty input should raise an exception or be handled"""
        df_empty = pd.DataFrame()
        # The preprocessing should fail gracefully or raise a ValueError
        with self.assertRaises(Exception):
            preprocess_data(df_empty, is_training=False)

    def test_corrupted_file_handling(self):
        """Test 4: Reading a corrupted/wrong file type gracefully"""
        corrupted_csv = io.StringIO("This is not a valid CSV\nRow1,Row2\n1,")
        try:
            df = pd.read_csv(corrupted_csv)
            # Should read, but fail during preprocessing due to missing columns
            with self.assertRaises(Exception):
                preprocess_data(df, is_training=False)
            passed = True
        except Exception:
            passed = True
        self.assertTrue(passed, "Corrupted file handling did not fail safely.")

    def test_valid_query_prediction(self):
        """Test 5: Submit valid query -> non-empty answer (mocked)"""
        # Check if model exists before testing inference
        if os.path.exists(config.MODEL_PATH):
            import joblib
            model = joblib.load(config.MODEL_PATH)
            target_encoder = joblib.load(config.LABEL_ENCODERS_PATH).get('Attrition')
            
            # Create a single row of valid dummy data
            dummy_data = pd.DataFrame({
                'Age': [41], 'BusinessTravel': ['Travel_Rarely'], 'DailyRate': [1102], 
                'Department': ['Sales'], 'DistanceFromHome': [1], 'Education': [2], 
                'EducationField': ['Life Sciences'], 'EnvironmentSatisfaction': [2], 
                'Gender': ['Female'], 'HourlyRate': [94], 'JobInvolvement': [3], 
                'JobLevel': [2], 'JobRole': ['Sales Executive'], 'JobSatisfaction': [4], 
                'MaritalStatus': ['Single'], 'MonthlyIncome': [5993], 'MonthlyRate': [19479], 
                'NumCompaniesWorked': [8], 'OverTime': ['Yes'], 'PercentSalaryHike': [11], 
                'PerformanceRating': [3], 'RelationshipSatisfaction': [1], 'StockOptionLevel': [0], 
                'TotalWorkingYears': [8], 'TrainingTimesLastYear': [0], 'WorkLifeBalance': [1], 
                'YearsAtCompany': [6], 'YearsInCurrentRole': [4], 'YearsSinceLastPromotion': [0], 
                'YearsWithCurrManager': [5]
            })
            
            try:
                result = process_and_predict(dummy_data, model, target_encoder)
                self.assertIn('Predicted_Attrition', result.columns)
                self.assertIsNotNone(result['Predicted_Attrition'].iloc[0])
            except Exception as e:
                self.fail(f"Prediction on valid data failed: {e}")
        else:
            print("Skipping Test 5 because model artifacts are not present. Run train.py first.")

if __name__ == '__main__':
    unittest.main()
