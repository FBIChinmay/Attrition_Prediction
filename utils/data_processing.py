import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import joblib

# Import logger and config
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logger import setup_logger

logger = setup_logger("data_processing")

def handle_outliers(data: pd.DataFrame, column: str = 'TrainingTimesLastYear') -> pd.DataFrame:
    """
    Handles outliers in the specified column using the 3-sigma rule, replacing them with the mean.
    
    Args:
        data (pd.DataFrame): The input dataframe.
        column (str): The column to check for outliers.
        
    Returns:
        pd.DataFrame: DataFrame with handled outliers.
        
    Example:
        >>> df_cleaned = handle_outliers(df, 'TrainingTimesLastYear')
    """
    try:
        logger.info(f"Handling outliers for column: {column}")
        
        # Copy data to avoid SettingWithCopyWarning
        df = data.copy()
        
        if column in df.columns:
            mean_val = df[column].mean()
            std_val = df[column].std()
            
            # Calculate 3 sigma limits
            lower_limit = mean_val - 3 * std_val
            upper_limit = mean_val + 3 * std_val
            
            # Identify outliers and replace with mean
            outlier_mask = (df[column] < lower_limit) | (df[column] > upper_limit)
            num_outliers = outlier_mask.sum()
            
            if num_outliers > 0:
                # Ensure column is float so we can insert a float mean
                df[column] = df[column].astype(float)
                df.loc[outlier_mask, column] = mean_val
                logger.info(f"Replaced {num_outliers} outliers in {column} with mean value.")
            else:
                logger.info(f"No outliers found in {column}.")
        else:
            logger.warning(f"Column {column} not found in dataset. Skipping outlier handling.")
            
        return df
    except Exception as e:
        logger.error(f"Error handling outliers in column {column}: {e}")
        raise

def preprocess_data(data: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the data: drops columns, handles outliers, scales, and encodes.
    
    Args:
        data (pd.DataFrame): Raw input data.
        is_training (bool): If True, fits and saves encoders/scalers. If False, loads them for inference.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed features (X) and target (y). Target may be None during inference.
        
    Example:
        >>> X, y = preprocess_data(raw_data, is_training=True)
    """
    try:
        logger.info(f"Starting data preprocessing. is_training={is_training}")
        
        # 1. Handle Outliers
        df = handle_outliers(data, 'TrainingTimesLastYear')
        
        # 2. Extract Target if available
        y = None
        if 'Attrition' in df.columns:
            logger.info("Extracting target variable 'Attrition'")
            # Fit or load label encoder for target
            if is_training:
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(df['Attrition'])
                y = pd.Series(y_encoded, name='Attrition')
                joblib.dump({'Attrition': le_target}, config.LABEL_ENCODERS_PATH)
            else:
                le_dict = joblib.load(config.LABEL_ENCODERS_PATH)
                if 'Attrition' in le_dict:
                    y_encoded = le_dict['Attrition'].transform(df['Attrition'])
                    y = pd.Series(y_encoded, name='Attrition')
                    
            df = df.drop('Attrition', axis=1)
            
        # 3. Drop unneeded columns
        logger.info("Dropping unnecessary columns.")
        cols_to_drop = [col for col in config.COLUMNS_TO_DROP if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # 4. Label Encoding for binary categorical features (e.g., OverTime)
        label_cols = [col for col in config.LABEL_ENCODE_COLS if col != 'Attrition' and col in df.columns]
        if label_cols:
            if is_training:
                le_dict = joblib.load(config.LABEL_ENCODERS_PATH) if os.path.exists(config.LABEL_ENCODERS_PATH) else {}
                for col in label_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    le_dict[col] = le
                joblib.dump(le_dict, config.LABEL_ENCODERS_PATH)
            else:
                le_dict = joblib.load(config.LABEL_ENCODERS_PATH)
                for col in label_cols:
                    if col in le_dict:
                        df[col] = le_dict[col].transform(df[col])
                        
        # 5. One-Hot Encoding
        ohe_cols = [col for col in config.ONEHOT_ENCODE_COLS if col in df.columns]
        if ohe_cols:
            if is_training:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_arr = ohe.fit_transform(df[ohe_cols])
                encoded_cols = ohe.get_feature_names_out(ohe_cols)
                encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=df.index)
                
                df = df.drop(columns=ohe_cols)
                df = pd.concat([df, encoded_df], axis=1)
                
                joblib.dump(ohe, config.ONEHOT_ENCODER_PATH)
            else:
                ohe = joblib.load(config.ONEHOT_ENCODER_PATH)
                encoded_arr = ohe.transform(df[ohe_cols])
                encoded_cols = ohe.get_feature_names_out(ohe_cols)
                encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=df.index)
                
                df = df.drop(columns=ohe_cols)
                df = pd.concat([df, encoded_df], axis=1)
                
        # 6. Scaling Numerical Columns
        num_cols = [col for col in config.NUMERIC_COLS if col in df.columns]
        if num_cols:
            if is_training:
                scaler = StandardScaler()
                df[num_cols] = scaler.fit_transform(df[num_cols])
                joblib.dump(scaler, config.SCALER_PATH)
            else:
                scaler = joblib.load(config.SCALER_PATH)
                df[num_cols] = scaler.transform(df[num_cols])
                
        # 7. Ensure feature order matches training data
        if is_training:
            joblib.dump(list(df.columns), config.FEATURES_PATH)
        else:
            expected_features = joblib.load(config.FEATURES_PATH)
            # Add missing columns with 0s (in case of OHE mismatches in inference)
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0
            df = df[expected_features]
            
        logger.info("Data preprocessing completed successfully.")
        return df, y
        
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise
