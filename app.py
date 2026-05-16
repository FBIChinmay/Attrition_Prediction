import streamlit as st
import pandas as pd
import joblib
import os
import sys

import config
from utils.logger import setup_logger
from utils.data_processing import preprocess_data

# Initialize logger
logger = setup_logger("streamlit_app")

# Page Configuration
st.set_page_config(
    page_title="HR Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: 800;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """
    Loads the trained model and LabelEncoder for the target.
    
    Returns:
        tuple: (model, target_encoder) or (None, None) if not found.
        
    Example:
        >>> model, encoder = load_artifacts()
    """
    try:
        logger.info("Loading model artifacts.")
        if not os.path.exists(config.MODEL_PATH):
            st.warning("Model not found. Please run `python train.py` first.")
            return None, None
            
        model = joblib.load(config.MODEL_PATH)
        le_dict = joblib.load(config.LABEL_ENCODERS_PATH)
        target_encoder = le_dict.get('Attrition', None)
        return model, target_encoder
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        st.error(f"Error loading model artifacts: {e}")
        return None, None

def process_and_predict(data: pd.DataFrame, model, target_encoder) -> pd.DataFrame:
    """
    Processes input data, predicts attrition, and decodes the prediction.
    
    Args:
        data (pd.DataFrame): Raw input dataframe.
        model: Trained Gradient Boosting model.
        target_encoder: LabelEncoder for decoding Attrition.
        
    Returns:
        pd.DataFrame: Original data with appended predictions.
        
    Example:
        >>> results = process_and_predict(df, model, le)
    """
    try:
        logger.info("Starting prediction process.")
        # Preprocess for inference (is_training=False)
        X, _ = preprocess_data(data, is_training=False)
        
        # Predict
        predictions = model.predict(X)
        
        # Decode predictions if possible
        if target_encoder:
            decoded_preds = target_encoder.inverse_transform(predictions)
        else:
            decoded_preds = ["Yes" if p == 1 else "No" for p in predictions]
            
        # Append to original data
        result_df = data.copy()
        result_df['Predicted_Attrition'] = decoded_preds
        
        logger.info("Prediction successful.")
        return result_df
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ValueError(f"Prediction failed: {str(e)}")

def main():
    st.title("👥 HR Attrition Prediction System")
    st.markdown("Upload employee data to predict attrition using our trained Machine Learning pipeline.")
    
    model, target_encoder = load_artifacts()
    
    if model is None:
        st.stop()
        
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            logger.info(f"File uploaded: {uploaded_file.name}")
            
            # Read CSV with encoding fallbacks for non-UTF-8 files
            def load_csv(file_obj):
                encodings = ['utf-8', 'latin1', 'cp1252']
                for enc in encodings:
                    try:
                        file_obj.seek(0)
                        return pd.read_csv(file_obj, encoding=enc)
                    except UnicodeDecodeError:
                        logger.warning(f"Failed to decode file with encoding {enc}. Trying next encoding.")
                file_obj.seek(0)
                raise UnicodeDecodeError('utf-8', b'', 0, 1, 'Unable to decode file with supported encodings.')

            df = load_csv(uploaded_file)
            
            # Check for empty input
            if df.empty:
                st.error("Uploaded file is empty. Please provide valid data.")
                logger.warning("Empty file uploaded.")
                return
                
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            if st.button("Predict Attrition"):
                with st.spinner("Processing data and generating predictions..."):
                    # Process and predict
                    result_df = process_and_predict(df, model, target_encoder)
                    
                    st.success("Predictions generated successfully!")
                    
                    st.subheader("Prediction Results")
                    st.dataframe(result_df)
                    
                    # Optional: Provide download link for results
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='attrition_predictions.csv',
                        mime='text/csv',
                    )
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty or corrupted.")
            logger.error("EmptyDataError encountered.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            logger.error(f"File processing error: {e}")
    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
