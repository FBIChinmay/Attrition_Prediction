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
    body, .main, .stApp {
        background: #eef2f7;
        color: #1f2937;
        font-family: 'Inter', sans-serif;
    }
    .app-header {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        color: white;
        padding: 32px;
        border-radius: 20px;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px rgba(33, 33, 33, 0.12);
    }
    .app-header h1 {
        margin: 0;
        font-size: 2.8rem;
        letter-spacing: -0.03em;
    }
    .app-header p {
        margin: 8px 0 0;
        font-size: 1.05rem;
        opacity: 0.88;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 16px;
        margin-bottom: 32px;
    }
    .feature-card,
    .empty-state,
    .result-banner {
        background: white;
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0 18px 32px rgba(15, 23, 42, 0.08);
    }
    .feature-card h3,
    .empty-state h3,
    .result-banner h3 {
        margin-top: 0;
        color: #111827;
    }
    .feature-card p,
    .empty-state p,
    .result-banner p {
        margin-bottom: 0;
        color: #4b5563;
        line-height: 1.7;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        transition: transform 0.2s ease, background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
    }
    div[data-testid="stSidebar"], .css-1d391kg, .css-18e3th9 {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%) !important;
        color: white !important;
    }
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
    .stFileUploader>div {
        border-radius: 16px !important;
    }
    .stDownloadButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
    }
    .stDownloadButton>button:hover {
        background-color: #0f766e;
    }
    .metric-tile {
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 20px 24px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }
    .metric-title {
        display: block;
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2.4rem;
        color: #111827;
        font-weight: 700;
    }
    .stDataFrame div {
        border-radius: 16px;
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
    st.markdown("""
        <div class="app-header">
            <h1>👥 HR Attrition Prediction System</h1>
            <p>Upload employee data to predict attrition using our trained machine learning pipeline. Fast, reliable predictions for smarter workforce planning.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="feature-grid">
            <div class="feature-card">
                <h3>Upload your CSV</h3>
                <p>Select a file containing employee records and let the system analyze attrition risk.</p>
            </div>
            <div class="feature-card">
                <h3>Run prediction</h3>
                <p>Our trained model processes your data and predicts which employees may leave.</p>
            </div>
            <div class="feature-card">
                <h3>Download results</h3>
                <p>Export predictions instantly and use them to guide retention initiatives.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    model, target_encoder = load_artifacts()
    
    if model is None:
        st.stop()
        
    st.sidebar.header("Data Upload")
    st.sidebar.markdown("Upload a CSV file with employee details such as Age, Department, and Job Role.")
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
                    
                    # Summary metrics for the prediction result
                    yes_count = int((result_df['Predicted_Attrition'] == 'Yes').sum())
                    no_count = int((result_df['Predicted_Attrition'] == 'No').sum())

                    st.markdown("""
                        <div class="result-banner">
                            <h3>Prediction Summary</h3>
                            <p>Review the attrition risk summary and download your results for follow-up action.</p>
                        </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    col1.markdown(f"""
                        <div class="metric-tile">
                            <span class="metric-title">Predicted Attrition</span>
                            <span class="metric-value">{yes_count}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    col2.markdown(f"""
                        <div class="metric-tile">
                            <span class="metric-title">Predicted Stay</span>
                            <span class="metric-value">{no_count}</span>
                        </div>
                    """, unsafe_allow_html=True)

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
        st.markdown("""
            <div class="empty-state">
                <h3>Ready when you are</h3>
                <p>Upload a CSV file from the sidebar to start predicting employee attrition. Supported format: <strong>CSV</strong>.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
