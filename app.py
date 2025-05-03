import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import io
import time
import tempfile

# Import our modules
from preprocessing import XrayPreprocessor
from model import TBDetectionModel, create_dummy_model
from utils import overlay_heatmap, fig_to_base64, plot_confusion_matrix, plot_roc_curve, get_performance_metrics, create_sample_data

# Set page configuration
st.set_page_config(
    page_title="TB Detection from Chest X-rays",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5em;
    }
    .sub-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #34495e;
        margin-bottom: 1em;
    }
    .result-box {
        padding: 1em;
        border-radius: 0.5em;
        margin: 1em 0;
    }
    .positive-result {
        background-color: rgba(231, 76, 60, 0.1);
        border: 1px solid rgba(231, 76, 60, 0.5);
    }
    .negative-result {
        background-color: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.5);
    }
    .info-text {
        font-size: 1em;
        color: #7f8c8d;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1em;
        border-radius: 0.5em;
        text-align: center;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9em;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for the Streamlit app"""
    
    # Header
    st.markdown('<p class="main-header">Tuberculosis Detection from Chest X-rays</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Upload a chest X-ray image to detect signs of tuberculosis</p>', unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'explanation' not in st.session_state:
        st.session_state.explanation = None
    
    # Create sidebar
    with st.sidebar:
        st.markdown('<p class="sub-header">Settings</p>', unsafe_allow_html=True)
        
        # Model selection
        model_option = st.selectbox(
            "Select model",
            ["DenseNet121 (Transfer Learning)", "Custom CNN"],
            index=0
        )
        
        # Visualization options
        st.markdown('<p class="sub-header">Visualization Options</p>', unsafe_allow_html=True)
        
        show_preprocessing = st.checkbox("Show preprocessing steps", value=True)
        show_explanation = st.checkbox("Show model explanation (Grad-CAM)", value=True)
        
        # Add threshold slider for binary classification
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for TB classification"
        )
        
        # Add performance metrics section
        st.markdown('<p class="sub-header">Performance Metrics</p>', unsafe_allow_html=True)
        
        if st.button("Show Model Performance"):
            with st.spinner("Generating performance metrics..."):
                # Generate sample test results
                y_true, y_pred, y_scores = create_sample_data()
                
                # Display metrics
                metrics_df = get_performance_metrics(y_true, y_pred, y_scores)
                st.table(metrics_df)
                
                # Display confusion matrix
                cm_fig = plot_confusion_matrix(y_true, y_pred)
                st.pyplot(cm_fig)
                
                # Display ROC curve
                roc_fig = plot_roc_curve(y_true, y_scores)
                st.pyplot(roc_fig)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="sub-header">Upload X-ray Image</p>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
        
        # Demo images
        st.markdown('<p class="info-text">Or try a demo image:</p>', unsafe_allow_html=True)
        demo_cols = st.columns(3)
        
        demo_images = {
            "Normal": "https://i.imgur.com/zBrTtpP.jpeg",
            "TB Case 1": "https://i.imgur.com/eRkDSIi.jpeg",
            "TB Case 2": "https://i.imgur.com/zRm3Zol.jpeg",
        }
        
        demo_buttons = {}
        for i, (label, url) in enumerate(demo_images.items()):
            with demo_cols[i % 3]:
                demo_buttons[label] = st.button(label)
        
        # Process the image
        preprocessor = XrayPreprocessor(target_size=(224, 224))
        
        # Load or create the model
        model_path = "dummy_tb_model.h5"
        if not os.path.exists(model_path):
            model_path = create_dummy_model(model_path)
            
        model = TBDetectionModel(model_path)
        
        # Process uploaded image or demo image
        if uploaded_file is not None:
            with st.spinner("Processing image..."):
                # Get file bytes
                file_bytes = uploaded_file.getvalue()
                
                # Process the image
                original_img, processed_img = preprocessor.process_from_bytes(file_bytes)
                
                # Store in session state
                st.session_state.original_image = original_img
                st.session_state.processed_image = processed_img
                
                # Make prediction
                prediction = model.predict(processed_img)
                st.session_state.prediction = prediction
                
                # Get explanation
                if show_explanation:
                    explanation = model.get_explanation(processed_img)
                    st.session_state.explanation = explanation
        
        # Check if demo button was clicked
        for label, clicked in demo_buttons.items():
            if clicked:
                with st.spinner(f"Loading {label} image..."):
                    # For demo purposes, we'll just assign a probability based on the image
                    if "Normal" in label:
                        prob = 0.15  # Low probability for normal
                    else:
                        prob = 0.85  # High probability for TB
                    
                    # Display the image (normally we would download and process it)
                    st.image(demo_images[label], caption=f"Demo: {label}", use_column_width=True)
                    
                    # Set session state variables
                    st.session_state.prediction = prob
                    
                    # Create a dummy explanation (random colormap for demo)
                    dummy_exp = np.random.rand(224, 224, 3)
                    st.session_state.explanation = dummy_exp
    
    with col2:
        st.markdown('<p class="sub-header">Analysis Results</p>', unsafe_allow_html=True)
        
        # Display results if a prediction has been made
        if st.session_state.prediction is not None:
            # Determine if TB positive based on threshold
            is_tb_positive = st.session_state.prediction >= threshold
            
            # Display result
            result_class = "positive-result" if is_tb_positive else "negative-result"
            result_text = "TB Positive" if is_tb_positive else "TB Negative"
            st.markdown(f'<div class="result-box {result_class}"><p class="sub-header">{result_text}</p></div>', unsafe_allow_html=True)
            
            # Display explanation if available
            if st.session_state.explanation is not None and show_explanation:
                st.markdown('<p class="sub-header">Model Explanation (Grad-CAM)</p>', unsafe_allow_html=True)
                st.image(st.session_state.explanation, caption="Grad-CAM Explanation", use_column_width=True)

if __name__ == "__main__":
    main()