# TB Chest X-ray Detection System

## Overview
This project is a comprehensive tuberculosis (TB) detection system built using deep learning and transfer learning techniques. It provides an intuitive user interface for analyzing chest X-ray images, training models, and visualizing results. The system is designed to be user-friendly and educational, with features like Grad-CAM visualizations and preprocessing step explanations.

---

## Key Features

### 1. User Interface
- Clean, modern interface built with Streamlit.
- Sidebar for settings and navigation.
- Upload individual X-rays or use demo images.
- Batch processing capability for multiple X-rays.
- Model training interface with customizable parameters.

### 2. Analysis Features
- TB detection with probability scores.
- Visualization of model decisions using Grad-CAM.
- Preprocessing pipeline visualization.
- Performance metrics display.

### 3. Technical Implementation
- **Transfer Learning**: DenseNet121 pre-trained on ImageNet.
- **Image Preprocessing**:
  - CLAHE enhancement for better contrast.
  - Lung segmentation.
  - Resizing and normalization.
- **Model Explanation**: Grad-CAM for visualizing model decisions.

### 4. Educational Value
- Visual explanations of model decisions.
- Preprocessing step visualization.
- Performance metrics with detailed explanations.

---

## Project Structure

```
tb_xray_classifier/
├── app.py                # Main Streamlit application
├── src/
│   ├── preprocessing.py  # Image preprocessing functions
│   ├── model.py          # TB detection model implementation
│   ├── utils.py          # Utility functions for visualization and evaluation
├── data/                 # Directory for training and testing data
├── models/               # Directory for saving trained models
├── requirements.txt      # Dependencies needed to run the application
├── README.md             # Project documentation
└── description.txt       # Detailed project description
```

---

## How to Run the Project

### 1. Install Dependencies
Make sure you have Python 3.7 or higher installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Application
Launch the Streamlit application:
```bash
streamlit run app.py
```

### 3. Open in Browser
After running the above command, open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`).

---

## Using the Application

### Single Image Analysis
1. Upload a chest X-ray image or use one of the demo images.
2. View the TB detection result with a probability score.
3. Visualize the model's decision using Grad-CAM.

### Batch Analysis
1. Upload multiple chest X-ray images.
2. Run batch analysis to get results for all images at once.

### Model Training
1. Upload a dataset (organized into `train/` and `test/` directories).
2. Set training parameters (e.g., epochs, batch size).
3. Monitor training progress with metrics and plots.

---

## Technical Details

### Transfer Learning
The project uses DenseNet121 pre-trained on ImageNet for feature extraction and fine-tuning.

### Image Preprocessing
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization for better contrast.
- **Lung Segmentation**: Focuses on the lung region for better predictions.
- **Resizing and Normalization**: Images are resized to 224x224 and pixel values are normalized to [0, 1].

### Grad-CAM
Grad-CAM is used to visualize the regions of the X-ray image that the model focuses on when making predictions.

---

## Dependencies
The project requires the following Python packages:
- `streamlit`
- `tensorflow`
- `opencv-python`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `Pillow`

Install them using:
```bash
pip install -r requirements.txt
```

---

## Educational Value
This project is designed to be educational, providing:
- Visual explanations of model decisions using Grad-CAM.
- Insights into preprocessing steps like CLAHE and lung segmentation.
- Performance metrics with detailed explanations.

---

## Acknowledgments
This project is inspired by research in medical imaging and deep learning. Special thanks to the open-source community for providing tools and resources.

---
