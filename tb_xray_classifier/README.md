# TB Chest X-ray Classification System

## Overview
This application provides a complete solution for training, evaluating, and making predictions using deep learning models on chest X-ray images to detect tuberculosis (TB). The system includes a user-friendly graphical interface for all operations.

## Features
- Data loading and preprocessing for chest X-ray images
- Transfer learning with multiple pre-trained models (DenseNet121, ResNet50V2, EfficientNetB3)
- Model training with customizable parameters
- Model evaluation with comprehensive metrics
- Visual explanations using Grad-CAM
- User-friendly interface for all operations

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone the repository:
```
git clone https://github.com/parastiware/tb-xray-classifier.git
cd tb-xray-classifier
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Running the Application
```
python main.py
```

### Data Directory Structure
The application expects data to be organized in the following structure:
```
data/
├── train/
│   ├── normal/
│   │   ├── normal_001.png
│   │   ├── normal_002.png
│   │   └── ...
│   └── tuberculosis/
│       ├── tb_001.png
│       ├── tb_002.png
│       └── ...
└── test/
    ├── normal/
    │   ├── normal_test_001.png
    │   ├── normal_test_002.png
    │   └── ...
    └── tuberculosis/
        ├── tb_test_001.png
        ├── tb_test_002.png
        └── ...
```

### Training a Model
1. Select the "Train Model" tab
2. Browse and select your data directory
3. Choose a base model (DenseNet121, ResNet50V2, or EfficientNetB3)
4. Set the number of epochs and unfreeze layers for fine-tuning
5. Click "Build Model" followed by "Train Model"
6. After training, save your model using the "Save Model" button

### Evaluating a Model
1. Select the "Evaluate Model" tab
2. Browse and select your test data directory
3. Click "Evaluate Model" to see comprehensive metrics

### Making Predictions
1. Select the "Make Predictions" tab
2. Load a trained model if not already loaded
3. Browse and select a chest X-ray image
4. Click "Predict" to see the results
5. View the Grad-CAM visualization to understand model focus areas

## Project Structure
- `main.py`: Main entry point
- `src/`: Source code directory
  - `classifier.py`: Core classifier implementation
  - `data_handler.py`: Data loading and preprocessing
  - `model_builder.py`: Model architecture and building
  - `utils.py`: Utility functions
  - `visualization.py`: Visualization functions
  - `ui/`: UI related code
- `models/`: Directory for saved models
- `data/`: Directory for data
- `tests/`: Tests directory

## References
- Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.
- Rahman, T., et al. (2020). Reliable Tuberculosis Detection Using Chest X-ray With Deep Learning, Segmentation and Visualization.
- Lakhani, P., & Sundaram, B. (2017). Deep Learning at Chest Radiography: Automated Classification of Pulmonary Tuberculosis by Using Convolutional Neural Networks.
- Selvaraju, R. R., et al. (2020). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.

## License
MIT License