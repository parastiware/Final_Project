import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.cm as cm

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    Overlay a heatmap on an original image
    
    Args:
        original_img (numpy.ndarray): Original image
        heatmap (numpy.ndarray): Heatmap image
        alpha (float): Transparency factor
        
    Returns:
        numpy.ndarray: Image with heatmap overlay
    """
    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert heatmap to RGB if it's grayscale
    if len(heatmap.shape) == 2:
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
    # Convert original image to RGB if it's grayscale
    if len(original_img.shape) == 2 or original_img.shape[2] == 1:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    # Blend images
    return cv2.addWeighted(original_img, 1-alpha, heatmap, alpha, 0)

def fig_to_base64(fig):
    """
    Convert matplotlib figure to base64 string for display in HTML
    
    Args:
        fig (matplotlib.figure.Figure): Figure to convert
        
    Returns:
        str: Base64 encoded string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'TB']):
    """
    Create a confusion matrix plot
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        classes (list): Class names
        
    Returns:
        matplotlib.figure.Figure: Figure with confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them with class names
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, 
           yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    return fig

def plot_roc_curve(y_true, y_scores):
    """
    Create a ROC curve plot
    
    Args:
        y_true (array-like): True labels
        y_scores (array-like): Predicted scores
        
    Returns:
        matplotlib.figure.Figure: Figure with ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    return fig

def get_performance_metrics(y_true, y_pred, y_scores=None):
    """
    Get performance metrics as a DataFrame
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        y_scores (array-like, optional): Predicted scores for ROC AUC
        
    Returns:
        pandas.DataFrame: Performance metrics
    """
    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extract metrics
    metrics = {
        'Accuracy': (np.sum(y_true == y_pred) / len(y_true)),
        'Precision': report['1']['precision'],
        'Recall (Sensitivity)': report['1']['recall'],
        'F1-Score': report['1']['f1-score'],
        'Specificity': report['0']['recall']
    }
    
    # Add AUC if scores are provided
    if y_scores is not None:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        metrics['AUC'] = auc(fpr, tpr)
        
    # Convert to DataFrame
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    return df

def create_sample_data():
    """
    Create sample test results for demonstration
    
    Returns:
        tuple: (y_true, y_pred, y_scores)
    """
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.clip(y_true + np.random.normal(0, 0.3, 100), 0, 1)
    y_pred = (y_scores > 0.5).astype(int)
    return y_true, y_pred, y_scores