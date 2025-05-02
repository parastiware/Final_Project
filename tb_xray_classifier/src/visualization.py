"""
Visualization functions for model training and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model


def plot_training_history(history):
    """
    Plot the training history.
    
    Args:
        history: Training history object
        
    Returns:
        Figure with training history plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='train')
    axes[0, 0].plot(history.history['val_accuracy'], label='validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='train')
    axes[0, 1].plot(history.history['val_loss'], label='validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    
    # Plot AUC
    axes[1, 0].plot(history.history['auc'], label='train')
    axes[1, 0].plot(history.history['val_auc'], label='validation')
    axes[1, 0].set_title('Model AUC')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    
    # Plot Precision & Recall
    axes[1, 1].plot(history.history['precision'], label='precision')
    axes[1, 1].plot(history.history['recall'], label='recall')
    axes[1, 1].plot(history.history['val_precision'], label='val_precision')
    axes[1, 1].plot(history.history['val_recall'], label='val_recall')
    axes[1, 1].set_title('Precision and Recall')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    
    plt.tight_layout()
    return fig


def plot_evaluation_results(results, class_names):
    """
    Plot evaluation results including confusion matrix and ROC curve.
    
    Args:
        results: Evaluation results dictionary
        class_names: List of class names
        
    Returns:
        Figure with evaluation plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot confusion matrix
    sns.heatmap(
        results['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Plot ROC curve
    axes[1].plot(
        results['roc']['fpr'], 
        results['roc']['tpr'], 
        label=f'AUC = {results["roc"]["auc"]:.3f}'
    )
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_title('ROC Curve')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def plot_gradcam_overlay(image_path, heatmap, input_shape):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image_path: Path to the original image
        heatmap: Grad-CAM heatmap
        input_shape: Shape of input images (height, width, channels)
        
    Returns:
        Figure with original image and heatmap overlay
    """
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape[:2])
    
    # Create heatmap overlay
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on original image
    alpha = 0.4
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot overlay
    axes[1].imshow(superimposed_img)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def generate_gradcam(model, preprocessed_img, model_name, layer_name=None):
    """
    Generate Grad-CAM heatmap for model interpretability.
    
    Args:
        model: Trained Keras model
        preprocessed_img: Single preprocessed image (no batch dimension)
        model_name: Name of the model architecture
        layer_name: Optional name of the layer to use for Grad-CAM
        
    Returns:
        Heatmap as numpy array
    """
    # For different models, we need to get the last convolutional layer
    if layer_name is None:
        if model_name == 'DenseNet121':
            last_conv_layer = model.get_layer('densenet121').get_layer('conv5_block16_concat')
        elif model_name == 'ResNet50V2':
            last_conv_layer = model.get_layer('resnet50v2').get_layer('conv5_block3_out')
        elif model_name == 'EfficientNetB3':
            last_conv_layer = model.get_layer('efficientnetb3').get_layer('top_activation')
        else:
            # If model name is unknown, try a common approach
            for layer in reversed(model.layers[0].layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
            else:
                # If no convolutional layer is found, return empty heatmap
                print("Could not find convolutional layer for Grad-CAM.")
                return np.zeros((224, 224))
    else:
        last_conv_layer = model.get_layer(layer_name)
    
    # Create a model that maps the input image to the activations of the last conv layer
    last_conv_layer_model = Model(inputs=model.inputs, 
                               outputs=last_conv_layer.output)
    
    # Create a model that maps the activations of the last conv layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]:
        x = layer(x)
    classifier_model = Model(inputs=classifier_input, outputs=x)
    
    # Watch the gradients
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(np.expand_dims(preprocessed_img, axis=0))
        tape.watch(last_conv_layer_output)
        
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    
    # Gradient of the top predicted class with regard to the output feature map
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    
    # Vector of mean intensity of the gradient over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradient values
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    # Average all channels for the heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    
    # ReLU & normalization
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    heatmap = cv2.resize(heatmap, (224, 224))
    
    return heatmap