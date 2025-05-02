"""
Core classifier implementation for TB Chest X-ray Classification System.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from src.data_handler import DataHandler
from src.model_builder import ModelBuilder
from src.visualization import Visualizer

class TBChestXrayClassifier:
    """Core classifier class for TB Chest X-ray Classification."""
    
    def __init__(self):
        """Initialize the TB Chest X-ray Classifier."""
        self.model = None
        self.current_model_name = None
        self.class_names = ['Normal', 'Tuberculosis']
        self.input_shape = (224, 224, 3)  # Standard input shape for many models
        
        # Initialize components
        self.data_handler = DataHandler(self.input_shape)
        self.model_builder = ModelBuilder(self.input_shape)
        self.visualizer = Visualizer()
    
    def load_and_prepare_data(self, data_dir, test_size=0.2, validation_split=0.1):
        """
        Load and prepare data using the data handler.
        
        Args:
            data_dir (str): Path to data directory
            test_size (float): Proportion of data for testing
            validation_split (float): Proportion of training data for validation
            
        Returns:
            tuple: train_data, val_data, test_data
        """
        train_data, val_data, test_data = self.data_handler.load_and_prepare_data(
            data_dir, test_size, validation_split)
        
        # Update class names from data handler
        self.class_names = self.data_handler.get_class_names()
        
        return train_data, val_data, test_data
    
    def build_model(self, model_name='DenseNet121', learning_rate=0.0001):
        """
        Build a model for TB classification.
        
        Args:
            model_name (str): Name of the base model to use
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        self.model = self.model_builder.build_model(model_name, learning_rate)
        self.current_model_name = model_name
        return self.model
    
    def train_model(self, train_data, val_data, epochs=20, unfreeze_layers=0):
        """
        Train the model on the provided data.
        
        Args:
            train_data: Training data generator
            val_data: Validation data generator
            epochs (int): Number of epochs to train
            unfreeze_layers (int): Number of layers to unfreeze for fine-tuning
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        # Set up callbacks
        checkpoint = ModelCheckpoint(
            f'models/best_{self.current_model_name}_model.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks = [checkpoint, early_stop, reduce_lr]
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning if specified
        if unfreeze_layers > 0:
            print(f"Fine-tuning last {unfreeze_layers} layers...")
            
            # Unfreeze the last n layers
            self.model = self.model_builder.unfreeze_layers(self.model, unfreeze_layers)
            
            # Continue training
            fine_tune_history = self.model.fit(
                train_data,
                epochs=10,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            for key in history.history:
                history.history[key].extend(fine_tune_history.history[key])
        
        return history.history
    
    def evaluate_model(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data generator
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or trained. Call build_model and train_model first.")
        
        print("Evaluating model...")
        evaluation = self.model.evaluate(test_data, verbose=1)
        
        # Get predictions
        y_pred_prob = self.model.predict(test_data)
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_true = test_data.classes
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Create evaluation results dictionary
        results = {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'auc': evaluation[2],
            'precision': evaluation[3],
            'recall': evaluation[4],
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'roc': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        }
        
        return results
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            str: Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the model file
            
        Returns:
            tensorflow.keras.Model: Loaded model
        """
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        # Extract model name from filepath if possible
        model_name = os.path.basename(filepath).split('_')[1] if '_' in os.path.basename(filepath) else "Unknown"
        self.current_model_name = model_name
        return self.model
    
    def predict(self, image_path):
        """
        Predict TB probability for a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")
        
        # Load and preprocess image
        img = self.data_handler.preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(img)
        probability = float(prediction[0][0])
        predicted_class = 'Tuberculosis' if probability >= 0.5 else 'Normal'
        
        # Generate Grad-CAM heatmap
        heatmap = self.generate_gradcam(img[0])
        
        return {
            'probability': probability,
            'predicted_class': predicted_class,
            'heatmap': heatmap
        }
    
    def generate_gradcam(self, preprocessed_img):
        """
        Generate Grad-CAM heatmap for model interpretability.
        
        Args:
            preprocessed_img (numpy.ndarray): Single preprocessed image
            
        Returns:
            numpy.ndarray: Heatmap as numpy array
        """
        # Get the last convolutional layer
        last_conv_layer = self.model_builder.get_last_conv_layer(self.model, self.current_model_name)
        if last_conv_layer is None:
            return np.zeros(self.input_shape[:2])
        
        # Create a model that maps the input image to the activations of the last conv layer
        last_conv_layer_model = Model(inputs=self.model.inputs, 
                                   outputs=last_conv_layer.output)
        
        # Create a model that maps the activations of the last conv layer to the final class predictions
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer in self.model.layers[1:]:
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