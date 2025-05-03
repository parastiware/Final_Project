import numpy as np
import tensorflow as tf  # type: ignore
from tf_explain.core.grad_cam import GradCAM
import os

class TBDetectionModel:
    """Class for TB detection model creation and inference"""
    
    def __init__(self, model_path=None):
        """
        Initialize the TB detection model
        
        Args:
            model_path (str, optional): Path to saved model for loading
        """
        self.input_shape = (224, 224, 3)
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self.build()
    
    def build(self):
        """Build the model architecture using transfer learning with DenseNet121"""
        # Base model (DenseNet121)
        base_model = tf.keras.applications.DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Add custom layers
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        # Create the model
        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def predict(self, image):
        """
        Make a prediction for a preprocessed image
        
        Args:
            image (numpy.ndarray): Preprocessed image
            
        Returns:
            float: Probability of TB
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build() or load() first.")
            
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Make prediction
        prediction = self.model.predict(image)
        
        return float(prediction[0][0])
    
    def load(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(model_path)
        
    def get_explanation(self, image):
        """
        Generate Grad-CAM visualization to explain the model's decision
        
        Args:
            image (numpy.ndarray): Preprocessed image
            
        Returns:
            numpy.ndarray: Grad-CAM visualization or a fallback image
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build() or load() first.")

        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        try:
            # Initialize Grad-CAM explainer
            explainer = GradCAM()

            # Find the last convolutional layer in the model
            conv_layers = []
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    conv_layers.append(layer)
                # If it's a functional model, check its layers
                elif hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if isinstance(sublayer, tf.keras.layers.Conv2D):
                            conv_layers.append(sublayer)
            
            if not conv_layers:
                # If no conv layers found, return a grayscale version of the input
                return np.mean(image[0], axis=-1, keepdims=True)
            
            # Use the last convolutional layer
            conv_layer = conv_layers[-1]
            
            # Generate Grad-CAM visualization
            grid = explainer.explain((image, None), self.model, conv_layer.name, 0)
            
            if grid is None or grid.size == 0:
                # If Grad-CAM fails, return a grayscale version of the input
                return np.mean(image[0], axis=-1, keepdims=True)
                
            return grid
            
        except Exception as e:
            # If any error occurs, return a grayscale version of the input
            return np.mean(image[0], axis=-1, keepdims=True)

def create_dummy_model(save_path="dummy_tb_model.h5"):
    """
    Create a dummy pre-trained model for demonstration purposes
    
    Args:
        save_path (str): Path to save the dummy model
        
    Returns:
        str: Path to the saved model
    """
    model = TBDetectionModel()
    
    # Create a simple model that always predicts TB with ~70% probability
    model.predict = lambda image: 0.7
    model.get_explanation = lambda image: np.random.rand(224, 224, 3)
    
    # Save the model's architecture only (weights will be random)
    model.save(save_path)
    
    return save_path
