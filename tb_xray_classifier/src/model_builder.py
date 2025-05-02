"""
Model builder module for TB Chest X-ray Classification System.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import DenseNet121, ResNet50V2, EfficientNetB3
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    """Class for building and compiling models."""
    
    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize the ModelBuilder.
        
        Args:
            input_shape (tuple): Input shape for the model (height, width, channels)
        """
        self.input_shape = input_shape
        self.base_models = {
            'DenseNet121': DenseNet121,
            'ResNet50V2': ResNet50V2,
            'EfficientNetB3': EfficientNetB3
        }
    
    def build_model(self, model_name='DenseNet121', learning_rate=0.0001):
        """
        Build a model for TB classification using transfer learning.
        
        Args:
            model_name (str): Name of the base model to use
            learning_rate (float): Learning rate for the optimizer
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        print(f"Building model with {model_name} as base...")
        
        if model_name not in self.base_models:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(self.base_models.keys())}")
        
        # Load pre-trained model with weights
        base_model = self.base_models[model_name](
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Create new model on top
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), 
                     tf.keras.metrics.Precision(), 
                     tf.keras.metrics.Recall()]
        )
        
        return model
    
    def unfreeze_layers(self, model, num_layers, learning_rate=1e-5):
        """
        Unfreeze the last n layers of the base model for fine-tuning.
        
        Args:
            model (tensorflow.keras.Model): Model to fine-tune
            num_layers (int): Number of layers to unfreeze
            learning_rate (float): Learning rate for the optimizer
            
        Returns:
            tensorflow.keras.Model: Model with unfrozen layers
        """
        if num_layers <= 0:
            return model
        
        # Unfreeze the last n layers
        base_model = model.layers[0]
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), 
                     tf.keras.metrics.Precision(), 
                     tf.keras.metrics.Recall()]
        )
        
        return model
    
    def get_model_summary(self, model):
        """
        Get model summary as a string.
        
        Args:
            model (tensorflow.keras.Model): Model to summarize
            
        Returns:
            str: Model summary
        """
        # Redirect stdout to capture summary
        import io
        import sys
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        # Print model summary
        model.summary()
        
        # Get output and restore stdout
        summary = new_stdout.getvalue()
        sys.stdout = old_stdout
        
        return summary
    
    def get_last_conv_layer(self, model, model_name):
        """
        Get the last convolutional layer of the model for Grad-CAM.
        
        Args:
            model (tensorflow.keras.Model): Model
            model_name (str): Name of the base model
            
        Returns:
            tensorflow.keras.layers.Layer: Last convolutional layer
        """
        # For DenseNet and similar models, we need to get the last convolutional layer
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
                # If no convolutional layer is found, return None
                print("Could not find convolutional layer for Grad-CAM.")
                return None
        
        return last_conv_layer