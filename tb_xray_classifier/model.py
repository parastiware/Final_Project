import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tf_explain.core.grad_cam import GradCAM
import matplotlib.pyplot as plt
import os

class TBDetectionModel:
    """Class for TB detection model creation, training and inference"""
    
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
        """
        Build the model architecture using transfer learning with DenseNet121
        """
        # Base model (DenseNet121)
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Compile the model
        self.model.compile(
            optimizer=Adam(0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
    def fine_tune(self, unfreeze_layers=30):
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            unfreeze_layers (int): Number of layers to unfreeze from the end
        """
        # Unfreeze the last layers
        trainable_base_layers = self.model.layers[0].layers[-unfreeze_layers:]
        for layer in trainable_base_layers:
            layer.trainable = True
            
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=Adam(0.00001),  # Lower learning rate
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
    def train(self, train_data, val_data, epochs=10, callbacks=None, fine_tune_after=5):
        """
        Train the model
        
        Args:
            train_data: Training data generator
            val_data: Validation data generator
            epochs (int): Number of epochs
            callbacks (list): List of callbacks
            fine_tune_after (int): Epoch after which to fine-tune
            
        Returns:
            History object
        """
        # Initial training with frozen base layers
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=fine_tune_after,
            callbacks=callbacks
        )
        
        # Fine-tuning
        print("Fine-tuning the model...")
        self.fine_tune()
        
        # Continue training with unfrozen layers
        history_fine_tune = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs - fine_tune_after,
            initial_epoch=fine_tune_after,
            callbacks=callbacks
        )
        
        # Combine histories
        for k in history.history:
            history.history[k].extend(history_fine_tune.history[k])
            
        return history
    
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
    
    def save(self, model_path):
        """
        Save the model
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build() or load() first.")
            
        self.model.save(model_path)
        
    def load(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = load_model(model_path)
        
    def get_explanation(self, image):
        """
        Generate Grad-CAM visualization to explain the model's decision
        
        Args:
            image (numpy.ndarray): Preprocessed image
            
        Returns:
            numpy.ndarray: Heatmap overlay on original image
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build() or load() first.")
            
        # Create Grad-CAM explainer
        explainer = GradCAM()
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Get model's convolutional layer to explain
        conv_layer = self.model.get_layer('densenet121').get_layer('conv5_block16_concat')
        
        # Generate heatmap
        grid = explainer.explain((image, None), self.model, conv_layer.name, 0)
        
        return grid
        
    def plot_training_history(self, history):
        """
        Plot training history
        
        Args:
            history: History object from model.fit()
            
        Returns:
            matplotlib.figure.Figure: Figure with accuracy and loss plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper left')
        
        # Loss plot
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper left')
        
        fig.tight_layout()
        return fig


# Helper function to create a dummy model for demo purposes
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
    def dummy_predict(self, image):
        return 0.7
        
    # Monkey patch the predict method
    model.predict = lambda image: 0.7
    model.get_explanation = lambda image: np.random.rand(224, 224, 3)
    
    # Save the model's architecture only (weights will be random)
    model.save(save_path)
    
    return save_path