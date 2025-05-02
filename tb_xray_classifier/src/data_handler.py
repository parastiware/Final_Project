"""
Data loading and preprocessing module for TB Chest X-ray Classification System.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class DataHandler:
    """Class for handling data loading and preprocessing."""
    
    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize the DataHandler.
        
        Args:
            input_shape (tuple): Input shape for the model (height, width, channels)
        """
        self.input_shape = input_shape
        self.class_names = []
        
    def load_and_prepare_data(self, data_dir, test_size=0.2, validation_split=0.1):
        """
        Load images from directory structure and prepare for training.
        
        Args:
            data_dir (str): Path to data directory with subdirectories for each class
            test_size (float): Proportion of data to use for testing
            validation_split (float): Proportion of training data to use for validation
        
        Returns:
            tuple: train_data, val_data, test_data (data generators for each split)
        """
        print(f"Loading data from {data_dir}...")
        
        # Create data generators with augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training and validation data
        train_data = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=32,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        val_data = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=32,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        # Get file paths for test data (assuming a test directory exists)
        test_dir = os.path.join(os.path.dirname(data_dir), 'test')
        if os.path.exists(test_dir):
            test_data = test_datagen.flow_from_directory(
                test_dir,
                target_size=self.input_shape[:2],
                batch_size=32,
                class_mode='binary',
                shuffle=False
            )
        else:
            print("No separate test directory found. Using validation data for testing.")
            test_data = val_data
        
        self.class_names = list(train_data.class_indices.keys())
        print(f"Classes: {self.class_names}")
        print(f"Training samples: {train_data.samples}")
        print(f"Validation samples: {val_data.samples}")
        print(f"Test samples: {test_data.samples}")
        
        return train_data, val_data, test_data
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image as numpy array
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to input shape
        img = cv2.resize(img, self.input_shape[:2])
        
        # Normalize pixel values
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def get_class_names(self):
        """
        Get class names.
        
        Returns:
            list: List of class names
        """
        return self.class_names
    
    def apply_data_augmentation(self, image):
        """
        Apply data augmentation to a single image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Augmented image
        """
        # Create an image data generator with augmentation settings
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Reshape image to (1, height, width, channels)
        img = image.reshape((1,) + image.shape)
        
        # Get the first augmented image
        for augmented_img in datagen.flow(img, batch_size=1):
            break
        
        return augmented_img[0]