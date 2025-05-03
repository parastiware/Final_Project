import cv2
import numpy as np
from PIL import Image
import io

class XrayPreprocessor:
    """Class for preprocessing chest X-ray images for TB detection"""
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the preprocessor
        
        Args:
            target_size (tuple): Target size for resizing images (height, width)
        """
        self.target_size = target_size
        
    def load_image(self, image_path):
        """
        Load an image from path
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Original loaded image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        return image
    
    def preprocess_image(self, image):
        """
        Preprocess an image for model input
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image ready for model input
        """
        try:
            # Convert to grayscale if not already
            if len(image.shape) > 2 and image.shape[2] > 1:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Ensure the image is of type uint8
            gray = gray.astype('uint8')
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Resize
            resized = cv2.resize(enhanced, self.target_size)
            
            # Normalize to [0, 1]
            normalized = resized / 255.0
            
            # Convert to RGB (3 channels)
            rgb = np.stack([normalized] * 3, axis=-1)
            
            return rgb
            
        except Exception as e:
            raise ValueError(f"Error in preprocessing: {str(e)}")
    
    def process_from_path(self, image_path):
        """
        Load and preprocess an image from path
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (original_image, processed_image)
        """
        original = self.load_image(image_path)
        processed = self.preprocess_image(original)
        return original, processed
    
    def process_from_bytes(self, image_bytes):
        """
        Process an image from bytes
        
        Args:
            image_bytes (bytes): Image data in bytes
            
        Returns:
            tuple: (original_image, processed_image)
        """
        try:
            # First try with PIL
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert('RGB')  # Ensure RGB
            img = img.resize(self.target_size)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Store original for display
            original = img_array.copy()
            
            # Preprocess for model input
            processed = self.preprocess_image(img_array)
            
            return original, processed
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}. Please make sure the image is a valid chest X-ray.")
        
    def get_lung_segmentation(self, image):
        """
        Perform basic lung segmentation on the input image
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: Segmented lung image
        """
        # Convert to grayscale if not already
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Threshold the image to get a binary mask
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask
        mask = np.zeros_like(gray)
        
        # Draw contours of sufficient size (to avoid small artifacts)
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:  # Arbitrary threshold
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                
        # Apply the mask to the original image
        segmented = cv2.bitwise_and(gray, gray, mask=mask)
        
        return segmented