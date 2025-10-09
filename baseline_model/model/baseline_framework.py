############################################
# PyTorch Implementation of EmoPy ConvolutionalNN
# Baseline Model Wrapper
#
# by Gregor Autischer (August 2025)
############################################

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
import io
from sklearn.metrics import confusion_matrix

# Add parent directories to path for imports  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import model class and trainer from convolutional_nn_pytorch
from develop_baseline_model.models.convolutional_nn_pytorch import EmotionRecognitionTrainer

class BaselineModel:
    
    def __init__(self, model_path: str = 'model.pth'):
        self.emotion_labels = ['anger', 'fear', 'calm', 'surprise']
        self.model = None
        self.is_loaded = False
        
        # Load model silently
        self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        try:
            if not os.path.exists(model_path):
                return False
            
            # Load model silently
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.model, _ = EmotionRecognitionTrainer.load_model(model_path)
            
            self.model.eval()
            self.is_loaded = True
            return True
            
        except Exception:
            self.is_loaded = False
            return False
    
    def _preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        try:
            if not os.path.exists(image_path):
                return None
            
            img = Image.open(image_path).convert('L')
            img = img.resize((48, 48))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
            
            return img_tensor
            
        except Exception:
            return None
    
    def predict_image(self, image_path: str) -> Optional[Dict]:
        # Returns: dict: {'emotion': str, 'confidence': float, 'probabilities': dict}

        if not self.is_loaded:
            return None
        
        img_tensor = self._preprocess_image(image_path)
        if img_tensor is None:
            return None
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        return {
            'emotion': self.emotion_labels[predicted_class],
            'confidence': probabilities[0][predicted_class].item() * 100,
            'probabilities': {
                emotion: probabilities[0][i].item() * 100 
                for i, emotion in enumerate(self.emotion_labels)
            }
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Optional[Dict]]:
        return [self.predict_image(path) for path in image_paths]
    
    def _extract_label_from_path(self, image_path: str) -> Optional[str]:
        """Extract emotion label from image file path."""
        path_parts = image_path.replace('\\', '/').split('/')
        
        # Look for emotion folder in path
        for part in path_parts:
            if part.lower() in [label.lower() for label in self.emotion_labels]:
                return part.lower()
        
        return None
    
    def evaluate_batch(self, image_paths: List[str]) -> Optional[Dict]:
        """
        Evaluate a batch of images and return confusion matrix data and accuracy.
        
        Args:
            image_paths: List of image paths where emotion can be extracted from path
            
        Returns:
            dict: {
                'accuracy': float,
                'confusion_matrix': numpy.ndarray,
                'true_labels': list,
                'predicted_labels': list,
                'emotion_labels': list
            }
        """
        if not self.is_loaded:
            return None
        
        predictions = []
        true_labels = []
        predicted_labels = []
        
        for image_path in image_paths:
            # Extract true label from path
            true_emotion = self._extract_label_from_path(image_path)
            if true_emotion is None:
                continue  # Skip if can't extract label
            
            # Get prediction
            result = self.predict_image(image_path)
            if result is None:
                continue  # Skip if prediction failed
            
            true_label_idx = self.emotion_labels.index(true_emotion)
            pred_label_idx = self.emotion_labels.index(result['emotion'])
            
            true_labels.append(true_label_idx)
            predicted_labels.append(pred_label_idx)
        
        if not true_labels:
            return None  # No valid predictions
        
        # Calculate metrics
        accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels) * 100
        cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(self.emotion_labels)))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'emotion_labels': self.emotion_labels
        }
    
    def entropy_batch(self, image_paths: List[str]) -> Optional[List[float]]:
        """
        Calculate entropy for a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            list: List of entropy values for each image (None for failed predictions)
        """
        if not self.is_loaded:
            return None
        
        entropies = []
        
        for image_path in image_paths:
            img_tensor = self._preprocess_image(image_path)
            if img_tensor is None:
                entropies.append(None)
                continue
            
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                # Calculate entropy: -sum(p * log(p))
                # Add small epsilon to avoid log(0)
                eps = 1e-8
                log_probs = torch.log(probabilities + eps)
                entropy = -torch.sum(probabilities * log_probs, dim=1).item()
                
                entropies.append(entropy)
        
        return entropies