############################################
# PyTorch Implementation of EmoPy ConvolutionalNN
# Faithful recreation of original Keras implementation
#
# by Gregor Autischer (August 2025)
############################################

import os
import sys
import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import csv
import random
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image

class ConvolutionalNN(nn.Module):
    """
    PyTorch implementation of the Keras ConvolutionalNN from EmoPy.
    """
    
    def __init__(self, image_size, channels, num_classes, filters=10, kernel_size=(4, 4), verbose=False):
        super(ConvolutionalNN, self).__init__()
        
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.verbose = verbose
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=filters, 
                              kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                              kernel_size=kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                              kernel_size=kernel_size)
        self.conv4 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                              kernel_size=kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        self._calculate_linear_input_size()
        
        # Fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.linear_input_size, num_classes)
        
        if self.verbose:
            self._print_model_summary()
    
    def _calculate_linear_input_size(self):
        """Calculate the input size for the linear layer based on conv operations"""
        # Simulate forward pass through conv layers to get size
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.channels, *self.image_size)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = self.pool1(x)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool2(x)
            x = x.view(x.size(0), -1)
            self.linear_input_size = x.size(1)
    
    def _print_model_summary(self):
        """Print model architecture summary similar to Keras"""
        print("Model Architecture:")
        print(f"Input shape: {self.channels} x {self.image_size[0]} x {self.image_size[1]}")
        print(f"Conv2d-1: {self.filters} filters, kernel_size={self.kernel_size}")
        print(f"Conv2d-2: {self.filters} filters, kernel_size={self.kernel_size}")
        print("MaxPool2d-1: 2x2")
        print(f"Conv2d-3: {self.filters} filters, kernel_size={self.kernel_size}")
        print(f"Conv2d-4: {self.filters} filters, kernel_size={self.kernel_size}")
        print("MaxPool2d-2: 2x2")
        print("Flatten")
        print(f"Linear: {self.linear_input_size} -> {self.num_classes}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
    
    def forward(self, x):
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Second convolutional block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Flatten and fully connected
        x = self.flatten(x)
        x = self.fc(x)
        x = F.relu(x)  # ReLU activation on final layer as specified
        
        return x

class CosineProximityLoss(nn.Module):
    """
    Cosine Proximity Loss function to match Keras implementation.
    Loss = -mean(cosine_similarity(y_true, y_pred))
    """
    def __init__(self):
        super(CosineProximityLoss, self).__init__()
        
    def forward(self, y_pred, y_true_indices):
        # Convert indices to one-hot encoding
        num_classes = y_pred.size(1)
        y_true = torch.zeros_like(y_pred)
        y_true.scatter_(1, y_true_indices.unsqueeze(1), 1)
        
        # Normalize vectors
        y_true_norm = F.normalize(y_true, p=2, dim=1)
        y_pred_norm = F.normalize(y_pred, p=2, dim=1)
        
        # Compute cosine similarity
        cosine_sim = (y_true_norm * y_pred_norm).sum(dim=1)
        
        # Return negative mean (to minimize loss)
        return -cosine_sim.mean()

class EmotionRecognitionTrainer:
    """
    Training class for the ConvolutionalNN with validation support.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = CosineProximityLoss()  # Using cosine proximity loss
        # ORIGINAL HAD OTHER VALUE: RMSprop with lr=0.001, changed to Adam with much smaller lr=0.0001 and weight_decay=1e-4 for better convergence
        self.optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
        # self.optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        
        # Learning rate scheduler - reduces LR when validation loss plateaus
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
    def train(self, train_loader, fer_val_loader=None, ckp_val_loader=None, combined_val_loader=None, epochs=50, verbose=True):
        """
        Train the model with separate FER and CKP validation, plus combined validation.
        """
        train_losses = []
        train_accuracies = []
        fer_val_losses = []
        fer_val_accuracies = []
        ckp_val_losses = []
        ckp_val_accuracies = []
        combined_val_losses = []
        combined_val_accuracies = []
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = 100. * train_correct / train_total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # FER Validation phase
            if fer_val_loader is not None:
                fer_val_loss, fer_val_accuracy = self._validate(fer_val_loader)
                fer_val_losses.append(fer_val_loss)
                fer_val_accuracies.append(fer_val_accuracy)
            else:
                fer_val_losses.append(0)
                fer_val_accuracies.append(0)
            
            # CKP Validation phase
            if ckp_val_loader is not None:
                ckp_val_loss, ckp_val_accuracy = self._validate(ckp_val_loader)
                ckp_val_losses.append(ckp_val_loss)
                ckp_val_accuracies.append(ckp_val_accuracy)
            else:
                ckp_val_losses.append(0)
                ckp_val_accuracies.append(0)
            
            # Combined Validation phase (FER + CKP)
            if combined_val_loader is not None:
                combined_val_loss, combined_val_accuracy = self._validate(combined_val_loader)
                combined_val_losses.append(combined_val_loss)
                combined_val_accuracies.append(combined_val_accuracy)
                
                # Update learning rate scheduler based on combined validation loss
                self.scheduler.step(combined_val_loss)
                
                # Early stopping check
                if combined_val_loss < best_val_loss:
                    best_val_loss = combined_val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f'\nEarly stopping triggered after {epoch+1} epochs')
                            print(f'Best validation loss: {best_val_loss:.4f}')
                        # Restore best model
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                        break
            else:
                combined_val_losses.append(0)
                combined_val_accuracies.append(0)
            
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | '
                      f'FER Val Acc: {fer_val_accuracies[-1]:.2f}% | '
                      f'CKP Val Acc: {ckp_val_accuracies[-1]:.2f}% | '
                      f'Combined Val Acc: {combined_val_accuracies[-1]:.2f}%')
                if combined_val_loader is not None:
                    print(f'  -> LR: {self.optimizer.param_groups[0]["lr"]:.6f}, '
                          f'Best Val Loss: {best_val_loss:.4f}, '
                          f'Patience: {patience_counter}/{patience}')
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'fer_val_losses': fer_val_losses,
            'fer_val_accuracies': fer_val_accuracies,
            'ckp_val_losses': ckp_val_losses,
            'ckp_val_accuracies': ckp_val_accuracies,
            'combined_val_losses': combined_val_losses,
            'combined_val_accuracies': combined_val_accuracies
        }
    
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        return val_loss, val_accuracy
    
    def predict(self, data_loader, return_targets=False):
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                predictions.extend(predicted.cpu().numpy())
                if return_targets:
                    targets.extend(target.cpu().numpy())
        
        if return_targets:
            return np.array(predictions), np.array(targets)
        return np.array(predictions)
    
    def get_confusion_matrix(self, data_loader):
        predictions, targets = self.predict(data_loader, return_targets=True)
        return confusion_matrix(targets, predictions)
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        Saves both the model state and training configuration.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': {
                'image_size': self.model.image_size,
                'channels': self.model.channels,
                'num_classes': self.model.num_classes,
                'filters': self.model.filters,
                'kernel_size': self.model.kernel_size
            }
        }, filepath)
        print(f"Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath, device='cpu'):
        """
        Load a trained model from a file.
        Returns the model and trainer ready for predictions.
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Recreate model with saved configuration
        config = checkpoint['model_config']
        model = ConvolutionalNN(
            image_size=config['image_size'],
            channels=config['channels'],
            num_classes=config['num_classes'],
            filters=config['filters'],
            kernel_size=config['kernel_size']
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create trainer and load optimizer/scheduler states
        trainer = EmotionRecognitionTrainer(model, device=device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded from: {filepath}")
        return model, trainer

def load_emotion_data(data_path, emotion_labels=['anger', 'fear', 'calm', 'surprise'], usage_factor=1.0, random_seed=42):
    """
    Load images directly from emotion dirs that are created by helper script
    
    Args:
        data_path: Path to dataset directory
        emotion_labels: List of emotion labels to load
        usage_factor: Fraction of data to use (0.0 to 1.0)
        random_seed: Random seed for reproducible sampling
    """
    # If usage_factor is 0, return empty tensors
    if usage_factor == 0:
        print(f" -> Skipping {data_path} (usage_factor=0)")
        empty_images = torch.FloatTensor(0, 1, 48, 48)
        empty_labels = torch.LongTensor(0)
        empty_paths = []
        emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
        return empty_images, empty_labels, emotion_to_idx, empty_paths
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Create emotion mapping
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
    
    images = []
    labels = []
    image_paths = []  # Track paths for each image
    
    print(f" -> Loading images from: {data_path}")
    
    for emotion in emotion_labels:
        emotion_dir = os.path.join(data_path, emotion)
        
        if not os.path.exists(emotion_dir):
            print(f"Warning: Emotion directory not found: {emotion_dir}")
            continue
            
        # Get all PNG files in emotion directory
        png_files = [f for f in os.listdir(emotion_dir) if f.lower().endswith('.png')]
        
        if not png_files:
            print(f"Warning: No PNG files found in {emotion_dir}")
            continue

        # Apply usage factor - randomly sample if less than 1.0
        if usage_factor < 1.0:
            num_to_use = max(1, int(len(png_files) * usage_factor))
            random.seed(random_seed + hash(emotion))  # Consistent sampling per emotion
            png_files = random.sample(png_files, num_to_use)
        
        print(f" └-> Loading {len(png_files)} images for {emotion} (usage_factor={usage_factor})...")

        for png_file in png_files:
            img_path = os.path.join(emotion_dir, png_file)
            
            try:
                # Load image as grayscale
                img = Image.open(img_path).convert('L')
                
                # Check image size - must be exactly 48x48
                if img.size != (48, 48):
                    raise ValueError(f" XX Image {img_path} has size {img.size}, expected (48, 48)")
                
                # Convert to numpy array and normalize to [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Add to lists
                images.append(img_array)
                labels.append(emotion_to_idx[emotion])
                image_paths.append(str(img_path))
                
            except Exception as e:
                print(f" XX Error loading {img_path}: {e}")
                raise
    
    if len(images) == 0:
        raise ValueError(f" XX No valid images found in {data_path}")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Add channel dimension: (N, H, W) -> (N, 1, H, W) 
    images = images[:, np.newaxis, :, :]
    
    # Convert to PyTorch tensors
    images_tensor = torch.FloatTensor(images)
    labels_tensor = torch.LongTensor(labels)
    
    print(f" -> Loaded {len(images)} images from {len(emotion_labels)} emotion classes")
    print(f" -> Image tensor shape: {images_tensor.shape}")
    print(f" -> Labels tensor shape: {labels_tensor.shape}")
    
    return images_tensor, labels_tensor, emotion_to_idx, image_paths

def get_next_model_folder(models_dir):
    """
    Get the next available model folder name (model_00001, model_00002, etc.)
    """
    existing_folders = []
    if os.path.exists(models_dir):
        for folder in os.listdir(models_dir):
            if folder.startswith('model_') and len(folder) == 11:
                try:
                    num = int(folder.split('_')[1])
                    existing_folders.append(num)
                except ValueError:
                    continue
    
    next_num = max(existing_folders) + 1 if existing_folders else 1
    return f"model_{next_num:05d}"

# Example usage and training script
if __name__ == "__main__":
    
    # Create argument parser for command line arguments
    parser = argparse.ArgumentParser(description='Train ConvolutionalNN with configurable data usage factors')
    parser.add_argument('--fer', type=float, default=1.0, 
                        help='FER original usage factor (0.8-1.0, default: 1.0)')
    parser.add_argument('--fer-aug', type=float, default=0.01,
                        help='FER augmented usage factor (0.0-1.0, default: 0.01)')
    parser.add_argument('--ckp', type=float, default=1.0,
                        help='CKP original usage factor (0.8-1.0, default: 1.0)')
    parser.add_argument('--ckp-aug', type=float, default=0.01,
                        help='CKP augmented usage factor (0.0-1.0, default: 0.01)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.8 <= args.fer <= 1.0:
        print(f"Warning: --fer should be between 0.8 and 1.0, got {args.fer}")
    if not 0.0 <= args.fer_aug <= 1.0:
        print(f"Warning: --fer-aug should be between 0.0 and 1.0, got {args.fer_aug}")
    if not 0.8 <= args.ckp <= 1.0:
        print(f"Warning: --ckp should be between 0.8 and 1.0, got {args.ckp}")
    if not 0.0 <= args.ckp_aug <= 1.0:
        print(f"Warning: --ckp-aug should be between 0.0 and 1.0, got {args.ckp_aug}")
    
    # Get the directory where this script is located (models directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # SET THE PARAMETERS HERE
    image_size = (48, 48)  # Changed from 64x64 to match actual image size
    channels = 1           # Grayscale images
    filters = 10           # Using original 10 filters as specified
    kernel_size = (4, 4)   # Same as original default
    batch_size = 32 
    epochs = 50
    # Fixed emotion order for consistency
    emotion_labels = ['anger', 'fear', 'calm', 'surprise']
    
    # DATA USAGE PARAMETERS
    # These control what fraction of available data to use (0.0 to 1.0)
    # FER datasets: 85% will be used for training, 15% for validation
    # CKP datasets: 100% will be used for validation only
    # Values come from command line arguments or defaults
    fer_usage_factor = args.fer            # Use 0.8 to 1.0 of FER original images
    fer_aug_usage_factor = args.fer_aug    # Use 0.0 to 1.0 of FER augmented images  
    ckp_usage_factor = args.ckp            # Use 0.8 to 1.0 of CKP original images
    ckp_aug_usage_factor = args.ckp_aug    # Use 0.0 to 1.0 of CKP augmented images
    
    print("\n" + "="*60)
    print("DATA USAGE FACTORS (from command line or defaults)")
    print("="*60)
    print(f"FER Original: {fer_usage_factor:.2f}")
    print(f"FER Augmented: {fer_aug_usage_factor:.2f}")
    print(f"CKP Original: {ckp_usage_factor:.2f}")
    print(f"CKP Augmented: {ckp_aug_usage_factor:.2f}")
    print("="*60)
    
    train_val_split_ratio = 0.85    # 85% train, 15% validation for FER
    random_seed = 42                # For reproducible data splits
    
    # Set all random seeds for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # SET THE PARAMETERS HERE
    
    # Define data paths for original and augmented datasets - go up to project root
    project_root = os.path.dirname(os.path.dirname(script_dir))  # models -> develop_baseline_model -> EmoTorch
    fer_original_path = os.path.join(project_root, "FER-Original-Dataset", "FER")
    fer_augmented_path = os.path.join(project_root, "FER-Original-Dataset-Augmented", "FER")
    ckp_original_path = os.path.join(project_root, "FER-Original-Dataset", "CKP")
    ckp_augmented_path = os.path.join(project_root, "FER-Original-Dataset-Augmented", "CKP")
    
    print("\n" + "="*60)
    print("LOADING DATASETS WITH USAGE FACTORS")
    print("="*60)
    
    # Load FER original dataset with usage factor
    print(f"\n[FER Original] Loading with usage_factor={fer_usage_factor}...")
    fer_images, fer_labels, emotion_mapping, fer_paths = load_emotion_data(
        fer_original_path, emotion_labels, usage_factor=fer_usage_factor, random_seed=random_seed
    )
    
    # Load FER augmented dataset with usage factor
    print(f"\n[FER Augmented] Loading with usage_factor={fer_aug_usage_factor}...")
    fer_aug_images, fer_aug_labels, _, fer_aug_paths = load_emotion_data(
        fer_augmented_path, emotion_labels, usage_factor=fer_aug_usage_factor, random_seed=random_seed
    )
    
    # Combine FER original and augmented
    fer_all_images = torch.cat([fer_images, fer_aug_images], dim=0)
    fer_all_labels = torch.cat([fer_labels, fer_aug_labels], dim=0)
    fer_all_paths = fer_paths + fer_aug_paths
    
    # Split FER data into 85% train and 15% validation
    print(f"\n[FER Split] Splitting into {train_val_split_ratio*100:.0f}% train and {(1-train_val_split_ratio)*100:.0f}% validation...")
    fer_indices = np.arange(len(fer_all_labels))
    train_indices, val_indices = train_test_split(
        fer_indices, test_size=(1-train_val_split_ratio), random_state=random_seed, 
        stratify=fer_all_labels.numpy()
    )
    
    # Create FER train and validation sets
    train_images = fer_all_images[train_indices]
    train_labels = fer_all_labels[train_indices]
    train_paths = [fer_all_paths[i] for i in train_indices]
    
    fer_val_images = fer_all_images[val_indices]
    fer_val_labels = fer_all_labels[val_indices]
    fer_val_paths = [fer_all_paths[i] for i in val_indices]
    
    # Load CKP original dataset with usage factor (validation only)
    print(f"\n[CKP Original] Loading with usage_factor={ckp_usage_factor} (validation only)...")
    ckp_images, ckp_labels, _, ckp_paths = load_emotion_data(
        ckp_original_path, emotion_labels, usage_factor=ckp_usage_factor, random_seed=random_seed
    )
    
    # Load CKP augmented dataset with usage factor (validation only)
    print(f"\n[CKP Augmented] Loading with usage_factor={ckp_aug_usage_factor} (validation only)...")
    ckp_aug_images, ckp_aug_labels, _, ckp_aug_paths = load_emotion_data(
        ckp_augmented_path, emotion_labels, usage_factor=ckp_aug_usage_factor, random_seed=random_seed
    )
    
    # Combine CKP original and augmented for validation
    ckp_val_images = torch.cat([ckp_images, ckp_aug_images], dim=0)
    ckp_val_labels = torch.cat([ckp_labels, ckp_aug_labels], dim=0)
    ckp_val_paths = ckp_paths + ckp_aug_paths
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"FER Original: {len(fer_images)} images (factor={fer_usage_factor})")
    print(f"FER Augmented: {len(fer_aug_images)} images (factor={fer_aug_usage_factor})")
    print(f"FER Combined: {len(fer_all_images)} images")
    print(f" -> Training: {len(train_images)} images")
    print(f" -> Validation: {len(fer_val_images)} images")
    print(f"\nCKP Original: {len(ckp_images)} images (factor={ckp_usage_factor})")
    print(f"CKP Augmented: {len(ckp_aug_images)} images (factor={ckp_aug_usage_factor})")
    print(f"CKP Combined (val only): {len(ckp_val_images)} images")
    print("="*60 + "\n")
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(train_images, train_labels)
    fer_val_dataset = TensorDataset(fer_val_images, fer_val_labels)
    ckp_val_dataset = TensorDataset(ckp_val_images, ckp_val_labels)
    
    # Create combined validation dataset (FER + CKP)
    combined_val_images = torch.cat([fer_val_images, ckp_val_images], dim=0)
    combined_val_labels = torch.cat([fer_val_labels, ckp_val_labels], dim=0)
    combined_val_dataset = TensorDataset(combined_val_images, combined_val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    fer_val_loader = DataLoader(fer_val_dataset, batch_size=batch_size, shuffle=False)
    ckp_val_loader = DataLoader(ckp_val_dataset, batch_size=batch_size, shuffle=False)
    combined_val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get number of classes
    num_classes = len(emotion_mapping)
    
    print(f"Emotion mapping: {emotion_mapping}")
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"FER Val batches: {len(fer_val_loader)}")
    print(f"CKP Val batches: {len(ckp_val_loader)}")
    print(f"Combined Val batches: {len(combined_val_loader)}")
    
    # Create model
    model = ConvolutionalNN(
        image_size=image_size,
        channels=channels, 
        num_classes=num_classes,
        filters=filters,
        kernel_size=kernel_size,
        verbose=True
    )
    
    # Initialize trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    trainer = EmotionRecognitionTrainer(model, device=device)
    
    # Train the model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        fer_val_loader=fer_val_loader,
        ckp_val_loader=ckp_val_loader,
        combined_val_loader=combined_val_loader,
        epochs=epochs,
        verbose=True
    )
    
    print("\nTraining completed!")
    print(f"Final train accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"Final FER validation accuracy: {history['fer_val_accuracies'][-1]:.2f}%")
    print(f"Final CKP validation accuracy: {history['ckp_val_accuracies'][-1]:.2f}%")
    print(f"Final Combined validation accuracy: {history['combined_val_accuracies'][-1]:.2f}%")
    
    # Create model folder with auto-incrementing number in the same directory as this script
    model_folder_name = get_next_model_folder(script_dir)
    model_dir = os.path.join(script_dir, model_folder_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\nCreated model folder: {model_folder_name}")
    
    # Save FER validation history
    fer_history_csv = os.path.join(model_dir, f"training_history_fer.csv")
    with open(fer_history_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'fer_val_loss', 'fer_val_accuracy'])
        for i in range(len(history['train_losses'])):
            writer.writerow([
                i+1,
                history['train_losses'][i],
                history['train_accuracies'][i],
                history['fer_val_losses'][i],
                history['fer_val_accuracies'][i]
            ])
    print(f" --> FER validation history saved to: {fer_history_csv}")
    
    # Save CKP validation history
    ckp_history_csv = os.path.join(model_dir, f"training_history_ckp.csv")
    with open(ckp_history_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'ckp_val_loss', 'ckp_val_accuracy'])
        for i in range(len(history['train_losses'])):
            writer.writerow([
                i+1,
                history['train_losses'][i],
                history['train_accuracies'][i],
                history['ckp_val_losses'][i],
                history['ckp_val_accuracies'][i]
            ])
    print(f" --> CKP validation history saved to: {ckp_history_csv}")
    
    # Save Combined validation history
    combined_history_csv = os.path.join(model_dir, f"training_history_combined.csv")
    with open(combined_history_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'combined_val_loss', 'combined_val_accuracy'])
        for i in range(len(history['train_losses'])):
            writer.writerow([
                i+1,
                history['train_losses'][i],
                history['train_accuracies'][i],
                history['combined_val_losses'][i],
                history['combined_val_accuracies'][i]
            ])
    print(f" --> Combined validation history saved to: {combined_history_csv}")

    # Generate confusion matrices for validation data
    print("\n -> Generating validation confusion matrices...")
    fer_val_cm = trainer.get_confusion_matrix(fer_val_loader)
    ckp_val_cm = trainer.get_confusion_matrix(ckp_val_loader)
    
    # Save the trained model
    model_save_path = os.path.join(model_dir, "model.pth")
    trainer.save_model(model_save_path)
    
    # Save confusion matrices to CSV
    fer_cm_csv = os.path.join(model_dir, f"confusion_matrix_fer_val.csv")
    ckp_cm_csv = os.path.join(model_dir, f"confusion_matrix_ckp_val.csv")
    
    # Emotion labels are already in the correct order since we use fixed emotion_labels
    # No reordering needed - confusion matrices are already in anger, fear, calm, surprise order
    
    # Convert confusion matrices to proportions (normalize by row, values between 0 and 1)
    fer_cm_normalized = fer_val_cm.astype('float') / fer_val_cm.sum(axis=1)[:, np.newaxis]
    ckp_cm_normalized = ckp_val_cm.astype('float') / ckp_val_cm.sum(axis=1)[:, np.newaxis]
    
    # Save FER validation confusion matrix (as proportions 0-1)
    with open(fer_cm_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + emotion_labels)  # Header row
        for i, row in enumerate(fer_cm_normalized):
            writer.writerow([emotion_labels[i]] + [f"{val:.3f}" for val in row])
    print(f" --> FER validation confusion matrix saved to: {fer_cm_csv}")
    
    # Save CKP validation confusion matrix (as proportions 0-1)
    with open(ckp_cm_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + emotion_labels)  # Header row
        for i, row in enumerate(ckp_cm_normalized):
            writer.writerow([emotion_labels[i]] + [f"{val:.3f}" for val in row])
    print(f" --> CKP validation confusion matrix saved to: {ckp_cm_csv}")
    
    # Save model parameters to text file
    params_file = os.path.join(model_dir, "model_parameters.txt")
    with open(params_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL TRAINING PARAMETERS AND INFORMATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Folder: {model_folder_name}\n\n")
        
        f.write("ARCHITECTURE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Type: ConvolutionalNN (PyTorch)\n")
        f.write(f"Input Shape: {channels} x {image_size[0]} x {image_size[1]}\n")
        f.write(f"Filters: {filters}\n")
        f.write(f"Kernel Size: {kernel_size}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Emotion Labels: {', '.join(emotion_labels)}\n\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Optimizer: RMSprop (lr=0.001)\n")
        f.write(f"Loss Function: Cosine Proximity\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs Configured: {epochs}\n")
        f.write(f"Epochs Completed: {len(history['train_losses'])}\n")
        f.write(f"Early Stopping: patience=3\n")
        f.write(f"ReduceLROnPlateau: factor=0.5, patience=5\n\n")
        
        f.write("DATA USAGE FACTORS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"FER Original Usage: {fer_usage_factor:.6f} ({fer_usage_factor*100:.3f}%)\n")
        f.write(f"FER Augmented Usage: {fer_aug_usage_factor:.6f} ({fer_aug_usage_factor*100:.3f}%)\n")
        f.write(f"CKP Original Usage: {ckp_usage_factor:.6f} ({ckp_usage_factor*100:.3f}%)\n")
        f.write(f"CKP Augmented Usage: {ckp_aug_usage_factor:.6f} ({ckp_aug_usage_factor*100:.3f}%)\n")
        f.write(f"Train/Val Split (FER): {train_val_split_ratio*100:.1f}/{(1-train_val_split_ratio)*100:.1f}\n\n")
        
        f.write("RANDOM SEEDS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Main Random Seed: {random_seed}\n")
        f.write(f"Data Sampling Seed: {random_seed}\n")
        f.write(f"Train/Val Split Seed: {random_seed}\n")
        f.write(f"PyTorch Manual Seed: {random_seed}\n")
        f.write(f"NumPy Random Seed: {random_seed}\n")
        f.write(f"Python Random Seed: {random_seed}\n")
        f.write(f"CUDA Deterministic: True\n")
        f.write(f"CUDNN Benchmark: False\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Images: {len(train_dataset)}\n")
        f.write(f"FER Validation Images: {len(fer_val_dataset)}\n")
        f.write(f"CKP Validation Images: {len(ckp_val_dataset)}\n")
        f.write(f"Combined Validation Images: {len(combined_val_dataset)}\n")
        f.write(f"Total Images: {len(train_dataset) + len(fer_val_dataset) + len(ckp_val_dataset)}\n\n")
        
        f.write("FINAL PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final Training Accuracy: {history['train_accuracies'][-1]:.2f}%\n")
        f.write(f"Final Training Loss: {history['train_losses'][-1]:.4f}\n")
        f.write(f"Final FER Validation Accuracy: {history['fer_val_accuracies'][-1]:.2f}%\n")
        f.write(f"Final CKP Validation Accuracy: {history['ckp_val_accuracies'][-1]:.2f}%\n")
        f.write(f"Final Combined Validation Accuracy: {history['combined_val_accuracies'][-1]:.2f}%\n")
        f.write(f"Final Combined Validation Loss: {history['combined_val_losses'][-1]:.4f}\n\n")
        
        f.write("DEVICE INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total Model Parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    print(f" --> Model parameters saved to: {params_file}")
    
    # Save final accuracies to CSV
    final_accuracies_csv = os.path.join(model_dir, "final_accuracies.csv")
    with open(final_accuracies_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Training Accuracy', f"{history['train_accuracies'][-1]:.2f}"])
        writer.writerow(['Training Loss', f"{history['train_losses'][-1]:.4f}"])
        writer.writerow(['FER Validation Accuracy', f"{history['fer_val_accuracies'][-1]:.2f}"])
        writer.writerow(['FER Validation Loss', f"{history['fer_val_losses'][-1]:.4f}"])
        writer.writerow(['CKP Validation Accuracy', f"{history['ckp_val_accuracies'][-1]:.2f}"])
        writer.writerow(['CKP Validation Loss', f"{history['ckp_val_losses'][-1]:.4f}"])
        writer.writerow(['Combined Validation Accuracy', f"{history['combined_val_accuracies'][-1]:.2f}"])
        writer.writerow(['Combined Validation Loss', f"{history['combined_val_losses'][-1]:.4f}"])
        writer.writerow(['Epochs Completed', len(history['train_losses'])])
        writer.writerow(['Total Training Images', len(train_dataset)])
        writer.writerow(['Total Validation Images', len(combined_val_dataset)])
    
    print(f" --> Final accuracies saved to: {final_accuracies_csv}")
    
    # Save image usage tracking
    def save_image_usage_info():
        """Save detailed information about which images were used for training/validation"""
        
        # Create training set info
        train_usage = []
        for i, path in enumerate(train_paths):
            train_usage.append({
                'image_path': path,
                'usage': 'training',
                'emotion': emotion_labels[train_labels[i].item()],
                'label_index': train_labels[i].item(),
                'dataset_index': i
            })
        
        # Create FER validation set info  
        fer_val_usage = []
        for i, path in enumerate(fer_val_paths):
            fer_val_usage.append({
                'image_path': path,
                'usage': 'fer_validation',
                'emotion': emotion_labels[fer_val_labels[i].item()],
                'label_index': fer_val_labels[i].item(),
                'dataset_index': i
            })
        
        # Create CKP validation set info
        ckp_val_usage = []
        for i, path in enumerate(ckp_val_paths):
            ckp_val_usage.append({
                'image_path': path,
                'usage': 'ckp_validation', 
                'emotion': emotion_labels[ckp_val_labels[i].item()],
                'label_index': ckp_val_labels[i].item(),
                'dataset_index': i
            })
        
        # Combine all usage info
        all_usage = train_usage + fer_val_usage + ckp_val_usage
        
        # Save to CSV
        usage_csv = os.path.join(model_dir, "image_usage.csv")
        with open(usage_csv, 'w', newline='') as f:
            if all_usage:
                writer = csv.DictWriter(f, fieldnames=['image_path', 'usage', 'emotion', 'label_index', 'dataset_index'])
                writer.writeheader()
                writer.writerows(all_usage)
        
        print(f" --> Image usage tracking saved to: {usage_csv}")
        
        # Save summary statistics
        summary_csv = os.path.join(model_dir, "usage_summary.csv")
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Usage Type', 'Total Images', 'Anger', 'Fear', 'Calm', 'Surprise'])
            
            # Count by usage and emotion
            for usage_type, usage_data in [('Training', train_usage), ('FER_Validation', fer_val_usage), ('CKP_Validation', ckp_val_usage)]:
                emotion_counts = {emotion: 0 for emotion in emotion_labels}
                for item in usage_data:
                    emotion_counts[item['emotion']] += 1
                
                writer.writerow([
                    usage_type,
                    len(usage_data),
                    emotion_counts['anger'],
                    emotion_counts['fear'], 
                    emotion_counts['calm'],
                    emotion_counts['surprise']
                ])
        
        print(f" --> Usage summary saved to: {summary_csv}")
    
    save_image_usage_info()
