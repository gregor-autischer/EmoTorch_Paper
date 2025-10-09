############################################
# PyTorch Implementation of New ConvolutionalNN
#
# by Gregor Autischer (September 2025)
############################################

import os
import sys
import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
import csv
import random
import argparse
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
from tqdm import tqdm
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class EnhancedConvolutionalNN(nn.Module):
    def __init__(self, image_size, channels, num_classes, dropout_rate=0.4):
        super(EnhancedConvolutionalNN, self).__init__()

        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Filters
        self.filters_1 = 32
        self.filters_2 = 64
        self.filters_3 = 128

        # 3x3 kernel with padding=1
        conv_kernel = (3, 3)
        conv_padding = 1

        # First convolutional block (32 filters)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=self.filters_1,
                              kernel_size=conv_kernel, padding=conv_padding)
        self.bn1 = nn.BatchNorm2d(self.filters_1)
        self.conv2 = nn.Conv2d(in_channels=self.filters_1, out_channels=self.filters_1,
                              kernel_size=conv_kernel, padding=conv_padding)
        self.bn2 = nn.BatchNorm2d(self.filters_1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=dropout_rate * 0.5)  # Less dropout early

        # Second convolutional block (64 filters)
        self.conv3 = nn.Conv2d(in_channels=self.filters_1, out_channels=self.filters_2,
                              kernel_size=conv_kernel, padding=conv_padding)
        self.bn3 = nn.BatchNorm2d(self.filters_2)
        self.conv4 = nn.Conv2d(in_channels=self.filters_2, out_channels=self.filters_2,
                              kernel_size=conv_kernel, padding=conv_padding)
        self.bn4 = nn.BatchNorm2d(self.filters_2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=dropout_rate * 0.75)

        # Third convolutional block (128 filters)
        self.conv5 = nn.Conv2d(in_channels=self.filters_2, out_channels=self.filters_3,
                              kernel_size=conv_kernel, padding=conv_padding)
        self.bn5 = nn.BatchNorm2d(self.filters_3)
        self.conv6 = nn.Conv2d(in_channels=self.filters_3, out_channels=self.filters_3,
                              kernel_size=conv_kernel, padding=conv_padding)
        self.bn6 = nn.BatchNorm2d(self.filters_3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(self.filters_3, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third convolutional block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layer
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

class FERCSVDataset(Dataset):
    def __init__(self, csv_path, split='train'):
        self.csv_path = csv_path
        self.split = split

        # Load CSV
        print(f"Loading dataset from: {csv_path}")
        self.df = pd.read_csv(csv_path)

        # Filter by split
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        print(f"Loaded {len(self.df)} images for {split} split")

        # Emotion mapping
        self.emotion_to_idx = {
            'anger': 0,
            'fear': 1,
            'calm': 2,
            'surprise': 3
        }

        # Print class distribution
        print(f"\nClass distribution for {split}:")
        self.class_counts = {}
        for emotion in ['anger', 'fear', 'calm', 'surprise']:
            count = len(self.df[self.df['emotion'] == emotion])
            self.class_counts[emotion] = count
            print(f"  {emotion}: {count}")

        # Print demographic distribution
        if 'gender' in self.df.columns:
            print(f"\nGender distribution for {split}:")
            for gender in self.df['gender'].unique():
                count = len(self.df[self.df['gender'] == gender])
                print(f"  {gender}: {count}")

        if 'race' in self.df.columns:
            print(f"\nRace distribution for {split}:")
            for race in self.df['race'].unique():
                if pd.notna(race):
                    count = len(self.df[self.df['race'] == race])
                    print(f"  {race}: {count}")

    def get_labels(self):
        """Return all labels for the dataset (used for WeightedRandomSampler)"""
        return [self.emotion_to_idx[emotion] for emotion in self.df['emotion']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        emotion = row['emotion']
        label = self.emotion_to_idx[emotion]

        # Store metadata for fairness analysis
        metadata = {
            'gender': row.get('gender', None),
            'race': row.get('race', None),
            'age': row.get('age', None),
            'augmentation': row.get('augmented', 'original')
        }

        try:
            # Load image as grayscale
            img = Image.open(img_path).convert('L')

            # Resize to 48x48 if needed (should not be needed with my dataset)
            if img.size != (48, 48):
                img = img.resize((48, 48))

            # Convert to numpy and normalize
            img_array = np.array(img, dtype=np.float32)

            # Validate image data before normalization
            if img_array.size == 0:
                raise ValueError("Empty image array")
            if not np.isfinite(img_array).all():
                raise ValueError("Image contains non-finite values")

            img_array = img_array / 255.0

            # Check for NaN or Inf after normalization
            if not np.isfinite(img_array).all():
                raise ValueError("Normalized image contains non-finite values")

            # Add channel dimension: (H, W) -> (1, H, W)
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            # Final check for NaN in tensor
            if not torch.isfinite(img_tensor).all():
                raise ValueError("Tensor contains non-finite values")

            return img_tensor, label, metadata

        except Exception as e:
            print(f"[ERROR] Error loading {img_path}: {type(e).__name__}: {e}")
            # Return a small random tensor to avoid model issues
            return torch.rand(1, 48, 48) * 0.1 + 0.5, 0, metadata

class EmotionRecognitionTrainer:

    def __init__(self, model, device='cpu', class_weights=None):
        self.model = model.to(device)
        self.device = device

        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Weighted CrossEntropyLoss")

        # AdamW optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        # CosineAnnealingWarmRestarts scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )

    def train(self, train_loader, epochs,val_loader=None, verbose=True):
        """
        Train the model with validation support.
        """
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # Early stopping patience
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None

        # Progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc='Epochs', position=0)

        for epoch in epoch_pbar:
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Progress bar for training batches
            train_pbar = tqdm(train_loader, desc='Training', leave=False)

            for batch_idx, batch_data in enumerate(train_pbar):
                # Handle both (data, target) and (data, target, metadata) formats
                if len(batch_data) == 3:
                    data, target, _ = batch_data
                else:
                    data, target = batch_data

                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

                # Update progress bar
                current_acc = 100.0 * train_correct / train_total
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%'})

            train_loss /= len(train_loader)
            train_accuracy = 100. * train_correct / train_total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Step the scheduler after each epoch
            self.scheduler.step()

            # Validation phase
            if val_loader is not None:
                val_loss, val_accuracy = self._validate(val_loader)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
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
                val_losses.append(0)
                val_accuracies.append(0)

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{train_accuracy:.2f}%',
                'val_loss': f'{val_losses[-1]:.4f}',
                'val_acc': f'{val_accuracies[-1]:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            if verbose:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | '
                      f'Val Acc: {val_accuracies[-1]:.2f}%')
                if val_loader is not None:
                    print(f'  -> LR: {self.optimizer.param_groups[0]["lr"]:.6f}, '
                          f'Best Val Loss: {best_val_loss:.4f}, '
                          f'Patience: {patience_counter}/{patience}')

        # Close progress bar
        epoch_pbar.close()

        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Progress bar for validation batches
        val_pbar = tqdm(val_loader, desc='Validation', leave=False)

        with torch.no_grad():
            for batch_data in val_pbar:
                # Handle both (data, target) and (data, target, metadata) formats
                if len(batch_data) == 3:
                    data, target, _ = batch_data
                else:
                    data, target = batch_data

                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

                # Update progress bar
                current_acc = 100.0 * val_correct / val_total
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%'})

        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        return val_loss, val_accuracy

    def predict(self, data_loader, return_targets=False):
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_data in data_loader:
                # Handle both (data, target) and (data, target, metadata) formats
                if len(batch_data) == 3:
                    data, target, _ = batch_data
                else:
                    data, target = batch_data

                data = data.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                predictions.extend(predicted.cpu().numpy())
                if return_targets:
                    targets.extend(target.cpu().numpy())

        if return_targets:
            return np.array(predictions), np.array(targets)
        return np.array(predictions)

    def get_predictions_with_metadata(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_metadata = []

        with torch.no_grad():
            for batch_data in data_loader:
                # Handle both (data, target) and (data, target, metadata) formats
                if len(batch_data) == 3:
                    data, target, metadata = batch_data

                    # Metadata is a dict of lists (batched), convert to list of dicts
                    batch_size = len(target)
                    for i in range(batch_size):
                        sample_metadata = {
                            'gender': metadata['gender'][i] if 'gender' in metadata else None,
                            'race': metadata['race'][i] if 'race' in metadata else None,
                            'age': metadata['age'][i] if 'age' in metadata else None,
                            'augmentation': metadata['augmentation'][i] if 'augmentation' in metadata else None
                        }
                        all_metadata.append(sample_metadata)
                else:
                    data, target = batch_data

                data = data.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.numpy())

        return np.array(all_preds), np.array(all_labels), all_metadata

    def get_confusion_matrix(self, data_loader):
        predictions, targets = self.predict(data_loader, return_targets=True)
        return confusion_matrix(targets, predictions)

    @staticmethod
    def save_confusion_matrix_image(cm, emotion_labels, save_path, title="Confusion Matrix", normalize=True):
        """
        Save confusion matrix as an image file.
        """
        if normalize:
            cm_display = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            fmt = '.3f'
            cmap = 'Blues'
        else:
            cm_display = cm
            fmt = 'd'
            cmap = 'Blues'

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=emotion_labels, yticklabels=emotion_labels,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        plt.title(title, fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': {
                'image_size': self.model.image_size,
                'channels': self.model.channels,
                'num_classes': self.model.num_classes,
                'dropout_rate': self.model.dropout_rate
            }
        }, filepath)
        print(f"Model saved to: {filepath}")

def compute_fairness_metrics(labels, preds, metadata_list, attribute_name, emotion_labels):
    """
    Compute fairness metrics for a specific demographic attribute
    Returns: Dictionary with metrics for each group
    """
    results = {}

    # Extract attribute values
    if attribute_name == 'gender':
        attribute_values = [m['gender'] for m in metadata_list]
    elif attribute_name == 'race':
        attribute_values = [m['race'] for m in metadata_list]
    elif attribute_name == 'age':
        attribute_values = [m['age'] for m in metadata_list]
    elif attribute_name == 'augmentation':
        attribute_values = [m['augmentation'] for m in metadata_list]
    else:
        return results

    # Get unique groups (excluding None/NaN)
    unique_groups = list(set([v for v in attribute_values if v is not None and pd.notna(v)]))
    unique_groups.sort()

    for group in unique_groups:
        # Filter data for this group
        group_indices = [i for i, v in enumerate(attribute_values) if v == group]

        if len(group_indices) == 0:
            continue

        group_labels = labels[group_indices]
        group_preds = preds[group_indices]

        # Compute confusion matrix
        cm = confusion_matrix(group_labels, group_preds, labels=range(len(emotion_labels)))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        # Compute metrics
        accuracy = accuracy_score(group_labels, group_preds)
        precision_macro = precision_score(group_labels, group_preds, average='macro', zero_division=0)
        recall_macro = recall_score(group_labels, group_preds, average='macro', zero_division=0)
        f1_macro = f1_score(group_labels, group_preds, average='macro', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(group_labels, group_preds, average=None, zero_division=0)
        recall_per_class = recall_score(group_labels, group_preds, average=None, zero_division=0)
        f1_per_class = f1_score(group_labels, group_preds, average=None, zero_division=0)

        # Classification report
        report = classification_report(
            group_labels,
            group_preds,
            target_names=emotion_labels,
            digits=4,
            zero_division=0
        )

        results[group] = {
            'count': len(group_indices),
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'classification_report': report
        }

    return results


def save_fairness_report(fairness_results, output_path, emotion_labels):
    """
    Save comprehensive fairness report to text file
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FAIRNESS ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Gender Analysis
        if 'gender' in fairness_results and fairness_results['gender']:
            f.write("="*80 + "\n")
            f.write("GENDER ANALYSIS\n")
            f.write("="*80 + "\n\n")

            for gender, metrics in fairness_results['gender'].items():
                f.write("-"*80 + "\n")
                f.write(f"Gender: {gender.upper()}\n")
                f.write("-"*80 + "\n")
                f.write(f"Sample count: {metrics['count']}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
                f.write(f"Recall (macro): {metrics['recall_macro']:.4f}\n")
                f.write(f"F1-Score (macro): {metrics['f1_macro']:.4f}\n\n")

                f.write("Confusion Matrix (Normalized):\n")
                f.write("Image saved to model directory.\n\n")

                # Save confusion matrix image for this gender
                cm_path = os.path.join(model_dir, f"confusion_matrix_gender_{gender}.png")
                EmotionRecognitionTrainer.save_confusion_matrix_image(
                    metrics['confusion_matrix'], emotion_labels, cm_path,
                    title=f"Confusion Matrix - Gender: {gender.upper()} (Normalized)", normalize=True
                )

                f.write("\nPer-Class Metrics:\n")
                f.write(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
                f.write("-"*50 + "\n")
                for i, emotion in enumerate(emotion_labels):
                    f.write(f"{emotion:<12} {metrics['precision_per_class'][i]:<12.4f} "
                           f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f}\n")

                f.write("\nClassification Report:\n")
                f.write(metrics['classification_report'])
                f.write("\n\n")

        # Race Analysis
        if 'race' in fairness_results and fairness_results['race']:
            f.write("="*80 + "\n")
            f.write("RACE ANALYSIS\n")
            f.write("="*80 + "\n\n")

            for race, metrics in fairness_results['race'].items():
                f.write("-"*80 + "\n")
                f.write(f"Race: {race.upper()}\n")
                f.write("-"*80 + "\n")
                f.write(f"Sample count: {metrics['count']}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
                f.write(f"Recall (macro): {metrics['recall_macro']:.4f}\n")
                f.write(f"F1-Score (macro): {metrics['f1_macro']:.4f}\n\n")

                f.write("Confusion Matrix (Normalized):\n")
                f.write("Image saved to model directory.\n\n")

                # Save confusion matrix image for this race
                cm_path = os.path.join(model_dir, f"confusion_matrix_race_{race}.png")
                EmotionRecognitionTrainer.save_confusion_matrix_image(
                    metrics['confusion_matrix'], emotion_labels, cm_path,
                    title=f"Confusion Matrix - Race: {race.upper()} (Normalized)", normalize=True
                )

                f.write("\nPer-Class Metrics:\n")
                f.write(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
                f.write("-"*50 + "\n")
                for i, emotion in enumerate(emotion_labels):
                    f.write(f"{emotion:<12} {metrics['precision_per_class'][i]:<12.4f} "
                           f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f}\n")

                f.write("\nClassification Report:\n")
                f.write(metrics['classification_report'])
                f.write("\n\n")

        # Age Analysis
        if 'age' in fairness_results and fairness_results['age']:
            f.write("="*80 + "\n")
            f.write("AGE GROUP ANALYSIS\n")
            f.write("="*80 + "\n\n")

            for age, metrics in fairness_results['age'].items():
                f.write("-"*80 + "\n")
                f.write(f"Age Group: {age}\n")
                f.write("-"*80 + "\n")
                f.write(f"Sample count: {metrics['count']}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
                f.write(f"Recall (macro): {metrics['recall_macro']:.4f}\n")
                f.write(f"F1-Score (macro): {metrics['f1_macro']:.4f}\n\n")

                f.write("Confusion Matrix (Normalized):\n")
                f.write("Image saved to model directory.\n\n")

                # Save confusion matrix image for this age group
                cm_path = os.path.join(model_dir, f"confusion_matrix_age_{age}.png")
                EmotionRecognitionTrainer.save_confusion_matrix_image(
                    metrics['confusion_matrix'], emotion_labels, cm_path,
                    title=f"Confusion Matrix - Age Group: {age} (Normalized)", normalize=True
                )

                f.write("\nPer-Class Metrics:\n")
                f.write(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
                f.write("-"*50 + "\n")
                for i, emotion in enumerate(emotion_labels):
                    f.write(f"{emotion:<12} {metrics['precision_per_class'][i]:<12.4f} "
                           f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f}\n")

                f.write("\nClassification Report:\n")
                f.write(metrics['classification_report'])
                f.write("\n\n")

        # Augmentation Analysis
        if 'augmentation' in fairness_results and fairness_results['augmentation']:
            f.write("="*80 + "\n")
            f.write("AUGMENTATION TYPE ANALYSIS\n")
            f.write("="*80 + "\n\n")

            for aug_type, metrics in fairness_results['augmentation'].items():
                f.write("-"*80 + "\n")
                f.write(f"Augmentation Type: {aug_type.upper().replace('_', ' ')}\n")
                f.write("-"*80 + "\n")
                f.write(f"Sample count: {metrics['count']}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
                f.write(f"Recall (macro): {metrics['recall_macro']:.4f}\n")
                f.write(f"F1-Score (macro): {metrics['f1_macro']:.4f}\n\n")

                f.write("Confusion Matrix (Raw Counts):\n")
                f.write("              " + "  ".join([f"{e:>10s}" for e in emotion_labels]) + "\n")
                for i, row in enumerate(metrics['confusion_matrix']):
                    f.write(f"{emotion_labels[i]:>12s}  " + "  ".join([f"{val:>10d}" for val in row]) + "\n")

                f.write("\nConfusion Matrix (Normalized):\n")
                f.write("              " + "  ".join([f"{e:>10s}" for e in emotion_labels]) + "\n")
                for i, row in enumerate(metrics['confusion_matrix_normalized']):
                    f.write(f"{emotion_labels[i]:>12s}  " + "  ".join([f"{val:>10.4f}" for val in row]) + "\n")

                f.write("\nPer-Class Metrics:\n")
                f.write(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
                f.write("-"*50 + "\n")
                for i, emotion in enumerate(emotion_labels):
                    f.write(f"{emotion:<12} {metrics['precision_per_class'][i]:<12.4f} "
                           f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f}\n")

                f.write("\nClassification Report:\n")
                f.write(metrics['classification_report'])
                f.write("\n\n")


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

# RUN THE SCRIPT AND SET PARAMS
if __name__ == "__main__":

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # SET THE PARAMETERS HERE
    image_size = (48, 48)  # 48x48 image
    channels = 1           # Grayscale image
    dropout_rate = 0.4
    batch_size = 32
    epochs = 50

    # Fixed emotion order for consistency
    emotion_labels = ['anger', 'fear', 'calm', 'surprise']

    random_seed = 42                # For reproducibility

    # Set all random seeds for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # SET THE PARAMETERS HERE

    # Define data paths - go up to project root
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "FER-New-Dataset", "dataset_calm_reduced.csv")

    print("\n" + "="*60)
    print("ENHANCED CONVOLUTIONAL NN V2 - LOADING DATASET FROM CSV")
    print("="*60)
    print(f"CSV path: {csv_path}")

    # Load datasets using CSV
    print(f"\n[Train] Loading training split...")
    train_dataset = FERCSVDataset(csv_path, split='train')

    print(f"\n[Test] Loading test split...")
    test_dataset = FERCSVDataset(csv_path, split='test')

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Training: {len(train_dataset)} images")
    print(f"Test: {len(test_dataset)} images")
    print("="*60 + "\n")

    # Calculate class weights for handling imbalance
    print("\n" + "="*60)
    print("CALCULATING CLASS WEIGHTS")
    print("="*60)

    train_labels = train_dataset.get_labels()
    class_weights_np = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    print("Class weights (for handling imbalance):")
    for i, emotion in enumerate(emotion_labels):
        print(f"  {emotion}: {class_weights[i]:.3f}")

    # Create weighted sampler for oversampling minority classes
    print("\n" + "="*60)
    print("CREATING WEIGHTED SAMPLER FOR OVERSAMPLING")
    print("="*60)

    # Calculate sampling weights for each sample
    sample_weights = [class_weights[label].item() for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print(f"Using WeightedRandomSampler with {len(sample_weights)} samples")
    print("This will oversample minority classes (anger, fear) during training")

    # Create data loaders with weighted sampler for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Get number of classes
    num_classes = len(emotion_labels)

    print(f"\nEmotion mapping: {train_dataset.emotion_to_idx}")
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "="*60)
    print("CREATING ENHANCED MODEL")
    print("="*60)
    model = EnhancedConvolutionalNN(
        image_size=image_size,
        channels=channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    print("="*60)

    # Initialize trainer with class weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    trainer = EmotionRecognitionTrainer(
        model,
        device=device,
        class_weights=class_weights
    )

    # Train the model
    print("\nStarting training...")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        verbose=True
    )

    print("\nTraining completed!")
    print(f"Final train accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"Final test accuracy: {history['val_accuracies'][-1]:.2f}%")

    # Create model folder with auto-incrementing number in the same directory as this script
    model_folder_name = get_next_model_folder(script_dir)
    model_dir = os.path.join(script_dir, model_folder_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\nCreated model folder: {model_folder_name}")

    # Save training history
    history_csv = os.path.join(model_dir, f"training_history.csv")
    with open(history_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        for i in range(len(history['train_losses'])):
            writer.writerow([
                i+1,
                history['train_losses'][i],
                history['train_accuracies'][i],
                history['val_losses'][i],
                history['val_accuracies'][i]
            ])
    print(f" --> Training history saved to: {history_csv}")

    # Evaluate on test set with metadata
    print(f"\n" + "="*70)
    print("Final Evaluation with Fairness Analysis")
    print("="*70)

    test_preds, test_labels, test_metadata = trainer.get_predictions_with_metadata(test_loader)

    # Also get training predictions for confusion matrix
    print("\n -> Generating training confusion matrix...")
    train_preds, train_labels_cm, _ = trainer.get_predictions_with_metadata(train_loader)
    train_cm = confusion_matrix(train_labels_cm, train_preds)

    # Generate confusion matrices for test data
    print(" -> Generating test confusion matrix...")
    test_cm = confusion_matrix(test_labels, test_preds)

    # Training classification report
    train_report = classification_report(
        train_labels_cm,
        train_preds,
        target_names=emotion_labels,
        digits=4
    )

    # Test classification report
    test_report = classification_report(
        test_labels,
        test_preds,
        target_names=emotion_labels,
        digits=4
    )

    print("\nOverall Test Classification Report:")
    print(test_report)

    # Calculate recall scores for both sets
    train_recall_macro = recall_score(train_labels_cm, train_preds, average='macro')
    train_recall_weighted = recall_score(train_labels_cm, train_preds, average='weighted')
    train_recall_per_class = recall_score(train_labels_cm, train_preds, average=None)

    test_recall_macro = recall_score(test_labels, test_preds, average='macro')
    test_recall_weighted = recall_score(test_labels, test_preds, average='weighted')
    test_recall_per_class = recall_score(test_labels, test_preds, average=None)

    print("\nOverall Test Recall Scores:")
    print(f"  Macro Recall: {test_recall_macro:.4f}")
    print(f"  Weighted Recall: {test_recall_weighted:.4f}")
    print(f"\n  Per-Class Recall:")
    for emotion, recall in zip(emotion_labels, test_recall_per_class):
        print(f"    {emotion}: {recall:.4f}")

    # Compute prediction entropy and confidence
    print("\n -> Computing prediction entropy and confidence...")

    def compute_entropy_and_confidence(data_loader, model, device):
        """Compute average entropy and confidence of predictions"""
        model.eval()
        all_probs = []

        with torch.no_grad():
            for batch_data in data_loader:
                if len(batch_data) == 3:
                    data, _, _ = batch_data
                else:
                    data, _ = batch_data

                data = data.to(device)
                output = model(data)
                # Apply softmax to get probabilities
                probs = F.softmax(output, dim=1)
                all_probs.append(probs.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)

        # Compute entropy: -sum(p * log(p))
        epsilon = 1e-10  # To avoid log(0)
        entropy = -np.sum(all_probs * np.log(all_probs + epsilon), axis=1)
        avg_entropy = np.mean(entropy)

        # Compute confidence: max probability
        max_probs = np.max(all_probs, axis=1)
        avg_confidence = np.mean(max_probs)

        return avg_entropy, avg_confidence

    train_entropy, train_confidence = compute_entropy_and_confidence(train_loader, model, device)
    test_entropy, test_confidence = compute_entropy_and_confidence(test_loader, model, device)

    print(f"  Training - Avg Entropy: {train_entropy:.4f}, Avg Confidence: {train_confidence:.4f}")
    print(f"  Test - Avg Entropy: {test_entropy:.4f}, Avg Confidence: {test_confidence:.4f}")

    # Compute fairness metrics
    print(f"\n" + "="*70)
    print("Computing Fairness Metrics")
    print("="*70)

    fairness_results = {}

    # Gender fairness
    print("\nAnalyzing gender fairness...")
    fairness_results['gender'] = compute_fairness_metrics(
        test_labels, test_preds, test_metadata, 'gender', emotion_labels
    )

    # Race fairness
    print("Analyzing race fairness...")
    fairness_results['race'] = compute_fairness_metrics(
        test_labels, test_preds, test_metadata, 'race', emotion_labels
    )

    # Age fairness
    print("Analyzing age fairness...")
    fairness_results['age'] = compute_fairness_metrics(
        test_labels, test_preds, test_metadata, 'age', emotion_labels
    )

    # Augmentation fairness
    print("Analyzing augmentation type fairness...")
    fairness_results['augmentation'] = compute_fairness_metrics(
        test_labels, test_preds, test_metadata, 'augmentation', emotion_labels
    )

    # Save the trained model
    model_save_path = os.path.join(model_dir, "model.pth")
    trainer.save_model(model_save_path)

    # Save confusion matrix to CSV
    cm_csv = os.path.join(model_dir, f"confusion_matrix_test.csv")

    # Convert confusion matrix to proportions (normalize by row, values between 0 and 1)
    test_cm_normalized = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

    # Save test confusion matrix (as proportions 0-1)
    with open(cm_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + emotion_labels)  # Header row
        for i, row in enumerate(test_cm_normalized):
            writer.writerow([emotion_labels[i]] + [f"{val:.3f}" for val in row])
    print(f" --> Test confusion matrix saved to: {cm_csv}")

    # Save classification report with confusion matrices
    report_path = os.path.join(model_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        # Overall Summary
        f.write("="*70 + "\n")
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write("FINAL LOSS VALUES:\n")
        f.write(f"  Training Loss:   {history['train_losses'][-1]:.4f}\n")
        f.write(f"  Test Loss:       {history['val_losses'][-1]:.4f}\n\n")

        f.write("PREDICTION ENTROPY & CONFIDENCE:\n")
        f.write(f"  Training Set:\n")
        f.write(f"    Average Entropy:     {train_entropy:.4f}\n")
        f.write(f"    Average Confidence:  {train_confidence:.4f} ({train_confidence*100:.2f}%)\n")
        f.write(f"  Test Set:\n")
        f.write(f"    Average Entropy:     {test_entropy:.4f}\n")
        f.write(f"    Average Confidence:  {test_confidence:.4f} ({test_confidence*100:.2f}%)\n\n")

        f.write("Note: Lower entropy indicates more confident predictions.\n")
        f.write("      Entropy ranges from 0 (perfect confidence) to ~1.39 (uniform distribution over 4 classes).\n\n")

        # Training Performance
        f.write("="*70 + "\n")
        f.write("TRAINING SET PERFORMANCE\n")
        f.write("="*70 + "\n\n")
        f.write(train_report)
        f.write("\n\nTraining Recall Scores:\n")
        f.write(f"Macro Recall: {train_recall_macro:.4f}\n")
        f.write(f"Weighted Recall: {train_recall_weighted:.4f}\n")
        f.write(f"\nPer-Class Recall:\n")
        for emotion, recall in zip(emotion_labels, train_recall_per_class):
            f.write(f"  {emotion}: {recall:.4f}\n")

        # Test Performance
        f.write("\n" + "="*70 + "\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("="*70 + "\n\n")
        f.write(test_report)
        f.write("\n\nTest Recall Scores:\n")
        f.write(f"Macro Recall: {test_recall_macro:.4f}\n")
        f.write(f"Weighted Recall: {test_recall_weighted:.4f}\n")
        f.write(f"\nPer-Class Recall:\n")
        for emotion, recall in zip(emotion_labels, test_recall_per_class):
            f.write(f"  {emotion}: {recall:.4f}\n")

        # Save Training Confusion Matrix as Image (Normalized)
        f.write("\n" + "="*70 + "\n")
        f.write("TRAINING SET CONFUSION MATRIX\n")
        f.write("="*70 + "\n")
        f.write("Normalized confusion matrix image saved to model directory.\n")

        train_cm_path = os.path.join(model_dir, "confusion_matrix_train.png")
        EmotionRecognitionTrainer.save_confusion_matrix_image(
            train_cm, emotion_labels, train_cm_path,
            title="Training Set Confusion Matrix (Normalized)", normalize=True
        )

        # Save Test Confusion Matrix as Image (Normalized)
        f.write("\n" + "="*70 + "\n")
        f.write("TEST SET CONFUSION MATRIX\n")
        f.write("="*70 + "\n")
        f.write("Normalized confusion matrix image saved to model directory.\n")

        test_cm_path = os.path.join(model_dir, "confusion_matrix_test.png")
        EmotionRecognitionTrainer.save_confusion_matrix_image(
            test_cm, emotion_labels, test_cm_path,
            title="Test Set Confusion Matrix (Normalized)", normalize=True
        )

    print(f" --> Classification report saved to: {report_path}")

    # Save fairness report
    fairness_report_path = os.path.join(model_dir, 'fairness_analysis.txt')
    save_fairness_report(fairness_results, fairness_report_path, emotion_labels)
    print(f" --> Fairness analysis saved to: {fairness_report_path}")

    # Save model parameters to text file
    params_file = os.path.join(model_dir, "model_parameters.txt")
    with open(params_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ENHANCED MODEL V2 ARCHITECTURE\n")
        f.write("=" * 60 + "\n\n")

        # Write model architecture details from _print_model_summary
        f.write(f"Input shape: {channels} x {image_size[0]} x {image_size[1]}\n\n")
        f.write(f"Block 1 ({model.filters_1} filters):\n")
        f.write(f"  Conv2d-1 + BatchNorm2d: {model.filters_1} filters, kernel_size=3x3, padding=1\n")
        f.write(f"  Conv2d-2 + BatchNorm2d: {model.filters_1} filters, kernel_size=3x3, padding=1\n")
        f.write(f"  MaxPool2d: 2x2\n")
        f.write(f"  Dropout2d: p={dropout_rate * 0.5}\n\n")
        f.write(f"Block 2 ({model.filters_2} filters):\n")
        f.write(f"  Conv2d-3 + BatchNorm2d: {model.filters_2} filters, kernel_size=3x3, padding=1\n")
        f.write(f"  Conv2d-4 + BatchNorm2d: {model.filters_2} filters, kernel_size=3x3, padding=1\n")
        f.write(f"  MaxPool2d: 2x2\n")
        f.write(f"  Dropout2d: p={dropout_rate * 0.75}\n\n")
        f.write(f"Block 3 ({model.filters_3} filters):\n")
        f.write(f"  Conv2d-5 + BatchNorm2d: {model.filters_3} filters, kernel_size=3x3, padding=1\n")
        f.write(f"  Conv2d-6 + BatchNorm2d: {model.filters_3} filters, kernel_size=3x3, padding=1\n")
        f.write(f"  MaxPool2d: 2x2\n")
        f.write(f"  Dropout2d: p={dropout_rate}\n\n")
        f.write(f"Global Average Pooling\n\n")
        f.write(f"Fully Connected:\n")
        f.write(f"  Linear-1 + BatchNorm1d: {model.filters_3} -> 256\n")
        f.write(f"  Dropout: p={dropout_rate}\n")
        f.write(f"  Linear-2: 256 -> {num_classes}\n\n")
        f.write(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    print(f" --> Model parameters saved to: {params_file}")

    # Save final accuracies to CSV
    final_accuracies_csv = os.path.join(model_dir, "final_accuracies.csv")
    with open(final_accuracies_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Training Accuracy', f"{history['train_accuracies'][-1]:.2f}"])
        writer.writerow(['Training Loss', f"{history['train_losses'][-1]:.4f}"])
        writer.writerow(['Test Accuracy', f"{history['val_accuracies'][-1]:.2f}"])
        writer.writerow(['Test Loss', f"{history['val_losses'][-1]:.4f}"])
        writer.writerow(['Epochs Completed', len(history['train_losses'])])
        writer.writerow(['Total Training Images', len(train_dataset)])
        writer.writerow(['Total Test Images', len(test_dataset)])

    print(f" --> Final accuracies saved to: {final_accuracies_csv}")

    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"Model and results saved to: {model_dir}")
    print(f"Best Test Accuracy: {max(history['val_accuracies']):.2f}%")
    print(f"Test Macro Recall: {test_recall_macro:.4f}")
    print(f"Test Weighted Recall: {test_recall_weighted:.4f}")
    print(f"Test Avg Confidence: {test_confidence:.4f} ({test_confidence*100:.2f}%)")
    print(f"Test Avg Entropy: {test_entropy:.4f}")