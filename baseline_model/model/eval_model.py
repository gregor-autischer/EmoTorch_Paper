import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directories to path for imports  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import model class and trainer from convolutional_nn_pytorch
from develop_baseline_model.models.convolutional_nn_pytorch import EmotionRecognitionTrainer

# Define emotion labels
EMOTION_LABELS = ['anger', 'fear', 'calm', 'surprise']

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for prediction"""
    try:
        # Construct full path relative to the model directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, '..', image_path)
        
        img = Image.open(full_path).convert('L')
        img = img.resize((48, 48))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        return img_tensor
    except Exception as e:
        print(f" XX Error loading image {image_path}: {e}")
        return None

def evaluate_dataset(model, df_dataset, dataset_name):
    """Evaluate model on a specific dataset and return predictions and true labels"""
    print(f"\n[Evaluating {dataset_name} Dataset]")
    print(f" -> Total images: {len(df_dataset)}")
    
    predictions = []
    true_labels = []
    failed_images = 0
    
    for idx, row in df_dataset.iterrows():
        # Load and preprocess image
        img_tensor = load_and_preprocess_image(row['image_path'])
        
        if img_tensor is None:
            failed_images += 1
            continue
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        predictions.append(predicted_class)
        true_labels.append(row['label_index'])
        
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"   └-> Processed {idx + 1}/{len(df_dataset)} images...")
    
    if failed_images > 0:
        print(f" XX Warning: Failed to load {failed_images} images")
    
    return np.array(predictions), np.array(true_labels)

def create_combined_confusion_matrix_plot(fer_cm, ckp_cm, save_path="confusion_matrices.png"):
    """Create side-by-side confusion matrices styled like plot_confusion_matrix.py"""
    if fer_cm is None or ckp_cm is None:
        print(" XX Cannot create combined plot - missing data")
        return
    
    # Calculate percentages (normalize by row - true labels)
    fer_percent = fer_cm.astype('float') / fer_cm.sum(axis=1)[:, np.newaxis]
    ckp_percent = ckp_cm.astype('float') / ckp_cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with subplots
    matrix_size = max(4.5, len(EMOTION_LABELS) * 0.9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(matrix_size * 2.0, matrix_size))
    
    # Plot FER confusion matrix
    sns.heatmap(fer_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                cbar_kws={'shrink': 0.7},
                square=True, ax=ax1)
    
    # Add black outline around FER matrix
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')
    
    # Add black outline around colorbar
    cbar1 = ax1.collections[0].colorbar
    cbar1.outline.set_linewidth(1)
    cbar1.outline.set_edgecolor('black')
    
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion matrix,\nconvolutional model, FER dataset')
    
    # Plot CKP confusion matrix
    sns.heatmap(ckp_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                cbar_kws={'shrink': 0.7},
                square=True, ax=ax2)
    
    # Add black outline around CKP matrix
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')
    
    # Add black outline around colorbar
    cbar2 = ax2.collections[0].colorbar
    cbar2.outline.set_linewidth(1)
    cbar2.outline.set_edgecolor('black')
    
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Confusion matrix,\nconvolutional model, CKP dataset')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f" -> Combined confusion matrices saved to: {save_path}")
    
    plt.show()

def calculate_metrics(predictions, true_labels, dataset_name):
    """Calculate and print evaluation metrics"""
    # Check if we have any predictions
    if len(predictions) == 0:
        print(f"\n[{dataset_name} Results]")
        print(f" XX No images found for evaluation")
        return None
    
    # Overall accuracy
    accuracy = (predictions == true_labels).mean() * 100
    print(f"\n[{dataset_name} Results]")
    print(f" -> Overall Accuracy: {accuracy:.2f}%")
    
    # Per-class metrics
    cm = confusion_matrix(true_labels, predictions, labels=range(len(EMOTION_LABELS)))
    print("\n -> Per-Class Performance:")
    for i, emotion in enumerate(EMOTION_LABELS):
        if cm[i].sum() > 0:
            precision = cm[i, i] / cm[:, i].sum() * 100 if cm[:, i].sum() > 0 else 0
            recall = cm[i, i] / cm[i].sum() * 100
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"    └-> {emotion:8s}: Precision={precision:6.2f}%, Recall={recall:6.2f}%, F1={f1:6.2f}%")
    
    return cm

def main():
    # Load model
    print("[Loading Model]")
    model_path = 'model.pth'
    model, trainer = EmotionRecognitionTrainer.load_model(model_path)
    model.eval()
    print(f" -> Model loaded from: {model_path}")
    
    # Load image usage CSV
    print("\n[Loading Image Usage Data]")
    csv_path = 'image_usage.csv'
    df = pd.read_csv(csv_path)
    
    # Filter for training images only
    df_training = df[df['usage'] == 'training'].copy()
    print(f" -> Total training images: {len(df_training)}")
    
    # Separate FER and CKP datasets
    # For FER: use training images
    df_fer = df_training[df_training['image_path'].str.contains('FER')]
    # For CKP: use ckp_validation images since there are no CKP training images
    df_ckp = df[df['usage'] == 'ckp_validation'].copy()
    
    print(f"   └-> FER training images: {len(df_fer)}")
    print(f"   └-> CKP validation images (used for evaluation): {len(df_ckp)}")
    
    # Evaluate FER dataset
    fer_predictions, fer_true_labels = evaluate_dataset(model, df_fer, "FER")
    fer_cm = calculate_metrics(fer_predictions, fer_true_labels, "FER")
    
    # Evaluate CKP dataset
    ckp_predictions, ckp_true_labels = evaluate_dataset(model, df_ckp, "CKP")
    ckp_cm = calculate_metrics(ckp_predictions, ckp_true_labels, "CKP")
    
    # Create combined confusion matrix plot
    print("\n[Generating Confusion Matrices]")
    if fer_cm is not None and ckp_cm is not None:
        create_combined_confusion_matrix_plot(fer_cm, ckp_cm, "confusion_matrices.png")
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    if len(fer_predictions) > 0:
        print(f" -> FER Accuracy: {(fer_predictions == fer_true_labels).mean() * 100:.2f}%")
    if len(ckp_predictions) > 0:
        print(f" -> CKP Accuracy: {(ckp_predictions == ckp_true_labels).mean() * 100:.2f}%")
    if fer_cm is not None and ckp_cm is not None:
        print("\n -> Confusion matrices saved as: confusion_matrices.png")

if __name__ == "__main__":
    main()