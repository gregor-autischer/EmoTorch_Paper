############################################
# Evaluate Saved Model on Dataset
# 
# by Gregor Autischer (August 2025)
############################################

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import csv
from sklearn.metrics import confusion_matrix, classification_report

# Add parent directory to path for imports  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Go up to EmoTorch root

from develop_baseline_model.models.convolutional_nn_pytorch import EmotionRecognitionTrainer, load_emotion_data
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

# Fixed emotion labels
EMOTION_LABELS = ['anger', 'fear', 'calm', 'surprise']

def load_model_from_folder(model_folder):
    """Load model from a model folder"""
    model_path = os.path.join(model_folder, 'model.pth')
    if not os.path.exists(model_path):
        print(f" XX Model file not found: {model_path}")
        return None, None
    
    print(f" -> Loading model from: {model_folder}")
    model, trainer = EmotionRecognitionTrainer.load_model(model_path)
    model.eval()
    print(f" └-> Model loaded successfully")
    return model, trainer

def evaluate_on_dataset(model, trainer, data_path, dataset_name="Test"):
    """Evaluate model on a dataset and return detailed results"""
    print(f"\n[Evaluating on {dataset_name}]")
    print(f" -> Loading data from: {data_path}")
    
    # Load data
    try:
        images, labels, _, _ = load_emotion_data(data_path, EMOTION_LABELS)
        print(f" └-> Loaded {len(images)} images")
    except Exception as e:
        print(f" XX Error loading data: {e}")
        return None
    
    # Create dataloader
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    print(f" -> Running predictions...")
    predictions, true_labels = trainer.predict(dataloader, return_targets=True)
    
    # Calculate metrics
    accuracy = (predictions == true_labels).mean() * 100
    print(f" └-> Accuracy: {accuracy:.2f}%")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Find misclassifications
    misclassified_indices = np.where(predictions != true_labels)[0]
    
    results = {
        'dataset_name': dataset_name,
        'total_images': len(images),
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels,
        'confusion_matrix': cm,
        'misclassified_indices': misclassified_indices,
        'images': images,
        'labels': labels
    }
    
    return results

def analyze_misclassifications(results, output_dir, save_images=False, max_save=50):
    """Analyze and save misclassification details"""
    if results is None:
        return
    
    misclassified = results['misclassified_indices']
    print(f"\n[Misclassification Analysis]")
    print(f" -> Total misclassified: {len(misclassified)}/{results['total_images']} ({len(misclassified)/results['total_images']*100:.1f}%)")
    
    if len(misclassified) == 0:
        print(" -> Perfect classification! No errors.")
        return
    
    # Create misclassification matrix
    misclass_details = []
    for idx in misclassified:
        true_label = results['true_labels'][idx]
        pred_label = results['predictions'][idx]
        misclass_details.append({
            'image_index': idx,
            'true_emotion': EMOTION_LABELS[true_label],
            'predicted_emotion': EMOTION_LABELS[pred_label],
            'true_label_idx': true_label,
            'pred_label_idx': pred_label
        })
    
    # Count misclassification patterns
    patterns = {}
    for detail in misclass_details:
        pattern = f"{detail['true_emotion']} -> {detail['predicted_emotion']}"
        patterns[pattern] = patterns.get(pattern, 0) + 1
    
    # Sort patterns by frequency
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    
    print("\n[Most Common Misclassifications]")
    for pattern, count in sorted_patterns[:10]:
        print(f" -> {pattern}: {count} times")
    
    # Save misclassification details to CSV
    csv_path = os.path.join(output_dir, f"misclassifications_{results['dataset_name'].lower()}.csv")
    df = pd.DataFrame(misclass_details)
    df.to_csv(csv_path, index=False)
    print(f"\n -> Misclassification details saved to: {csv_path}")
    
    # Optionally save misclassified images
    if save_images and len(misclassified) > 0:
        save_misclassified_images(results, misclass_details, output_dir, max_save)

def save_misclassified_images(results, misclass_details, output_dir, max_save=50):
    """Save misclassified images for visual inspection"""
    img_dir = os.path.join(output_dir, f"misclassified_images_{results['dataset_name'].lower()}")
    os.makedirs(img_dir, exist_ok=True)
    
    num_to_save = min(len(misclass_details), max_save)
    print(f"\n[Saving Misclassified Images]")
    print(f" -> Saving {num_to_save} misclassified images to: {img_dir}")
    
    for i, detail in enumerate(misclass_details[:num_to_save]):
        idx = detail['image_index']
        # Get image tensor [1, 48, 48]
        img_tensor = results['images'][idx]
        
        # Convert to numpy and denormalize
        if len(img_tensor.shape) == 3:
            img_array = img_tensor.squeeze(0).numpy()  # Remove channel dim
        else:
            img_array = img_tensor.numpy()
        
        img_array = (img_array * 255).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(img_array, mode='L')
        filename = f"{i:03d}_true_{detail['true_emotion']}_pred_{detail['predicted_emotion']}.png"
        img.save(os.path.join(img_dir, filename))
    
    print(f" └-> Saved {num_to_save} images")

def save_detailed_report(results, output_dir):
    """Save detailed classification report"""
    if results is None:
        return
    
    report_path = os.path.join(output_dir, f"classification_report_{results['dataset_name'].lower()}.txt")
    
    # Generate classification report
    report = classification_report(
        results['true_labels'], 
        results['predictions'],
        target_names=EMOTION_LABELS,
        digits=3
    )
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"CLASSIFICATION REPORT - {results['dataset_name']}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Images: {results['total_images']}\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Misclassified: {len(results['misclassified_indices'])}\n\n")
        f.write("Detailed Metrics:\n")
        f.write("-"*40 + "\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write("-"*40 + "\n")
        
        # Format confusion matrix
        cm = results['confusion_matrix']
        f.write("True\\Pred\t" + "\t".join(EMOTION_LABELS) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{EMOTION_LABELS[i]}\t\t" + "\t".join(map(str, row)) + "\n")
    
    print(f" -> Classification report saved to: {report_path}")

def main():
    if len(sys.argv) < 2:
        print("\n[Model Evaluation Script]")
        print(" -> Usage: python evaluate_model.py <model_folder> [dataset_path]")
        print(" -> Examples:")
        print("    python evaluate_model.py model_00001")
        print("    python evaluate_model.py model_00001 ./FER-Original-Dataset/FER")
        print("    python evaluate_model.py model_original ./FER-Original-Dataset/CKP")
        return
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_folder_name = sys.argv[1]
    model_folder = os.path.join(script_dir, model_folder_name)
    
    # Default dataset path if not provided
    if len(sys.argv) > 2:
        dataset_path = sys.argv[2]
        dataset_name = os.path.basename(dataset_path)
    else:
        # Default to FER validation subset - go up to project root
        project_root = os.path.dirname(os.path.dirname(script_dir))  # models -> develop_baseline_model -> EmoTorch
        dataset_path = os.path.join(project_root, "FER-Original-Dataset", "FER")
        dataset_name = "FER-Original"
    
    # Check if model folder exists
    if not os.path.exists(model_folder):
        print(f" XX Model folder not found: {model_folder}")
        return
    
    # Load model
    model, trainer = load_model_from_folder(model_folder)
    if model is None:
        return
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f" XX Dataset not found: {dataset_path}")
        return
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(model_folder, f"evaluation_{dataset_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n -> Output directory: {output_dir}")
    
    # Evaluate on dataset
    results = evaluate_on_dataset(model, trainer, dataset_path, dataset_name)
    
    if results:
        # Analyze misclassifications
        analyze_misclassifications(results, output_dir, save_images=True, max_save=50)
        
        # Save detailed report
        save_detailed_report(results, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {model_folder}")
        print(f"Dataset: {dataset_name} ({results['total_images']} images)")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Errors: {len(results['misclassified_indices'])}")
        
        # Per-class accuracy
        print("\nPer-Class Performance:")
        cm = results['confusion_matrix']
        for i, emotion in enumerate(EMOTION_LABELS):
            if cm[i].sum() > 0:
                class_acc = cm[i, i] / cm[i].sum() * 100
                print(f" -> {emotion}: {class_acc:.1f}% ({cm[i, i]}/{cm[i].sum()})")
        
        print(f"\nResults saved in: {output_dir}")
        print(" -> misclassifications_*.csv - Details of all errors")
        print(" -> classification_report_*.txt - Full metrics report")
        print(" -> misclassified_images_*/ - Sample error images")

if __name__ == "__main__":
    main()