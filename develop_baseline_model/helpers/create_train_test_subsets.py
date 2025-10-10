############################################
# Create train/validation subsets from FER-Original-Dataset
# 
# by Gregor Autischer (August 2025)
############################################

import os
import shutil
from pathlib import Path
import random

# uses absolute paths from repository root
# Gets the repository root directory (parent of helpers directory)
REPO_ROOT = Path(__file__).parent.parent.absolute()

# Main output directory for all train/validation data
MAIN_OUTPUT_DIR = REPO_ROOT / "TRAIN-VAL DATA"

FER_SOURCE_DIR = REPO_ROOT / "FER-Original-Dataset" / "FER"
CKP_SOURCE_DIR = REPO_ROOT / "FER-Original-Dataset" / "CKP"
TRAIN_OUTPUT_DIR = MAIN_OUTPUT_DIR / "FER-Training-Subset"
FER_VAL_OUTPUT_DIR = MAIN_OUTPUT_DIR / "FER-Validation-Subset"
CKP_VAL_OUTPUT_DIR = MAIN_OUTPUT_DIR / "CKP-Validation-Subset"

# Percentage of total images to use (100% = all images, 50% = half of all images)
# Change this value to use only a subset of the data (e.g., 50.0 for 50% of images)
PERCENTAGE_OF_DATA_TO_USE = 100.0  # Default: use all available images

TRAIN_SPLIT = 0.85  # 85% of FER data for training, 15% for validation (research standard)

def create_train_test_subsets():
    
    # Paths are already Path objects from configuration
    fer_source_path = FER_SOURCE_DIR
    ckp_source_path = CKP_SOURCE_DIR
    train_path = TRAIN_OUTPUT_DIR
    fer_val_path = FER_VAL_OUTPUT_DIR
    ckp_val_path = CKP_VAL_OUTPUT_DIR
    
    # Create main output directory and subdirectories
    MAIN_OUTPUT_DIR.mkdir(exist_ok=True)
    train_path.mkdir(parents=True, exist_ok=True)
    fer_val_path.mkdir(parents=True, exist_ok=True)
    ckp_val_path.mkdir(parents=True, exist_ok=True)
    
    emotions = ['anger', 'fear', 'calm', 'surprise']
    
    print(f"Creating train/validation subsets following research standards:")
    print(f"  Repository root: {REPO_ROOT}")
    print(f"  Output directory: {MAIN_OUTPUT_DIR}")
    print(f"  Source FER: {fer_source_path}")
    print(f"  Source CK+: {ckp_source_path}")
    print(f"  Using {PERCENTAGE_OF_DATA_TO_USE:.1f}% of available data")
    print(f"  FER split: {TRAIN_SPLIT*100:.0f}% train / {(1-TRAIN_SPLIT)*100:.0f}% validation")
    print(f"  CK+ data: all images go to separate validation folder")
    print()
    
    total_train_copied = 0
    total_val_from_fer = 0
    total_val_from_ckp = 0
    
    for emotion in emotions:
        fer_emotion_dir = fer_source_path / emotion
        ckp_emotion_dir = ckp_source_path / emotion
        train_emotion_dir = train_path / emotion
        fer_val_emotion_dir = fer_val_path / emotion
        ckp_val_emotion_dir = ckp_val_path / emotion
        
        # Create target emotion directories
        train_emotion_dir.mkdir(exist_ok=True)
        fer_val_emotion_dir.mkdir(exist_ok=True)
        ckp_val_emotion_dir.mkdir(exist_ok=True)
        
        # Process FER data
        fer_image_files = list(fer_emotion_dir.glob("*.png"))
        
        # Calculate how many images to use based on percentage setting
        fer_images_to_use = int(len(fer_image_files) * (PERCENTAGE_OF_DATA_TO_USE / 100.0))
        
        if fer_images_to_use > len(fer_image_files):
            fer_images_to_use = len(fer_image_files)
        
        # Randomly sample FER images based on percentage
        if fer_images_to_use > 0:
            fer_sampled_files = random.sample(fer_image_files, fer_images_to_use)
            
            # Split FER data into train and validation
            fer_train_count = int(fer_images_to_use * TRAIN_SPLIT)
            fer_val_count = fer_images_to_use - fer_train_count
            
            fer_train_files = fer_sampled_files[:fer_train_count]
            fer_val_files = fer_sampled_files[fer_train_count:]
            
            # Copy FER training files
            for i, source_file in enumerate(fer_train_files, 1):
                target_file = train_emotion_dir / f"{i}.png"  # Simple numbering for training
                shutil.copy2(source_file, target_file)
            
            # Copy FER validation files to FER-Validation folder
            for i, source_file in enumerate(fer_val_files, 1):
                target_file = fer_val_emotion_dir / f"{i}.png"  # Simple numbering for FER validation
                shutil.copy2(source_file, target_file)
            
            total_train_copied += fer_train_count
            total_val_from_fer += fer_val_count
        else:
            fer_train_count = 0
            fer_val_count = 0
        
        # Process CK+ data (all goes to separate CK+ validation folder)
        ckp_val_count = 0
        if ckp_emotion_dir.exists():
            ckp_image_files = list(ckp_emotion_dir.glob("*.png"))
            
            # Use percentage of available CK+ images
            ckp_images_to_use = int(len(ckp_image_files) * (PERCENTAGE_OF_DATA_TO_USE / 100.0))
            
            if ckp_images_to_use > 0:
                ckp_sampled_files = random.sample(ckp_image_files, ckp_images_to_use)
                
                # Copy all CK+ images to CKP-Validation folder
                for i, source_file in enumerate(ckp_sampled_files, 1):
                    target_file = ckp_val_emotion_dir / f"{i}.png"  # Simple numbering for CK+ validation
                    shutil.copy2(source_file, target_file)
                
                ckp_val_count = len(ckp_sampled_files)
                total_val_from_ckp += ckp_val_count
        
        print(f"  {emotion}: {fer_train_count} train (FER) | {fer_val_count} val (FER) | {ckp_val_count} val (CK+)")
    
    print(f"\nDataset creation complete!")
    print(f"Training subset (FER only): {total_train_copied} images -> {TRAIN_OUTPUT_DIR}")
    print(f"FER Validation subset: {total_val_from_fer} images -> {FER_VAL_OUTPUT_DIR}")
    print(f"CK+ Validation subset: {total_val_from_ckp} images -> {CKP_VAL_OUTPUT_DIR}")
    print(f"Total validation images: {total_val_from_fer + total_val_from_ckp}")
    
    # Verify structure
    print("\nVerifying training subset (FER data only):")
    for emotion in emotions:
        emotion_dir = train_path / emotion
        count = len(list(emotion_dir.glob("*.png")))
        print(f"  {emotion}: {count} images")
    
    print("\nVerifying FER validation subset:")
    for emotion in emotions:
        emotion_dir = fer_val_path / emotion
        count = len(list(emotion_dir.glob("*.png")))
        print(f"  {emotion}: {count} images")
    
    print("\nVerifying CK+ validation subset:")
    for emotion in emotions:
        emotion_dir = ckp_val_path / emotion
        count = len(list(emotion_dir.glob("*.png")))
        print(f"  {emotion}: {count} images")

if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)
    create_train_test_subsets()