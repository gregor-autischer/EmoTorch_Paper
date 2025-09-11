############################################
# Create 4-emotion dataset structure from FER & CK+
# 
# by Gregor Autischer (August 2025)
############################################

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
FER_SOURCE = "./FER-Datasets/FER2013img"
CKP_SOURCE = "./FER-Datasets/CK+"
OUTPUT_DIR = "./FER-Original-Dataset"

# Emotion mappings
FER_MAPPING = {'angry': 'anger', 'fear': 'fear', 'neutral': 'calm', 'surprise': 'surprise'}
CKP_MAPPING = {'anger': 'anger', 'fear': 'fear', 'happy': 'calm', 'surprise': 'surprise'}

def copy_images(source_emotion, target_emotion, source_base, target_base):
    source_dir = Path(source_base) / source_emotion
    target_dir = Path(target_base) / target_emotion
    
    if not source_dir.exists():
        return 0
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get and sort images
    images = sorted(list(source_dir.glob('*.png')) + list(source_dir.glob('*.jpg')))
    
    # Copy with sequential numbering
    for i, img in enumerate(images, 1):
        shutil.copy2(img, target_dir / f"{i}.png")
    
    return len(images)

def create_dataset():
    # Check if sources exist
    if not Path(FER_SOURCE).exists():
        print(f" XX Error: FER source not found at {FER_SOURCE}")
        return
    
    print(f" -> Creating 4-emotion dataset in {OUTPUT_DIR}")
    
    # Create directories
    fer_dir = Path(OUTPUT_DIR) / "FER"
    ckp_dir = Path(OUTPUT_DIR) / "CKP"
    fer_dir.mkdir(parents=True, exist_ok=True)
    ckp_dir.mkdir(parents=True, exist_ok=True)
    
    # Process FER dataset
    print(" -> Processing FER images")
    fer_total = 0
    for source, target in tqdm(FER_MAPPING.items(), desc="FER emotions"):
        count = copy_images(source, target, FER_SOURCE, fer_dir)
        fer_total += count
    
    # Process CK+ dataset if exists
    ckp_total = 0
    if Path(CKP_SOURCE).exists():
        print(" -> Processing CK+ images")
        for source, target in tqdm(CKP_MAPPING.items(), desc="CK+ emotions"):
            count = copy_images(source, target, CKP_SOURCE, ckp_dir)
            ckp_total += count
    
    print(f" -> Created dataset: FER={fer_total} images, CK+={ckp_total} images")

if __name__ == "__main__":
    create_dataset()