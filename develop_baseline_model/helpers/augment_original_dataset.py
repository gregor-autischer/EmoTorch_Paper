############################################
# Apply 28 augmentations to each emotion image
# 
# by Gregor Autischer (August 2025)
############################################

import torch
import kornia.augmentation as KA
from PIL import Image
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

# Configuration
SOURCE_DIR = "./FER-Original-Dataset"
OUTPUT_DIR = "./FER-Original-Dataset-Augmented"

EMOTIONS = ['anger', 'fear', 'calm', 'surprise']
DATASETS = ['FER', 'CKP']

def load_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((48, 48))
    return torch.from_numpy(np.array(img)).float() / 255.0

def save_image(tensor, output_path):
    if len(tensor.shape) == 3:
        tensor = tensor.squeeze(0)
    img_array = (tensor * 255).numpy().astype(np.uint8)
    Image.fromarray(img_array).save(output_path)

def add_occlusion(img_tensor, occlusion_type):
    h, w = 48, 48
    mask = torch.zeros_like(img_tensor)
    
    if occlusion_type == 'snowflakes':
        # Add white dots
        for _ in range(20):
            x, y = random.randint(2, w-3), random.randint(2, h-3)
            mask[y-1:y+2, x-1:x+2] = 0.7
            
    elif occlusion_type == 'dust':
        # Add gray particles
        for _ in range(60):
            x, y = random.randint(0, w-1), random.randint(0, h-1)
            mask[y, x] = 0.4
            
    elif occlusion_type == 'hair':
        # Add dark strands
        for _ in range(4):
            start_x = random.randint(0, w-1)
            for i in range(h//2):
                y = i
                x = start_x + int(2 * np.sin(i * 0.15))
                if 0 <= x < w:
                    mask[y, x] = 0.3
    
    return torch.clamp(img_tensor + mask, 0, 1)

def get_augmentations(img_tensor):
    # Add batch dimension for Kornia
    img_batch = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    augmentations = []
    
    # Define all 25 Kornia augmentations
    transforms = {
        # Geometric (7)
        'rotation': KA.RandomRotation(degrees=25, p=1.0),
        'flip': KA.RandomHorizontalFlip(p=1.0),
        'zoom_in': KA.RandomResizedCrop(size=(48, 48), scale=(0.6, 0.75), p=1.0),
        'zoom_out': KA.RandomResizedCrop(size=(48, 48), scale=(1.1, 1.3), p=1.0),
        'shear': KA.RandomShear(shear=(0.0, 0.0, 12.0, 12.0), p=1.0),
        'perspective': KA.RandomPerspective(distortion_scale=0.25, p=1.0),
        'elastic': KA.RandomElasticTransform(kernel_size=(33, 33), sigma=(5, 5), p=1.0),
        
        # Brightness/Contrast (7)
        'dark': KA.ColorJitter(brightness=(0.2, 0.4), p=1.0),
        'bright': KA.ColorJitter(brightness=(1.6, 2.2), p=1.0),
        'low_contrast': KA.ColorJitter(contrast=(0.2, 0.4), p=1.0),
        'high_contrast': KA.ColorJitter(contrast=(2.5, 3.5), p=1.0),
        'gamma_dark': KA.RandomGamma(gamma=(0.2, 0.5), p=1.0),
        'gamma_bright': KA.RandomGamma(gamma=(1.8, 2.8), p=1.0),
        'posterize': KA.RandomPosterize(bits=2, p=1.0),
        
        # Noise/Blur (5)
        'heavy_noise': KA.RandomGaussianNoise(mean=0.0, std=0.12, p=1.0),
        'light_noise': KA.RandomGaussianNoise(mean=0.0, std=0.04, p=1.0),
        'blur': KA.RandomGaussianBlur(kernel_size=(9, 9), sigma=(3.0, 5.0), p=1.0),
        'motion_left': KA.RandomMotionBlur(kernel_size=9, angle=(-10, 10), direction=(-1, -1), p=1.0),
        'motion_right': KA.RandomMotionBlur(kernel_size=9, angle=(-10, 10), direction=(1, 1), p=1.0),
        
        # Masking (2)
        'squares': KA.RandomErasing(scale=(0.08, 0.2), ratio=(0.7, 1.3), p=1.0),
        'bars': KA.RandomErasing(scale=(0.03, 0.08), ratio=(4.0, 8.0), p=1.0),
        
        # Effects (4)
        'solarize': KA.RandomSolarize(thresholds=0.2, additions=0.4, p=1.0),
        'equalize': KA.RandomEqualize(p=1.0),
        'clahe': KA.RandomClahe(clip_limit=(4.0, 8.0), p=1.0),
        'invert': KA.RandomInvert(p=1.0),
    }
    
    # Apply Kornia transforms
    for name, transform in transforms.items():
        try:
            aug = transform(img_batch.clone())
            aug = aug.squeeze(0).squeeze(0)
            augmentations.append((name, aug))
        except:
            pass
    
    # Add custom occlusions (3)
    for occ_type in ['snowflakes', 'dust', 'hair']:
        aug = add_occlusion(img_tensor.clone(), occ_type)
        augmentations.append((occ_type, aug))
    
    return augmentations

def augment_dataset():
    print(f" -> Creating augmented dataset in {OUTPUT_DIR}")
    
    # Create output directories
    for dataset in DATASETS:
        for emotion in EMOTIONS:
            (Path(OUTPUT_DIR) / dataset / emotion).mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_generated = 0
    
    for dataset in DATASETS:
        print(f" -> Processing {dataset} dataset")
        
        for emotion in EMOTIONS:
            source_dir = Path(SOURCE_DIR) / dataset / emotion
            output_dir = Path(OUTPUT_DIR) / dataset / emotion
            
            if not source_dir.exists():
                continue
            
            images = list(source_dir.glob("*.png"))
            
            for img_path in tqdm(images, desc=f"  {emotion}"):
                # Load image
                img_tensor = load_image(img_path)
                
                # Get all augmentations
                augmentations = get_augmentations(img_tensor)
                
                # Save augmented images
                base_name = img_path.stem
                for aug_name, aug_tensor in augmentations:
                    output_path = output_dir / f"{base_name}_{aug_name}.png"
                    save_image(aug_tensor, output_path)
                    total_generated += 1
                
                total_processed += 1
    
    print(f" -> Processed {total_processed} images, generated {total_generated} augmented images")

if __name__ == "__main__":
    augment_dataset()