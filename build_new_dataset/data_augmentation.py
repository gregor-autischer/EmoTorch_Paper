############################################
# This script applies 10 specific augmentations to the FER-New-Dataset to create FER-New-Dataset-Augmented
# 
# by Gregor Autischer (September 2025)
############################################

# Augmentations:
# 1. Rotation (Geometric)
# 2. Dark (Lighting)
# 3. High Contrast (Lighting)
# 4. Light Noise (Image Quality)
# 5. Blur (Image Quality)
# 6. Top Rectangle (Head Covering - hats, hijabs)
# 7. Top Left Diagonal (Head Covering)
# 8. Top Right Diagonal (Head Covering)
# 9. Forehead Bar (Head Covering - headbands)
# 10. Heavy Hair (Head Covering)

import torch
import kornia.augmentation as KA
from PIL import Image
import numpy as np
from pathlib import Path
import random
import csv
from tqdm import tqdm


# Configuration
SOURCE_DIR = Path("./FER-New-Dataset/FER-New-Dataset")
OUTPUT_DIR = Path("./FER-New-Dataset/FER-New-Dataset-Augmented")
SOURCE_CSV = Path("./FER-New-Dataset/dataset_new_attributs.csv")
OUTPUT_CSV = Path("./FER-New-Dataset/dataset_new_attributs.csv")

EMOTIONS = ['anger', 'fear', 'calm', 'surprise']
DATASETS = ['FER', 'CKP', 'RAF']

# Augmentation names
AUGMENTATION_NAMES = [
    'rotation',
    'dark',
    'high_contrast',
    'light_noise',
    'blur',
    'top_rectangle',
    'top_left_diagonal',
    'top_right_diagonal',
    'forehead_bar',
    'heavy_hair'
]


def load_image(image_path):

    img = Image.open(image_path).convert('L')
    # Ensure 48x48 size
    if img.size != (48, 48):
        img = img.resize((48, 48))
    return torch.from_numpy(np.array(img)).float() / 255.0


def save_image(tensor, output_path):

    if len(tensor.shape) == 3:
        tensor = tensor.squeeze(0)
    img_array = (tensor * 255).numpy().astype(np.uint8)
    Image.fromarray(img_array).save(output_path)


def add_head_covering_occlusion(img_tensor, occlusion_type):
    # Add occlusions simulating head coverings (hats, hijabs, headbands, hair)

    h, w = 48, 48
    img_copy = img_tensor.clone()

    if occlusion_type == 'top_rectangle':
        # Covers 25-42% from top (simulates hats, hijabs, caps)
        coverage_percent = random.uniform(0.25, 0.42)
        height = int(h * coverage_percent)
        # Dark rectangle from top
        img_copy[:height, :] = torch.clamp(img_copy[:height, :] * 0.2, 0, 1)

    elif occlusion_type == 'top_left_diagonal':
        # Diagonal coverage from top-left corner
        for y in range(h):
            for x in range(w):
                # Create diagonal mask: top-left triangle
                if y < (h // 2) and x < (w - y):
                    img_copy[y, x] = torch.clamp(img_copy[y, x] * 0.25, 0, 1)

    elif occlusion_type == 'top_right_diagonal':
        # Diagonal coverage from top-right corner
        for y in range(h):
            for x in range(w):
                # Create diagonal mask: top-right triangle
                if y < (h // 2) and x > y:
                    img_copy[y, x] = torch.clamp(img_copy[y, x] * 0.25, 0, 1)

    elif occlusion_type == 'forehead_bar':
        # Horizontal bar 3-6px tall (headbands, bandanas)
        bar_height = random.randint(3, 6)
        start_y = random.randint(h // 6, h // 3)  # Position on forehead area
        end_y = min(start_y + bar_height, h)
        # Dark bar
        img_copy[start_y:end_y, :] = torch.clamp(img_copy[start_y:end_y, :] * 0.15, 0, 1)

    elif occlusion_type == 'heavy_hair':
        # 8-15 thick, dark hair strands from top of head
        num_strands = random.randint(8, 15)
        for _ in range(num_strands):
            start_x = random.randint(0, w - 1)
            # Create wavy strand
            for i in range(h // 2):  # Hair from top to middle
                y = i
                # Wavy pattern
                x = start_x + int(3 * np.sin(i * 0.2 + random.random() * 2))
                if 0 <= x < w and 0 <= y < h:
                    # Make it thick (2-3 pixels wide)
                    thickness = random.randint(2, 3)
                    for dx in range(-thickness // 2, thickness // 2 + 1):
                        if 0 <= x + dx < w:
                            # Dark hair strand
                            img_copy[y, x + dx] = torch.clamp(img_copy[y, x + dx] * 0.2, 0, 1)

    return img_copy


def get_augmentations(img_tensor):
    # Apply all 10 augmentations to the image

    # Add batch and channel dimensions for Kornia: [1, 1, H, W]
    img_batch = img_tensor.unsqueeze(0).unsqueeze(0)

    augmentations = []

    # 1. Rotation - Random rotation up to ±25 degrees
    try:
        rotation = KA.RandomRotation(degrees=25, p=1.0)
        aug = rotation(img_batch.clone()).squeeze(0).squeeze(0)
        augmentations.append(('rotation', aug))
    except Exception as e:
        print(f"Warning: Rotation failed - {e}")

    # 2. Dark - Reduced brightness (20-40%) for low-light conditions
    try:
        dark = KA.ColorJitter(brightness=(0.2, 0.4), p=1.0)
        aug = dark(img_batch.clone()).squeeze(0).squeeze(0)
        augmentations.append(('dark', aug))
    except Exception as e:
        print(f"Warning: Dark failed - {e}")

    # 3. High Contrast - Increased contrast (2.5-3.5x) for strong lighting
    try:
        high_contrast = KA.ColorJitter(contrast=(2.5, 3.5), p=1.0)
        aug = high_contrast(img_batch.clone()).squeeze(0).squeeze(0)
        augmentations.append(('high_contrast', aug))
    except Exception as e:
        print(f"Warning: High contrast failed - {e}")

    # 4. Light Noise - Gaussian noise (σ=0.04) simulating sensor noise
    try:
        light_noise = KA.RandomGaussianNoise(mean=0.0, std=0.04, p=1.0)
        aug = light_noise(img_batch.clone()).squeeze(0).squeeze(0)
        augmentations.append(('light_noise', aug))
    except Exception as e:
        print(f"Warning: Light noise failed - {e}")

    # 5. Blur - Gaussian blur (kernel 9x9, σ=3-5) for motion/focus issues
    try:
        blur = KA.RandomGaussianBlur(kernel_size=(9, 9), sigma=(3.0, 5.0), p=1.0)
        aug = blur(img_batch.clone()).squeeze(0).squeeze(0)
        augmentations.append(('blur', aug))
    except Exception as e:
        print(f"Warning: Blur failed - {e}")

    # 6-10. Head Covering Occlusions
    head_covering_types = [
        'top_rectangle',
        'top_left_diagonal',
        'top_right_diagonal',
        'forehead_bar',
        'heavy_hair'
    ]

    for occ_type in head_covering_types:
        try:
            aug = add_head_covering_occlusion(img_tensor.clone(), occ_type)
            augmentations.append((occ_type, aug))
        except Exception as e:
            print(f"Warning: {occ_type} failed - {e}")

    return augmentations


def create_output_directories():

    for dataset in DATASETS:
        for emotion in EMOTIONS:
            output_path = OUTPUT_DIR / dataset / emotion
            output_path.mkdir(parents=True, exist_ok=True)


def augment_images():

    print("Starting data augmentation for FER-New-Dataset...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Create output directories
    create_output_directories()

    total_processed = 0
    total_generated = 0

    # Process each dataset
    for dataset in DATASETS:
        dataset_path = SOURCE_DIR / dataset

        if not dataset_path.exists():
            print(f"Warning: Dataset {dataset} not found, skipping...")
            continue

        print(f"\nProcessing {dataset} dataset...")

        for emotion in EMOTIONS:
            source_dir = dataset_path / emotion
            output_dir = OUTPUT_DIR / dataset / emotion

            if not source_dir.exists():
                print(f"  Warning: {emotion} folder not found in {dataset}, skipping...")
                continue

            # Get all images
            images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))

            if not images:
                print(f"  Warning: No images found in {source_dir}")
                continue

            # Process each image
            for img_path in tqdm(images, desc=f"  {emotion}"):
                try:
                    # Load image
                    img_tensor = load_image(img_path)

                    # Get all augmentations
                    augmentations = get_augmentations(img_tensor)

                    # Save augmented images
                    base_name = img_path.stem
                    extension = img_path.suffix  # Keep original extension

                    for aug_name, aug_tensor in augmentations:
                        output_path = output_dir / f"{base_name}_{aug_name}{extension}"
                        save_image(aug_tensor, output_path)
                        total_generated += 1

                    total_processed += 1

                except Exception as e:
                    print(f"    Error processing {img_path}: {e}")

    print(f"\n=== Augmentation Complete ===")
    print(f"Total images processed: {total_processed}")
    print(f"Total augmented images generated: {total_generated}")


def update_csv_with_augmented_data():
    # Add entries for augmented images to dataset_new_attributs.csv

    print("\nUpdating CSV with augmented image entries...")

    # Read existing CSV
    existing_rows = []
    with open(SOURCE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            existing_rows.append(row)

    print(f"Read {len(existing_rows)} existing entries")

    # Generate augmented entries
    new_rows = []
    for row in existing_rows:
        # Skip already augmented entries (anything that's not 'original')
        if row['augmented'] != 'original':
            continue

        original_path = row['image_path']

        # For each augmentation type, create a new row
        for aug_name in AUGMENTATION_NAMES:
            # Convert path from FER-New-Dataset/FER-New-Dataset/... to FER-New-Dataset/FER-New-Dataset-Augmented/...
            augmented_path = original_path.replace(
                'FER-New-Dataset/FER-New-Dataset',
                'FER-New-Dataset/FER-New-Dataset-Augmented'
            )

            # Add augmentation suffix to filename
            path_obj = Path(augmented_path)
            augmented_path = str(path_obj.parent / f"{path_obj.stem}_{aug_name}{path_obj.suffix}")

            # Create new row with the augmentation type in the 'augmented' column
            new_row = {
                'image_path': augmented_path,
                'emotion': row['emotion'],
                'race': row['race'],
                'gender': row['gender'],
                'age': row['age'],
                'augmented': aug_name,  # Store the augmentation type
                'usage': ''
            }
            new_rows.append(new_row)

    print(f"Generated {len(new_rows)} augmented entries")

    # Append to CSV
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(new_rows)

    print(f"CSV updated: {OUTPUT_CSV}")


def main():
    print("=" * 60)
    print("FER-New-Dataset Augmentation Script")
    print("=" * 60)

    # Step 1: Augment images
    augment_images()

    # Step 2: Update CSV
    update_csv_with_augmented_data()

    print("\n" + "=" * 60)
    print("All tasks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
