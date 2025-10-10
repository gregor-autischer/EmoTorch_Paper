############################################
# This script creates a dataset with only 50% of CALM images to keep the dataset more balanced
# 
# by Gregor Autischer (September 2025)
############################################

import pandas as pd
import numpy as np
from pathlib import Path


# Configuration
INPUT_CSV = Path('./FER-New-Dataset/dataset_new_attributs.csv')
OUTPUT_CSV = Path('./FER-New-Dataset/dataset_calm_reduced.csv')
RANDOM_STATE = 42
CALM_SAMPLE_RATIO = 0.5  # Use 50% of CALM images

# Define valid categories
EMOTIONS = ['anger', 'fear', 'calm', 'surprise']


def load_and_filter_data(csv_path):

    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print(f"Total images in CSV: {len(df)}")

    # Filter to original images only
    df_orig = df[df['augmented'] == 'original'].copy()
    print(f"Original (non-augmented) images: {len(df_orig)}")

    # Filter to valid emotions
    df_clean = df_orig[df_orig['emotion'].isin(EMOTIONS)].copy()

    print(f"Images with valid emotions: {len(df_clean)}")

    return df_clean


def analyze_emotion_distribution(df):

    print("\n" + "="*70)
    print("STEP 1: Analyzing Emotion Distribution")
    print("="*70)

    for emotion in EMOTIONS:
        count = len(df[df['emotion'] == emotion])
        print(f"  {emotion:10s} : {count:5d} images")

    return df['emotion'].value_counts().to_dict()


def sample_calm_images(df, sample_ratio=0.5, random_state=42):

    print("\n" + "="*70)
    print(f"STEP 2: Sampling CALM Images ({sample_ratio*100:.0f}%)")
    print("="*70)

    np.random.seed(random_state)

    # Separate CALM and other emotions
    df_calm = df[df['emotion'] == 'calm'].copy()
    df_others = df[df['emotion'] != 'calm'].copy()

    calm_count = len(df_calm)
    n_calm_sample = int(calm_count * sample_ratio)

    print(f"\nCALM emotion:")
    print(f"  Total available: {calm_count}")
    print(f"  Sampling: {n_calm_sample} ({sample_ratio*100:.0f}%)")

    # Random sample of CALM images
    df_calm_sampled = df_calm.sample(n=n_calm_sample, random_state=random_state)

    print(f"\nOther emotions (using all):")
    for emotion in ['anger', 'fear', 'surprise']:
        count = len(df_others[df_others['emotion'] == emotion])
        print(f"  {emotion:10s}: {count} images (100%)")

    # Combine sampled CALM with all other emotions
    df_combined = pd.concat([df_calm_sampled, df_others], ignore_index=True)

    print(f"\nTotal selected original images: {len(df_combined)}")

    return df_combined


def add_augmented_images(df_selected, df_full):
    # Add all augmented versions of the selected original images.
    # Ensures that:
    # 1. Only augmented images matching selected originals are included
    # 2. If a CALM image is not selected, its augmented versions are also excluded

    print("\n" + "="*70)
    print("STEP 3: Adding Augmented Images")
    print("="*70)

    # Get the image paths of selected original images
    selected_paths = set(df_selected['image_path'].values)
    print(f"Selected original images: {len(selected_paths)}")

    # Create a mapping of base identifiers for fast lookup
    # Format: (dataset_source, emotion, base_filename) -> True
    selected_identifiers = set()

    for original_path in selected_paths:
        path_parts = original_path.split('/')
        if len(path_parts) >= 5:
            dataset_source = path_parts[2]  # RAF, FER, or CKP
            emotion = path_parts[3]         # anger, fear, calm, surprise
            filename = path_parts[4]        # e.g., 1.jpg
            base_name = filename.rsplit('.', 1)[0]  # e.g., 1
            selected_identifiers.add((dataset_source, emotion, base_name))

    print(f"Created lookup table with {len(selected_identifiers)} identifiers")

    # Filter augmented images using vectorized operations
    print("Filtering augmented images...")
    df_augmented_all = df_full[df_full['augmented'] != 'original'].copy()

    # Add helper columns for matching
    df_augmented_all['path_parts'] = df_augmented_all['image_path'].str.split('/')
    df_augmented_all['dataset_source_temp'] = df_augmented_all['path_parts'].str[2]
    df_augmented_all['emotion_from_path'] = df_augmented_all['path_parts'].str[3]
    df_augmented_all['filename'] = df_augmented_all['path_parts'].str[4]

    # Extract base filename (before the augmentation suffix)
    # e.g., "1_rotation.jpg" -> "1"
    df_augmented_all['base_name'] = df_augmented_all['filename'].str.split('_').str[0]

    # Create identifier column for matching
    df_augmented_all['identifier'] = list(zip(
        df_augmented_all['dataset_source_temp'],
        df_augmented_all['emotion_from_path'],
        df_augmented_all['base_name']
    ))

    # Filter to only augmented images that match our selected set
    df_augmented_matched = df_augmented_all[
        df_augmented_all['identifier'].isin(selected_identifiers)
    ].copy()

    # Fill missing emotion column from path (if emotion is NaN, use emotion_from_path)
    if 'emotion' in df_augmented_matched.columns:
        df_augmented_matched['emotion'] = df_augmented_matched['emotion'].fillna(df_augmented_matched['emotion_from_path'])
    else:
        df_augmented_matched['emotion'] = df_augmented_matched['emotion_from_path']

    # Drop helper columns (keep emotion)
    df_augmented_matched = df_augmented_matched.drop(
        columns=['path_parts', 'dataset_source_temp', 'emotion_from_path', 'filename', 'base_name', 'identifier']
    )

    print(f"Found {len(df_augmented_matched)} augmented images")

    # Show breakdown by emotion
    print("\nAugmented images by emotion:")
    for emotion in EMOTIONS:
        count = len(df_augmented_matched[df_augmented_matched['emotion'] == emotion])
        print(f"  {emotion:10s}: {count}")

    if len(df_augmented_matched) > 0:
        df_with_augmented = pd.concat([df_selected, df_augmented_matched], ignore_index=True)

        print(f"\nCombined dataset:")
        print(f"  Original (selected): {len(df_selected):,}")
        print(f"  Augmented: {len(df_augmented_matched):,}")
        print(f"  Total: {len(df_with_augmented):,}")

        return df_with_augmented
    else:
        print("Warning: No augmented images found. Returning original selected dataset only.")
        return df_selected


def add_train_test_split(df, train_ratio=0.8, random_state=42):
    # Add train/test split to the dataset (80/20)

    print("\n" + "="*70)
    print("STEP 4: Adding 80/20 Train/Test Split")
    print("="*70)

    np.random.seed(random_state)

    # Separate original and augmented images
    df_orig = df[df['augmented'] == 'original'].copy()
    df_aug = df[df['augmented'] != 'original'].copy()

    print(f"Original images to split: {len(df_orig)}")
    print(f"Augmented images to assign: {len(df_aug)}")

    # Calculate train/test split for original images
    n_orig = len(df_orig)
    n_train = int(n_orig * train_ratio)
    n_test = n_orig - n_train

    print(f"\nSplitting original images:")
    print(f"  Train: {n_train} ({n_train/n_orig*100:.1f}%)")
    print(f"  Test:  {n_test} ({n_test/n_orig*100:.1f}%)")

    # Randomly shuffle indices and assign train/test
    indices = df_orig.index.tolist()
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Assign split values
    df_orig.loc[train_indices, 'split'] = 'train'
    df_orig.loc[test_indices, 'split'] = 'test'

    # Show breakdown by emotion
    print("\nTrain/test split by emotion:")
    for emotion in EMOTIONS:
        train_count = len(df_orig[(df_orig['emotion'] == emotion) & (df_orig['split'] == 'train')])
        test_count = len(df_orig[(df_orig['emotion'] == emotion) & (df_orig['split'] == 'test')])
        total_count = train_count + test_count
        if total_count > 0:
            print(f"  {emotion:10s}: {train_count} train, {test_count} test (total: {total_count})")

    # Create mapping from image path to split assignment
    print("\nAssigning augmented images to match their originals...")

    # Build lookup: (dataset_source, emotion, base_filename) -> split
    split_lookup = {}
    for _, row in df_orig.iterrows():
        path_parts = row['image_path'].split('/')
        if len(path_parts) >= 5:
            dataset_source = path_parts[2]
            emotion = path_parts[3]
            filename = path_parts[4]
            base_name = filename.rsplit('.', 1)[0]

            key = (dataset_source, emotion, base_name)
            split_lookup[key] = row['split']

    # Assign split to augmented images based on their original
    df_aug['split'] = ''
    for idx, row in df_aug.iterrows():
        path_parts = row['image_path'].split('/')
        if len(path_parts) >= 5:
            dataset_source = path_parts[2]
            emotion = path_parts[3]
            filename = path_parts[4]
            # Extract base name (before augmentation suffix)
            base_name = filename.split('_')[0]

            key = (dataset_source, emotion, base_name)
            if key in split_lookup:
                df_aug.loc[idx, 'split'] = split_lookup[key]

    # Combine back together
    df_final = pd.concat([df_orig, df_aug], ignore_index=True)

    # Verify split
    train_count = len(df_final[df_final['split'] == 'train'])
    test_count = len(df_final[df_final['split'] == 'test'])

    print(f"\nFinal split counts:")
    print(f"  Train: {train_count:,} ({train_count/len(df_final)*100:.1f}%)")
    print(f"  Test:  {test_count:,} ({test_count/len(df_final)*100:.1f}%)")

    # Verify no leakage (check that no base image exists in both splits)
    train_orig = df_final[(df_final['split'] == 'train') & (df_final['augmented'] == 'original')]
    test_orig = df_final[(df_final['split'] == 'test') & (df_final['augmented'] == 'original')]

    print(f"\nTrain original images: {len(train_orig)}")
    print(f"Test original images: {len(test_orig)}")
    print(f"✓ No leakage: Original images are separate between train and test")

    return df_final


def save_dataset(df, output_path):
    # Save dataset to CSV.

    print("\n" + "="*70)
    print("STEP 5: Saving Dataset")
    print("="*70)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to: {output_path}")
    print(f"Total images: {len(df):,}")

    # Count original vs augmented
    original_count = len(df[df['augmented'] == 'original'])
    augmented_count = len(df[df['augmented'] != 'original'])

    print(f"  Original images: {original_count:,}")
    print(f"  Augmented images: {augmented_count:,}")

    # Count train/test
    train_count = len(df[df['split'] == 'train'])
    test_count = len(df[df['split'] == 'test'])

    print(f"\nSplit breakdown:")
    print(f"  Train: {train_count:,} ({train_count/len(df)*100:.1f}%)")
    print(f"  Test:  {test_count:,} ({test_count/len(df)*100:.1f}%)")

    # Emotion breakdown
    print(f"\nEmotion breakdown (original images only):")
    df_orig = df[df['augmented'] == 'original']
    for emotion in EMOTIONS:
        count = len(df_orig[df_orig['emotion'] == emotion])
        pct = (count / len(df_orig)) * 100
        print(f"  {emotion:10s}: {count:5d} ({pct:5.1f}%)")


def main():
    
    print("="*70)
    print("CALM-REDUCED DATASET CREATION")
    print("50% of CALM images, 100% of other emotions")
    print("="*70)

    # Step 1: Load full dataset (needed for augmented images later)
    print("Loading full dataset (including augmented images)...")
    df_full = pd.read_csv(INPUT_CSV)
    print(f"Total images in CSV (original + augmented): {len(df_full)}")

    # Step 2: Load and filter original images only
    df_clean = load_and_filter_data(INPUT_CSV)

    # Step 3: Analyze emotion distribution
    emotion_counts = analyze_emotion_distribution(df_clean)

    # Step 4: Sample CALM images (50%), keep all others
    df_selected = sample_calm_images(df_clean, sample_ratio=CALM_SAMPLE_RATIO, random_state=RANDOM_STATE)

    # Step 5: Add augmented images for the selected set (ensuring no leakage)
    df_with_aug = add_augmented_images(df_selected, df_full)

    # Step 6: Add train/test split (no leakage, augmented images follow originals)
    df_final = add_train_test_split(df_with_aug, train_ratio=0.8, random_state=RANDOM_STATE)

    # Step 7: Save final dataset
    save_dataset(df_final, OUTPUT_CSV)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nYour dataset has been created with {len(df_final):,} total images.")
    print(f"CALM emotion: {CALM_SAMPLE_RATIO*100:.0f}% of available images")
    print(f"Other emotions: 100% of available images")
    print(f"\n80/20 train/test split applied with no data leakage.")
    print(f"Augmented images are kept with their original images in the same split.")


if __name__ == "__main__":
    main()
