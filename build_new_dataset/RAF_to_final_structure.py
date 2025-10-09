"""
RAF_to_final_structure.py

This script processes RAF-DB dataset images and metadata into the final FER-New-Dataset structure.
It copies aligned images to emotion-specific folders, renames them sequentially, and creates
a CSV file with image paths and demographic attributes.
"""

import os
import shutil
import csv
from pathlib import Path
from PIL import Image


# Emotion mapping: RAF-DB labels to our emotion categories
EMOTION_MAPPING = {
    1: 'surprise',
    2: 'fear',
    6: 'anger',
    4: 'calm',  # happiness -> calm
    7: 'calm'   # neutral -> calm
}

# Gender mapping
GENDER_MAPPING = {
    0: 'male',
    1: 'female',
    2: 'unsure'
}

# Race mapping
RACE_MAPPING = {
    0: 'Caucasian',
    1: 'African-American',
    2: 'Asian'
}

# Age mapping
AGE_MAPPING = {
    0: '0-3',
    1: '4-19',
    2: '20-39',
    3: '40-69',
    4: '70+'
}

# Paths
RAF_DB_PATH = Path('./FER-Datasets/RAF-DB')
EMOTION_LABEL_FILE = RAF_DB_PATH / 'EmoLabel' / 'list_patition_label.txt'
MANUAL_ATTRI_DIR = RAF_DB_PATH / 'Annotation' / 'manual'
ALIGNED_IMAGES_DIR = RAF_DB_PATH / 'Image' / 'aligned'

FINAL_DATASET_PATH = Path('./FER-New-Dataset/FER-New-Dataset/RAF')
CSV_FILE_PATH = Path('./FER-New-Dataset/dataset_new_attributs.csv')


def read_emotion_labels():
    """Read emotion labels from list_patition_label.txt"""
    emotion_labels = {}
    with open(EMOTION_LABEL_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_name = parts[0]
                emotion_label = int(parts[1])
                emotion_labels[image_name] = emotion_label
    return emotion_labels


def read_manual_attributes(image_name):
    """Read gender, race, and age from manual attribute file"""
    # Convert image name to attribute file name
    # e.g., train_00001.jpg -> train_00001_manu_attri.txt
    base_name = image_name.replace('.jpg', '')
    attri_file = MANUAL_ATTRI_DIR / f"{base_name}_manu_attri.txt"

    if not attri_file.exists():
        return None, None, None

    with open(attri_file, 'r') as f:
        lines = f.readlines()

    # Skip first 5 lines (landmarks), then read gender, race, age
    if len(lines) >= 8:
        gender_val = int(lines[5].strip())
        race_val = int(lines[6].strip())
        age_val = int(lines[7].strip())

        gender = GENDER_MAPPING.get(gender_val, 'unknown')
        race = RACE_MAPPING.get(race_val, 'unknown')
        age = AGE_MAPPING.get(age_val, 'unknown')

        return gender, race, age

    return None, None, None


def create_directory_structure():
    """Create the emotion folders in the final dataset structure"""
    emotions = ['anger', 'calm', 'fear', 'surprise']
    for emotion in emotions:
        emotion_dir = FINAL_DATASET_PATH / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)


def initialize_csv():
    """Initialize the CSV file with headers if it doesn't exist"""
    CSV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and has content
    if CSV_FILE_PATH.exists() and CSV_FILE_PATH.stat().st_size > 0:
        return

    # Create new CSV with headers
    with open(CSV_FILE_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'emotion', 'race', 'gender', 'age', 'augmented', 'usage'])


def process_raf_images():
    """Main processing function"""
    print("Starting RAF-DB to final structure conversion...")

    # Read emotion labels
    print("Reading emotion labels...")
    emotion_labels = read_emotion_labels()
    print(f"Found {len(emotion_labels)} labeled images")

    # Create directory structure
    print("Creating directory structure...")
    create_directory_structure()

    # Initialize CSV
    print("Initializing CSV file...")
    initialize_csv()

    # Counter for each emotion to sequentially name files
    emotion_counters = {
        'anger': 1,
        'calm': 1,
        'fear': 1,
        'surprise': 1
    }

    # Track which emotion folders already have images
    for emotion in emotion_counters.keys():
        emotion_dir = FINAL_DATASET_PATH / emotion
        existing_images = list(emotion_dir.glob('*.jpg'))
        if existing_images:
            # Get the highest number
            max_num = max([int(img.stem) for img in existing_images])
            emotion_counters[emotion] = max_num + 1

    # Process each image
    csv_rows = []
    processed_count = 0
    skipped_count = 0

    for image_name, emotion_label in emotion_labels.items():
        # Check if this emotion is one we want
        if emotion_label not in EMOTION_MAPPING:
            skipped_count += 1
            continue

        emotion = EMOTION_MAPPING[emotion_label]

        # Source image path
        # Image names in label file are like "train_00001.jpg"
        # Aligned images are like "train_00001_aligned.jpg"
        aligned_image_name = image_name.replace('.jpg', '_aligned.jpg')
        source_path = ALIGNED_IMAGES_DIR / aligned_image_name

        if not source_path.exists():
            print(f"Warning: Image not found: {source_path}")
            skipped_count += 1
            continue

        # Get manual attributes
        gender, race, age = read_manual_attributes(image_name)

        # Destination path with sequential numbering
        dest_filename = f"{emotion_counters[emotion]}.jpg"
        dest_path = FINAL_DATASET_PATH / emotion / dest_filename

        # Load image, convert to grayscale, resize to 48x48, and save
        img = Image.open(source_path)
        img_gray = img.convert('L')
        img_resized = img_gray.resize((48, 48), Image.LANCZOS)
        img_resized.save(dest_path, 'JPEG')

        # Prepare CSV row
        relative_image_path = f"FER-New-Dataset/FER-New-Dataset/RAF/{emotion}/{dest_filename}"
        csv_rows.append([
            relative_image_path,
            emotion,
            race if race else '',
            gender if gender else '',
            age if age else '',
            'original',  # augmented
            ''     # usage (to be filled later)
        ])

        # Increment counter
        emotion_counters[emotion] += 1
        processed_count += 1

        if processed_count % 100 == 0:
            print(f"Processed {processed_count} images...")

    # Append to CSV
    print("Writing to CSV...")
    with open(CSV_FILE_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print("\n=== Processing Complete ===")
    print(f"Total images processed: {processed_count}")
    print(f"Images skipped: {skipped_count}")
    print("\nImages per emotion:")
    for emotion, count in emotion_counters.items():
        actual_count = count - 1  # Subtract 1 since counter is at next position
        print(f"  {emotion}: {actual_count}")
    print(f"\nCSV file updated: {CSV_FILE_PATH}")


if __name__ == "__main__":
    process_raf_images()
