"""
FER_CKP_to_final_structure.py

This script copies FER and CKP datasets from FER-Original-Dataset to FER-New-Dataset
and transfers demographic information from faces_gender_race.csv to dataset_new_attributs.csv
with proper mappings for race, gender, and age categories.
"""

import os
import shutil
import csv
from pathlib import Path


# Race mapping: Old -> New
RACE_MAPPING = {
    'White': 'Caucasian',
    'Black': 'African-American',
    'Indian': 'Asian',
    'Asian': 'Asian'
}

# Gender mapping: Capitalize -> lowercase
GENDER_MAPPING = {
    'Male': 'male',
    'Female': 'female'
}

# Age mapping: Old ranges -> New ranges
AGE_MAPPING = {
    '0-2': '0-3',
    '3-9': '4-19',
    '10-19': '4-19',
    '20-29': '20-39',
    '30-39': '20-39',
    '40-49': '40-69',
    '50-59': '40-69',
    '60-69': '40-69',
    '70+': '70+'
}

# Paths
SOURCE_DATASET_PATH = Path('./FER-Original-Dataset')
DEST_DATASET_PATH = Path('./FER-New-Dataset/FER-New-Dataset')
OLD_CSV_PATH = Path('./baseline_model/model/faces_gender_race.csv')
NEW_CSV_PATH = Path('./FER-New-Dataset/dataset_new_attributs.csv')


def copy_datasets():
    """Copy FER and CKP folders from FER-Original-Dataset to FER-New-Dataset"""
    datasets = ['FER', 'CKP']

    for dataset in datasets:
        source = SOURCE_DATASET_PATH / dataset
        dest = DEST_DATASET_PATH / dataset

        if not source.exists():
            print(f"Warning: Source {source} does not exist, skipping...")
            continue

        if dest.exists():
            print(f"Destination {dest} already exists, skipping copy...")
            continue

        print(f"Copying {dataset} dataset...")
        shutil.copytree(source, dest)
        print(f"  Copied {source} -> {dest}")

    print("\nDataset copying complete!")


def process_demographic_data():
    """Process demographic data from old CSV to new CSV format"""
    print("\nProcessing demographic data...")

    # Read old CSV
    old_data = {}
    with open(OLD_CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_path = row['img_path']
            old_data[old_path] = {
                'race4': row['race4'],
                'gender': row['gender'],
                'age': row['age']
            }

    print(f"Read {len(old_data)} entries from {OLD_CSV_PATH}")

    # Process each entry and create new CSV rows
    csv_rows = []
    processed_count = 0
    skipped_count = 0

    for old_path, attributes in old_data.items():
        # Skip augmented images (only process non-augmented)
        if 'Augmented' in old_path:
            skipped_count += 1
            continue

        # Convert old path to new path
        # Old: FER-Original-Dataset/FER/anger/3975.png
        # New: FER-New-Dataset/FER-New-Dataset/FER/anger/3975.png
        new_path = old_path.replace('FER-Original-Dataset', 'FER-New-Dataset/FER-New-Dataset')

        # Extract emotion from path (emotion is the second-to-last component)
        path_parts = Path(old_path).parts
        emotion = path_parts[-2]  # e.g., 'anger', 'calm', 'fear', 'surprise'

        # Map race4 to new race categories
        race4 = attributes['race4']
        race = RACE_MAPPING.get(race4, race4)  # Use original if not in mapping

        # Map gender to lowercase
        gender_old = attributes['gender']
        gender = GENDER_MAPPING.get(gender_old, gender_old.lower())

        # Map age ranges
        age_old = attributes['age']
        age = AGE_MAPPING.get(age_old, age_old)  # Use original if not in mapping

        # Create CSV row
        # Columns: image_path, emotion, race, gender, age, augmented, usage
        csv_rows.append([
            new_path,
            emotion,
            race,
            gender,
            age,
            'original',  # augmented - 'original' for non-augmented images
            ''     # usage - left empty
        ])

        processed_count += 1

    # Append to new CSV
    print(f"Writing {len(csv_rows)} entries to {NEW_CSV_PATH}...")

    # Check if CSV exists and has headers
    csv_exists = NEW_CSV_PATH.exists() and NEW_CSV_PATH.stat().st_size > 0

    with open(NEW_CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write headers if file doesn't exist or is empty
        if not csv_exists:
            writer.writerow(['image_path', 'emotion', 'race', 'gender', 'age', 'augmented', 'usage'])

        writer.writerows(csv_rows)

    print("\n=== Processing Complete ===")
    print(f"Total entries processed: {processed_count}")
    print(f"Augmented entries skipped: {skipped_count}")
    print(f"CSV file updated: {NEW_CSV_PATH}")


def main():
    """Main function"""
    print("Starting FER and CKP dataset migration...\n")

    # Step 1: Copy datasets
    copy_datasets()

    # Step 2: Process demographic data
    process_demographic_data()

    print("\n=== All tasks complete ===")


if __name__ == "__main__":
    main()
