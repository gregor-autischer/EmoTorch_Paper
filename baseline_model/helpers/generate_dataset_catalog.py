############################################
# Generate a CSV catalog containing all images from the baseline dataset,
# their annotations, and auxiliary data.
#
# by Gregor Autischer (August 2025)
############################################

import os
import csv
import pandas as pd


def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def normalize_path(path):
    return str(path).replace('\\', '/')


def traverse_emotion_dataset(base_path, dataset_name, augmented=False):
    images = []
    dataset_path = os.path.join(base_path, dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} does not exist")
        return images
    
    # Expected emotion directories
    emotion_dirs = ['anger', 'surprise', 'fear', 'calm']
    
    # Check for FER and CKP subdirectories
    for subdir in ['FER', 'CKP']:
        subdir_path = os.path.join(dataset_path, subdir)
        if os.path.exists(subdir_path):
            for emotion in emotion_dirs:
                emotion_path = os.path.join(subdir_path, emotion)
                if os.path.exists(emotion_path):
                    for filename in os.listdir(emotion_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # Create relative path from project root
                            relative_path = os.path.join(dataset_name, subdir, emotion, filename)
                            images.append({
                                'image_paths': normalize_path(relative_path),
                                'emotion': emotion,
                                'augmented': augmented
                            })
    
    return images


def load_usage_data(script_dir):
    usage_file = os.path.join(script_dir, '..', 'model', 'image_usage.csv')
    
    if not os.path.exists(usage_file):
        print(f"Warning: Usage file {usage_file} does not exist")
        return {}
    
    usage_data = {}
    try:
        df = pd.read_csv(usage_file)
        for _, row in df.iterrows():
            # Normalize the path for matching
            normalized_path = normalize_path(row['image_path'])
            usage_data[normalized_path] = row['usage']
    except Exception as e:
        print(f"Error loading usage data: {e}")
    
    return usage_data


def load_auxiliary_data(script_dir):
    aux_file = os.path.join(script_dir, '..', 'model', 'faces_gender_race.csv')
    
    if not os.path.exists(aux_file):
        print(f"Warning: Auxiliary file {aux_file} does not exist")
        return {}
    
    aux_data = {}
    try:
        df = pd.read_csv(aux_file)
        for _, row in df.iterrows():
            # Normalize the path for matching
            normalized_path = normalize_path(row['img_path'])
            aux_data[normalized_path] = {
                'race': row.get('race', 'unclassified'),
                'race4': row.get('race4', 'unclassified'),
                'gender': row.get('gender', 'unclassified'),
                'age': row.get('age', 'unclassified')
            }
    except Exception as e:
        print(f"Error loading auxiliary data: {e}")
    
    return aux_data


def get_source_image_path(augmented_path):
    import re
    
    # Replace augmented dataset name with original
    source_path = augmented_path.replace('FER-Original-Dataset-Augmented', 'FER-Original-Dataset')
    
    # Extract filename and path components
    path_parts = source_path.split('/')
    filename = path_parts[-1]
    
    # Remove augmentation suffixes from filename using regex
    # Pattern: match number followed by underscore and anything until .png
    # Example: 1_brightness_very_bright.png -> 1.png
    filename_clean = re.sub(r'(\d+)_.*\.png$', r'\1.png', filename)
    
    # If no match (no underscore after number), keep original filename
    if filename_clean == filename:
        # Fallback: remove common augmentation patterns
        filename_clean = re.sub(r'_aug.*\.png$', '.png', filename)
        filename_clean = re.sub(r'_augmented.*\.png$', '.png', filename_clean)
    
    # Rebuild the path
    path_parts[-1] = filename_clean
    source_path = '/'.join(path_parts)
    
    return source_path


def main():
    script_dir = get_script_dir()
    project_root = os.path.join(script_dir, '..', '..')
    
    print("Starting dataset catalog generation...")
    
    # Step 1 & 2: Traverse datasets
    print("Traversing original dataset...")
    original_images = traverse_emotion_dataset(project_root, 'FER-Original-Dataset', augmented=False)
    print(f"Found {len(original_images)} original images")
    
    print("Traversing augmented dataset...")
    augmented_images = traverse_emotion_dataset(project_root, 'FER-Original-Dataset-Augmented', augmented=True)
    print(f"Found {len(augmented_images)} augmented images")
    
    # Step 3: Combine datasets
    all_images = original_images + augmented_images
    print(f"Total images: {len(all_images)}")
    
    if not all_images:
        print("No images found. Please check dataset paths.")
        return
    
    # Step 4: Load and merge usage data
    print("Loading usage data...")
    usage_data = load_usage_data(script_dir)
    
    for image in all_images:
        image['usage'] = usage_data.get(image['image_paths'], 'not used')
    
    # Step 5: Load and merge auxiliary data
    print("Loading auxiliary data...")
    aux_data = load_auxiliary_data(script_dir)
    
    for image in all_images:
        aux_info = aux_data.get(image['image_paths'])
        
        # If not found and this is an augmented image, try to get from source
        if not aux_info and image['augmented']:
            source_path = get_source_image_path(image['image_paths'])
            aux_info = aux_data.get(source_path)
        
        # Fill in auxiliary data or use defaults
        if aux_info:
            image['race'] = aux_info.get('race', 'unclassified')
            image['race4'] = aux_info.get('race4', 'unclassified')
            image['gender'] = aux_info.get('gender', 'unclassified')
            image['age'] = aux_info.get('age', 'unclassified')
        else:
            image['race'] = 'unclassified'
            image['race4'] = 'unclassified'
            image['gender'] = 'unclassified'
            image['age'] = 'unclassified'
    
    # Step 6: Generate final CSV
    print("Generating final CSV...")
    output_file = os.path.join(script_dir, '..', 'model', 'dataset_catalog.csv')
    
    # Define column order
    columns = ['image_paths', 'emotion', 'usage', 'race', 'race4', 'gender', 'age', 'augmented']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        
        for image in all_images:
            # Include all columns including augmented flag
            row = {col: image[col] for col in columns}
            writer.writerow(row)
    
    print(f"Dataset catalog generated: {output_file}")
    print(f"Total entries: {len(all_images)}")
    
    # Validation summary
    usage_counts = {}
    emotion_counts = {}
    for image in all_images:
        usage_counts[image['usage']] = usage_counts.get(image['usage'], 0) + 1
        emotion_counts[image['emotion']] = emotion_counts.get(image['emotion'], 0) + 1
    
    print("\nUsage distribution:")
    for usage, count in usage_counts.items():
        print(f"  {usage}: {count}")
    
    print("\nEmotion distribution:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}")


if __name__ == "__main__":
    main()