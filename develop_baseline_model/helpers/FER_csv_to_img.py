############################################
# Convert FER-2013 CSV to organized image folders
# 
# by Gregor Autischer (August 2025)
############################################

import csv
import numpy as np
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
FER_CSV_PATH = "./FER-Datasets/fer-2013/fer2013/fer2013.csv"
OUTPUT_DIR = "./FER-Datasets/FER2013img"

EMOTIONS = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

def convert_csv_to_images():
    # Check if input file exists
    if not os.path.exists(FER_CSV_PATH):
        print(f" XX Error: CSV file not found at {FER_CSV_PATH}")
        return
    
    print(f" -> Reading CSV: {FER_CSV_PATH}")
    
    # Create output directories
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    for emotion_name in EMOTIONS.values():
        (Path(OUTPUT_DIR) / emotion_name).mkdir(exist_ok=True)
    
    print(f" -> Converting images to: {OUTPUT_DIR}")
    
    # Count total rows first
    with open(FER_CSV_PATH, 'r') as file:
        total_rows = sum(1 for _ in csv.DictReader(file))
    
    # Process CSV with progress bar
    count = 0
    with open(FER_CSV_PATH, 'r') as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader, total=total_rows, desc="Converting"):
            emotion = int(row['emotion'])
            pixels = np.array([int(x) for x in row['pixels'].split()], dtype=np.uint8)
            
            if len(pixels) != 2304:  # 48x48
                continue
                
            image = pixels.reshape(48, 48)
            emotion_name = EMOTIONS[emotion]
            filename = f"{emotion_name}_{count:06d}.png"
            filepath = os.path.join(OUTPUT_DIR, emotion_name, filename)
            
            Image.fromarray(image).save(filepath)
            count += 1
    
    print(f" -> Converted {count} images total")

if __name__ == "__main__":
    convert_csv_to_images()