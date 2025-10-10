# Train the New Model from the ground up
All the necessary scripts that were used to train the new model from the ground up are in the directories `build_new_dataset` and `build_new_model`.

## Quick Start - Execution Order
Run those commands from the project root.

### Step 0: Create VENV and Install Requirements
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Prepare New Dataset
```bash
# 1. Copy FER and CKP datasets and transfer demographics
python build_new_dataset/FER_CKP_to_final_structure.py

# 2. Process RAF-DB dataset and extract demographics
python build_new_dataset/RAF_to_final_structure.py

# 3. Generate augmented images (10 variations per image)
python build_new_dataset/data_augmentation.py
```

### Step 2: Balance Dataset
```bash
# Create balanced dataset with reduced CALM images
python build_new_model/balance_calm_reduced.py
```

### Step 3: Train New Model
```bash
# Train enhanced model with fairness analysis
python build_new_model/enhanced_model.py
```

### Step 4: Predict with Model
```bash
# Predict emotion for a single image
python build_new_model/predict_image.py /path/to/image.jpg
```

## Scripts Overview

### build_new_dataset/ (Data Preparation)

**`FER_CKP_to_final_structure.py`**
- Copies FER and CKP datasets from FER-Original-Dataset to FER-New-Dataset
- Transfers demographic information from baseline model's faces_gender_race.csv
- Maps demographics to new standardized format:
  - Race: White -> Caucasian, Black -> African-American, Indian/Asian -> Asian
  - Gender: Male -> male, Female -> female
  - Age: Maps old ranges to new (0-2 -> 0-3, 3-9/10-19 -> 4-19, 20-29/30-39 -> 20-39, etc.)
- Creates dataset_new_attributs.csv with columns: image_path, emotion, race, gender, age, augmented, usage

It reads from `./FER-Original-Dataset/` and `./baseline_model/model/faces_gender_race.csv`
and outputs to `./FER-New-Dataset/FER-New-Dataset/` and `./FER-New-Dataset/dataset_new_attributs.csv`.

**`RAF_to_final_structure.py`**
- Processes RAF-DB dataset images and metadata into final structure
- Reads emotion labels from RAF-DB annotation files
- Maps RAF-DB emotions to 4-class system:
  - 1 -> surprise, 2 -> fear, 6 -> anger, 4 -> calm (happiness), 7 -> calm (neutral)
- Extracts demographics from manual attribute files:
  - Gender: 0 -> male, 1 -> female, 2 -> unsure
  - Race: 0 -> Caucasian, 1 -> African-American, 2 -> Asian
  - Age: 0 -> 0-3, 1 -> 4-19, 2 -> 20-39, 3 -> 40-69, 4 -> 70+
- Converts aligned images to grayscale 48x48
- Renames files sequentially (1.jpg, 2.jpg, etc.) in each emotion folder
- Appends to dataset_new_attributs.csv

It reads from `./FER-Datasets/RAF-DB/`
and outputs to `./FER-New-Dataset/FER-New-Dataset/RAF/` and appends to `./FER-New-Dataset/dataset_new_attributs.csv`.

**`data_augmentation.py`**
- Creates 10 augmented versions per image (compared to 28 in baseline)
- Focus on head coverings and realistic variations:
  1. Rotation (25 degrees) - Geometric transformation
  2. Dark (20-40% brightness) - Low-light conditions
  3. High Contrast (2.5-3.5x) - Strong lighting
  4. Light Noise (sigma=0.04) - Sensor noise
  5. Blur (Gaussian 9x9, sigma=3-5) - Motion/focus issues
  6. Top Rectangle (25-42% coverage) - Hats, hijabs, caps
  7. Top Left Diagonal - Asymmetric head coverings
  8. Top Right Diagonal - Asymmetric head coverings
  9. Forehead Bar (3-6px) - Headbands, bandanas
  10. Heavy Hair (8-15 strands) - Hair occlusion
- Updates dataset_new_attributs.csv with augmentation type in 'augmented' column

It reads from `./FER-New-Dataset/FER-New-Dataset/` and `./FER-New-Dataset/dataset_new_attributs.csv`
and outputs to `./FER-New-Dataset/FER-New-Dataset-Augmented/` and updates CSV.

### build_new_model/ (Training & Analysis)

**`balance_calm_reduced.py`**
- Creates balanced dataset by reducing CALM class overrepresentation
- Uses 50% of CALM images, 100% of other emotions (anger, fear, surprise)
- Maintains augmentation integrity (augmented images follow their base images)
- Creates 80/20 train/test split with no data leakage
- Outputs dataset_calm_reduced.csv with 'split' column (train/test)
- Key features:
  - Random sampling with seed=42 for reproducibility
  - No stratification applied
  - Ensures augmented images stay with their original in same split

It reads from `./FER-New-Dataset/dataset_new_attributs.csv`
and outputs to `./FER-New-Dataset/dataset_calm_reduced.csv`.

**`enhanced_model.py`**
- Main training script with enhanced architecture and fairness analysis
- **Architecture:**
  - 3 convolutional blocks (32, 64, 128 filters) with 3x3 kernels
  - Batch normalization after each conv layer
  - Progressive dropout (0.2, 0.3, 0.4) to prevent overfitting
  - Global Average Pooling instead of flattening
  - Fully connected layers: 128 -> 256 -> 4 with dropout
- **Training features:**
  - Weighted CrossEntropyLoss for class imbalance
  - AdamW optimizer (lr=0.001, weight_decay=0.01)
  - CosineAnnealingWarmRestarts scheduler (T_0=10, T_mult=2)
  - WeightedRandomSampler for oversampling minority classes
  - Early stopping (patience=10)
  - Gradient clipping (max_norm=1.0)
- **Outputs:**
  - Model checkpoint (model.pth)
  - Training history CSV
  - Confusion matrices (train and test, normalized PNG images)
  - Classification report with precision/recall/F1 per class
  - Fairness analysis report with metrics by gender, race, age, augmentation type
  - Per-group confusion matrices saved as PNG images
  - Final accuracies CSV
  - Model parameters TXT

It reads from `./FER-New-Dataset/dataset_calm_reduced.csv`
and outputs to `./build_new_model/model_XXXXX/`.

**`predict_image.py`**
- Simple prediction script for single images
- Loads trained model from checkpoint
- Preprocesses image (grayscale, 48x48, normalize)
- Returns predicted emotion with confidence scores
- Displays probability distribution for all 4 emotions
- Usage: `python predict_image.py /path/to/image.jpg`

It uses model from `./build_new_model/model_00010_final/model.pth` (configurable).

## Required Data Structures

This is the general data structure for the new model with the finished dataset:
```
EmoTorch_Paper/
EmoTorch_Paper/
├── build_new_dataset/         (data preparation scripts)
├── build_new_model/            (training & analysis scripts)
│   └── model_XXXXX/            (trained models with fairness reports)
└── FER-New-Dataset/
    ├── dataset_new_attributs.csv       (full dataset with demographics)
    ├── dataset_calm_reduced.csv        (balanced dataset with train/test split)
    ├── FER-New-Dataset/
    │   ├── FER/
    │   │   ├── anger/
    │   │   ├── fear/
    │   │   ├── calm/
    │   │   └── surprise/
    │   ├── CKP/
    │   │   └── (same emotions)
    │   └── RAF/
    │       └── (same emotions)
    └── FER-New-Dataset-Augmented/
        └── (same structure with augmented images)
```

This is the source dataset structure that the preparation scripts use:
```
EmoTorch_Paper/
├── FER-Original-Dataset/       (from baseline model)
│   ├── FER/
│   │   ├── anger/
│   │   ├── fear/
│   │   ├── calm/
│   │   └── surprise/
│   └── CKP/
│       └── (same emotions)
├── FER-Datasets/
│   └── RAF-DB/
│       ├── EmoLabel/
│       │   └── list_patition_label.txt
│       ├── Annotation/
│       │   └── manual/              (demographic attributes)
│       └── Image/
│           └── aligned/             (aligned face images)
└── baseline_model/model/
    └── faces_gender_race.csv        (FER/CKP demographics)
```
