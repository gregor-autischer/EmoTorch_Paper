# EmoTorch - Paper

PyTorch implementation of emotion recognition using convolutional neural networks.

# TRAIN MODEL FROM GROUND UP
All the necessary scripts that were used to train the baseline model from the ground up are in the directory develop_baseline_model.

## Quick Start - Execution Order
Run those commands from the project root.

### Step 0: Create VENV and Install Requirements
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: fer_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Prepare Data
```bash
# 1. Convert FER-2013 CSV to images
python develop_baseline_model/helpers/FER_csv_to_img.py

# 2. Create organized dataset structure
python develop_baseline_model/helpers/create_original_dataset.py

# 3. Generate augmented images (28 variations per image)
python develop_baseline_model/helpers/augment_original_dataset.py
```

### Step 2: Train Models
```bash
# Train single model
python develop_baseline_model/models/convolutional_nn_pytorch.py

# Or train multiple models with different parameters
python develop_baseline_model/models/train_multiple_models.py

# Generate plots
python develop_baseline_model/models/plot_confusion_matrix.py model_00001
python develop_baseline_model/models/plot_training_history.py model_00001
```

### Step 3: Analyze Results
```bash
# Compare models to find best match to original
python develop_baseline_model/models/compare_models_to_original.py

# Evaluate model on test data
python develop_baseline_model/models/evaluate_model.py model_00001
reload

```

## Scripts Overview

### Helpers (Data Preparation)

**`FER_csv_to_img.py`**
- Converts FER-2013 CSV to PNG images organized by emotion
It reads the FER-2013 CSV from ./FER-Datasets/fer-2013/fer2013/fer2013.csv and saves the organized images to ./FER-Datasets/FER2013img/ with subdirectories for each emotion (angry/, fear/, etc.).

**`create_original_dataset.py`**
1. Copies images from FER-2013 and CK+ datasets
2. Filters to only 4 emotions: anger, fear, calm, surprise
3. Renames files sequentially (1.png, 2.png, etc.) in each emotion folder
4. Maps emotions: neutral→calm (FER), happy→calm (CK+)
5. Creates structure: FER-Original-Dataset/FER/ and FER-Original-Dataset/CKP/

It reads from ../../FER-Datasets/FER2013img and ../../FER-Datasets/CK+ 
and outputs to ../../FER-Original-Dataset/.

**`augment_original_dataset.py`**
- Creates 28 augmented versions per image
- Applies rotations, noise, blur, occlusions, etc.

It reads from ../../FER-Original-Dataset/ and outputs to ../../FER-Original-Dataset-Augmented/.

**`create_train_test_subsets.py`** *(optional)*
- Splits data into train/validation sets
- Not needed if using convolutional_nn_pytorch.py with dynamic splits

**`create_train_test_subsets_augmented.py`** *(optional)*
- Same as above but for augmented data

### Models (Training & Analysis)

**`convolutional_nn_pytorch.py`**
- Main training script
- Trains CNN with configurable data usage factors
- Saves model, metrics, confusion matrices

**`train_multiple_models.py`**
- Set in this script
    - How many runs of training
    - with which data usage ranges
- Trains multiple models with random parameter combinations
- Explores different data usage factors automatically

**`compare_models_to_original.py`**
- Compares all trained models to model_original
- Finds models with most similar performance

**`evaluate_model.py`**
- Tests saved model on any dataset
- Identifies misclassifications and saves error images

**`plot_confusion_matrix.py`**
- Creates confusion matrix heatmaps from model results

**`plot_training_history.py`**
- Plots training/validation accuracy and loss curves


# BASELINE MODEL
The recreated baseline model and all its required files are in the baseline_model directory.

## Scripts in baseline_model/model/

**`predict_image_baselinemodel.py`**
- Predicts emotion from a single image file using the baseline model
- Takes image path as command-line argument
- Displays predicted emotion with confidence score and probability distribution for all emotions
- Usage: `python predict_image_baselinemodel.py <image_path> [model_path]`

**`eval_model.py`**
- Evaluates the baseline model on FER and CKP datasets from image_usage.csv
- Generates side-by-side confusion matrices with percentages for both datasets
- Calculates per-class precision, recall, and F1 scores
- Creates a single combined visualization showing model performance on both datasets
- Usage: `python eval_model.py`





# Required Data Structure

```
EmoTorch_Paper/
├── develop_baseline_model/
│   ├── helpers/           (data preparation scripts)
│   └── models/            (training & analysis scripts)
│       ├── model_original/  (baseline model)
│       └── model_00001/     (trained models)
├── FER-Original-Dataset/
│   ├── FER/
│   │   ├── anger/
│   │   ├── fear/
│   │   ├── calm/
│   │   └── surprise/
│   └── CKP/
│       └── (same emotions)
└── FER-Original-Dataset-Augmented/
    └── (same structure)
```