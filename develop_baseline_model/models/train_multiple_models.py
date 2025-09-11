############################################
# Train Multiple Models with Different Data Configurations
# 
# by Gregor Autischer (August 2025)
############################################

import subprocess
import numpy as np
import time
import os
import csv
from datetime import datetime

# CONFIGURATION PARAMETERS
# Define the range for each data subset usage factor
FER_MIN = 0.9           # Minimum FER original usage
FER_MAX = 1.0           # Maximum FER original usage

FER_AUG_MIN = 0.0       # Minimum FER augmented usage  
FER_AUG_MAX = 0.1       # Maximum FER augmented usage

CKP_MIN = 0.9           # Minimum CKP original usage
CKP_MAX = 1.0           # Maximum CKP original usage

CKP_AUG_MIN = 0.0       # Minimum CKP augmented usage
CKP_AUG_MAX = 0.1       # Maximum CKP augmented usage

# Number of models to train
NUM_MODELS = 300

# Sampling strategy: 'random' or 'grid'
SAMPLING_STRATEGY = 'random'  # 'random' for random sampling, 'grid' for systematic grid

# Random seed for reproducibility
RANDOM_SEED = 42

def generate_random_configurations(num_models):
    """Generate random configurations for each model"""
    np.random.seed(RANDOM_SEED)
    
    configurations = []
    for i in range(num_models):
        config = {
            'model_num': i + 1,
            'fer': np.random.uniform(FER_MIN, FER_MAX),
            'fer_aug': np.random.uniform(FER_AUG_MIN, FER_AUG_MAX),
            'ckp': np.random.uniform(CKP_MIN, CKP_MAX),
            'ckp_aug': np.random.uniform(CKP_AUG_MIN, CKP_AUG_MAX)
        }
        configurations.append(config)
    
    return configurations

def generate_grid_configurations(num_models):
    # Generate grid-based configurations

    # Calculate points per dimension (approximate)
    points_per_dim = int(np.power(num_models, 0.25))  # 4D space
    
    # Create linearly spaced values for each dimension
    fer_values = np.linspace(FER_MIN, FER_MAX, points_per_dim)
    fer_aug_values = np.linspace(FER_AUG_MIN, FER_AUG_MAX, points_per_dim)
    ckp_values = np.linspace(CKP_MIN, CKP_MAX, points_per_dim)
    ckp_aug_values = np.linspace(CKP_AUG_MIN, CKP_AUG_MAX, points_per_dim)
    
    configurations = []
    model_num = 1
    
    for fer in fer_values:
        for fer_aug in fer_aug_values:
            for ckp in ckp_values:
                for ckp_aug in ckp_aug_values:
                    if model_num > num_models:
                        break
                    config = {
                        'model_num': model_num,
                        'fer': fer,
                        'fer_aug': fer_aug,
                        'ckp': ckp,
                        'ckp_aug': ckp_aug
                    }
                    configurations.append(config)
                    model_num += 1
                if model_num > num_models:
                    break
            if model_num > num_models:
                break
        if model_num > num_models:
            break
    
    return configurations[:num_models]

def train_model(config):
    """Train a single model with given configuration"""
    model_num = config['model_num']
    
    print(f"\n{'='*60}")
    print(f"Training Model {model_num}/{NUM_MODELS}")
    print(f"{'='*60}")
    print(f"FER Original: {config['fer']:.3f}")
    print(f"FER Augmented: {config['fer_aug']:.3f}")
    print(f"CKP Original: {config['ckp']:.3f}")
    print(f"CKP Augmented: {config['ckp_aug']:.3f}")
    
    # Build command - use absolute path to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'convolutional_nn_pytorch.py')
    
    cmd = [
        'python', script_path,
        '--fer', str(config['fer']),
        '--fer-aug', str(config['fer_aug']),
        '--ckp', str(config['ckp']),
        '--ckp-aug', str(config['ckp_aug'])
    ]
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        print(f"✓ Model {model_num} completed in {training_time:.1f} seconds")
        
        return {
            **config,
            'status': 'success',
            'training_time': training_time,
            'error': None
        }
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Model {model_num} failed: {e}")
        return {
            **config,
            'status': 'failed',
            'training_time': time.time() - start_time,
            'error': str(e)
        }
    except Exception as e:
        print(f"✗ Model {model_num} encountered error: {e}")
        return {
            **config,
            'status': 'error',
            'training_time': time.time() - start_time,
            'error': str(e)
        }

def main():
    print("="*60)
    print("MULTI-MODEL TRAINING SCRIPT")
    print("="*60)
    print(f"Number of models to train: {NUM_MODELS}")
    print(f"Sampling strategy: {SAMPLING_STRATEGY}")
    print(f"\nData usage ranges:")
    print(f"  FER Original: {FER_MIN:.1f} - {FER_MAX:.1f}")
    print(f"  FER Augmented: {FER_AUG_MIN:.1f} - {FER_AUG_MAX:.1f}")
    print(f"  CKP Original: {CKP_MIN:.1f} - {CKP_MAX:.1f}")
    print(f"  CKP Augmented: {CKP_AUG_MIN:.1f} - {CKP_AUG_MAX:.1f}")
    print("="*60)
    
    # Generate configurations
    if SAMPLING_STRATEGY == 'random':
        configurations = generate_random_configurations(NUM_MODELS)
    else:
        configurations = generate_grid_configurations(NUM_MODELS)
    
    # Train all models
    results = []
    total_start_time = time.time()
    
    for i, config in enumerate(configurations):
        result = train_model(config)
        results.append(result)
        
        # Print progress summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] != 'success')
        elapsed = time.time() - total_start_time
        avg_time = elapsed / len(results)
        remaining = avg_time * (NUM_MODELS - len(results))
        
        print(f"\nProgress: {len(results)}/{NUM_MODELS} | Success: {successful} | Failed: {failed}")
        print(f"Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min (estimated)")
    
    # Print summary
    total_time = time.time() - total_start_time
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] != 'success')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total models trained: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per model: {total_time/len(results):.1f} seconds")
    
    print(f"\nAll model results are saved in individual model_XXXXX folders.")
    print(f"To find the best model, check the final_accuracies.csv files in each folder.")

if __name__ == "__main__":
    # Check if convolutional_nn_pytorch.py exists in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'convolutional_nn_pytorch.py')
    
    if not os.path.exists(script_path):
        print(f"Error: convolutional_nn_pytorch.py not found at {script_path}")
        exit(1)
    
    main()