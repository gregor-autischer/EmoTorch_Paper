############################################
# Compare models to original
# 
# by Gregor Autischer (August 2025)
############################################

import os
import pandas as pd
import numpy as np
import csv
from datetime import datetime

# weights for similarity calc
ACCURACY_WEIGHT = 0
CONFUSION_MATRIX_WEIGHT = 1

# what to compare from final_accuracies.csv
ACCURACY_METRICS = ['Training Accuracy', 'Combined Validation Accuracy']

def read_final_accuracies(model_path):
    csv_path = os.path.join(model_path, 'final_accuracies.csv')
    if not os.path.exists(csv_path):
        return None
    
    accuracies = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row['Metric']
                value = row['Value']
                if metric in ACCURACY_METRICS:
                    accuracies[metric] = float(value.strip())
        return accuracies
    except:
        return None

def read_confusion_matrix(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0)
        return df.values.astype(float)
    except:
        return None

def calculate_accuracy_mse(orig, model):
    if orig is None or model is None:
        return float('inf')
    
    mse = 0.0
    count = 0
    
    for metric in ACCURACY_METRICS:
        if metric in orig and metric in model:
            diff = orig[metric] - model[metric]
            mse += diff ** 2
            count += 1
    
    return mse / count if count > 0 else float('inf')

def calculate_matrix_mse(orig, model):
    if orig is None or model is None:
        return float('inf')
    if orig.shape != model.shape:
        return float('inf')
    return np.mean((orig - model) ** 2)

def main():
    print("\n[Compare Models] Starting comparison to model_original...")
    
    # Get script directory to find model folders
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # load original
    orig_path = os.path.join(script_dir, 'model_original')
    if not os.path.exists(orig_path):
        print(" XX model_original not found")
        return
        
    print(" -> Loading original model data...")
    orig_acc = read_final_accuracies(orig_path)
    orig_fer_cm = read_confusion_matrix(os.path.join(orig_path, 'confusion_matrix_fer_val.csv'))
    orig_ckp_cm = read_confusion_matrix(os.path.join(orig_path, 'confusion_matrix_ckp_val.csv'))
    
    if not orig_acc or orig_fer_cm is None or orig_ckp_cm is None:
        print(" XX couldn't load original model data")
        return
    
    print(" └-> Loaded accuracies and confusion matrices")
    
    # find all model folders in script directory
    model_folders = [d for d in os.listdir(script_dir) 
                    if d.startswith('model_') and d != 'model_original' 
                    and os.path.isdir(os.path.join(script_dir, d))]
    
    if not model_folders:
        print(" XX no model folders found")
        return
    
    model_folders.sort()
    print(f" -> Found {len(model_folders)} models to compare")
    
    # compare each
    results = []
    for i, folder in enumerate(model_folders):
        if i % 100 == 0 and i > 0:
            print(f" └-> Processed {i}/{len(model_folders)}...")
            
        # read model data - use full path
        model_path = os.path.join(script_dir, folder)
        model_acc = read_final_accuracies(model_path)
        model_fer_cm = read_confusion_matrix(os.path.join(model_path, 'confusion_matrix_fer_val.csv'))
        model_ckp_cm = read_confusion_matrix(os.path.join(model_path, 'confusion_matrix_ckp_val.csv'))
        
        # calc mse
        acc_mse = calculate_accuracy_mse(orig_acc, model_acc)
        fer_mse = calculate_matrix_mse(orig_fer_cm, model_fer_cm)
        ckp_mse = calculate_matrix_mse(orig_ckp_cm, model_ckp_cm)
        
        if acc_mse == float('inf') or fer_mse == float('inf') or ckp_mse == float('inf'):
            combined = float('inf')
        else:
            combined = (ACCURACY_WEIGHT * acc_mse + 
                       CONFUSION_MATRIX_WEIGHT * fer_mse + 
                       CONFUSION_MATRIX_WEIGHT * ckp_mse)
        
        result = {
            'model': folder,
            'combined_score': combined,
            'accuracy_mse': acc_mse,
            'fer_cm_mse': fer_mse,
            'ckp_cm_mse': ckp_mse
        }
        
        # add individual values
        if model_acc:
            for metric in ACCURACY_METRICS:
                if metric in model_acc:
                    key = metric.lower().replace(' ', '_')
                    result[f'{key}'] = model_acc[metric]
                    result[f'{key}_diff'] = model_acc[metric] - orig_acc.get(metric, 0)
        
        results.append(result)
    
    # filter and sort
    valid = [r for r in results if r['combined_score'] != float('inf')]
    valid.sort(key=lambda x: x['combined_score'])
    
    print(f" -> Valid comparisons: {len(valid)}/{len(results)}")
    
    if not valid:
        print(" XX no valid models")
        return
    
    # save results to script directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(script_dir, f'comparison_results_{timestamp}.csv')
    
    df = pd.DataFrame(valid)
    df.to_csv(output_file, index=False)
    
    print(f" -> Results saved to: {output_file}")
    
    # show top 5
    print("\n[Top 5 Most Similar]")
    for i in range(min(5, len(valid))):
        print(f" {i+1}. {valid[i]['model']} (score: {valid[i]['combined_score']:.4f})")

if __name__ == "__main__":
    main()