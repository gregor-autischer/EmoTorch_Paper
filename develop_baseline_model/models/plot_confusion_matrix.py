############################################
# Plot confusion matrix from CSV files in model folders
# 
# by Gregor Autischer (August 2025)
############################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def plot_both_matrices(fer_csv, ckp_csv, output_dir):
    
    # Read both CSV files
    fer_df = pd.read_csv(fer_csv, index_col=0)
    ckp_df = pd.read_csv(ckp_csv, index_col=0)
    
    # Get class labels
    class_labels = fer_df.columns.tolist()
    
    # Convert to numpy arrays
    fer_cm = fer_df.values
    ckp_cm = ckp_df.values
    
    # Create figure with subplots
    matrix_size = max(4.5, len(class_labels) * 0.9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(matrix_size * 2.0, matrix_size))
    
    # Plot FER validation confusion matrix
    sns.heatmap(fer_cm, annot=True, fmt='.2f', cmap='Blues',
                      xticklabels=class_labels, yticklabels=class_labels,
                      cbar_kws={'shrink': 0.7},
                      square=True, ax=ax1)
    
    # Add black outline around confusion matrix
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')
    
    # Add black outline around colorbar
    cbar1 = ax1.collections[0].colorbar
    cbar1.outline.set_linewidth(1)
    cbar1.outline.set_edgecolor('black')
    
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion matrix,\nconvolutional model, FER dataset')
    
    # Plot CKP validation confusion matrix
    sns.heatmap(ckp_cm, annot=True, fmt='.2f', cmap='Blues',
                      xticklabels=class_labels, yticklabels=class_labels,
                      cbar_kws={'shrink': 0.7},
                      square=True, ax=ax2)
    
    # Add black outline around confusion matrix
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor('black')
    
    # Add black outline around colorbar
    cbar2 = ax2.collections[0].colorbar
    cbar2.outline.set_linewidth(1)
    cbar2.outline.set_edgecolor('black')
    
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Confusion matrix,\nconvolutional model, CK dataset')

    fig.tight_layout()
    
    # Save plot in the specified output directory
    plot_filename = os.path.join(output_dir, "confusion_matrices_fer_vs_ckp.png")
    fig.savefig(plot_filename, dpi=100, bbox_inches='tight')
    print(f" -> Confusion matrix plot saved to: {plot_filename}")
    plt.close()

def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_confusion_matrix.py <model_folder>")
        print("Example: python plot_confusion_matrix.py model_00001")
        
        # List available model folders
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = script_dir  # models directory is the current directory
        
        model_folders = [d for d in os.listdir(models_dir) 
                        if d.startswith('model_') and os.path.isdir(os.path.join(models_dir, d))]
        
        if model_folders:
            model_folders.sort()
            print(f"\nAvailable model folders: {', '.join(model_folders)}")
            print(f"Using most recent: {model_folders[-1]}")
            model_folder = model_folders[-1]
        else:
            print("\nNo model folders found.")
            sys.exit(1)
    else:
        model_folder = sys.argv[1]
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, model_folder)
    
    if not os.path.exists(model_dir):
        print(f"Error: Model folder '{model_folder}' not found.")
        sys.exit(1)
    
    print(f" -> Looking for confusion matrix CSVs in {model_folder}")
    
    # Look for CSV files in the model folder
    fer_csv = os.path.join(model_dir, "confusion_matrix_fer_val.csv")
    ckp_csv = os.path.join(model_dir, "confusion_matrix_ckp_val.csv")

    fer_exists = os.path.exists(fer_csv)
    ckp_exists = os.path.exists(ckp_csv)

    if fer_exists and ckp_exists:
        print(f" -> Plotting confusion matrices from {model_folder}")
        plot_both_matrices(fer_csv, ckp_csv, model_dir)
    else:
        missing = []
        if not fer_exists:
            missing.append("confusion_matrix_fer_val.csv")
        if not ckp_exists:
            missing.append("confusion_matrix_ckp_val.csv")
        print(f" XX Error: Missing files in {model_folder}: {', '.join(missing)}")
        sys.exit(1)

if __name__ == "__main__":
    main()