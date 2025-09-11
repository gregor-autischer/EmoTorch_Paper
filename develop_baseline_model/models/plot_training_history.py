############################################
# Plot training history from CSV files in model folders
# 
# by Gregor Autischer (August 2025)
############################################

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import glob

def plot_all_histories(model_dir):
    """
    Plot all training histories (FER, CKP, Combined) from a model folder
    
    Args:
        model_dir: Path to the model directory containing CSV files
    """
    # Find all training history CSVs in the model folder
    csv_files = glob.glob(os.path.join(model_dir, "training_history_*.csv"))
    
    if not csv_files:
        print(f" XX No training history CSV files found in {model_dir}")
        return
    
    # Sort to ensure consistent order
    csv_files.sort()
    print(f" -> Found {len(csv_files)} training history files")
    
    # Create a figure with subplots for each dataset
    num_files = len(csv_files)
    fig, axes = plt.subplots(num_files, 2, figsize=(14, 5 * num_files))
    
    if num_files == 1:
        axes = axes.reshape(1, -1)
    
    for idx, csv_file in enumerate(csv_files):
        # Read CSV file
        df = pd.read_csv(csv_file)
        base_name = os.path.basename(csv_file).replace('training_history_', '').replace('.csv', '').upper()
        
        ax1 = axes[idx, 0]
        ax2 = axes[idx, 1]
        
        # Plot accuracy
        ax1.plot(df['epoch'], df['train_accuracy'], label='Training Accuracy', marker='o', linewidth=2)
        
        # Check for different validation accuracy columns
        if 'fer_val_accuracy' in df.columns and df['fer_val_accuracy'].notna().any():
            ax1.plot(df['epoch'], df['fer_val_accuracy'], label='FER Validation Accuracy', marker='s', linewidth=2)
            val_col = 'fer_val_accuracy'
            val_type = 'FER Validation'
        elif 'ckp_val_accuracy' in df.columns and df['ckp_val_accuracy'].notna().any():
            ax1.plot(df['epoch'], df['ckp_val_accuracy'], label='CKP Validation Accuracy', marker='^', linewidth=2)
            val_col = 'ckp_val_accuracy'
            val_type = 'CKP Validation'
        elif 'combined_val_accuracy' in df.columns and df['combined_val_accuracy'].notna().any():
            ax1.plot(df['epoch'], df['combined_val_accuracy'], label='Combined Validation Accuracy', marker='d', linewidth=2)
            val_col = 'combined_val_accuracy'
            val_type = 'Combined Validation'
        else:
            val_col = None
            val_type = None
            
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title(f'{base_name} - Model Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o', linewidth=2)
        
        # Check for different validation loss columns
        if 'fer_val_loss' in df.columns and df['fer_val_loss'].notna().any():
            ax2.plot(df['epoch'], df['fer_val_loss'], label='FER Validation Loss', marker='s', linewidth=2)
        elif 'ckp_val_loss' in df.columns and df['ckp_val_loss'].notna().any():
            ax2.plot(df['epoch'], df['ckp_val_loss'], label='CKP Validation Loss', marker='^', linewidth=2)
        elif 'combined_val_loss' in df.columns and df['combined_val_loss'].notna().any():
            ax2.plot(df['epoch'], df['combined_val_loss'], label='Combined Validation Loss', marker='d', linewidth=2)
            
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title(f'{base_name} - Model Loss', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Print summary for this dataset
        if val_col:
            print(f" └-> {base_name}: Final train acc: {df['train_accuracy'].iloc[-1]:.1f}%, val acc: {df[val_col].iloc[-1]:.1f}%")
            # Check for overfitting
            final_train = df['train_accuracy'].iloc[-1]
            final_val = df[val_col].iloc[-1]
            if final_train - final_val > 10:
                print(f"     XX Overfitting detected (gap: {final_train - final_val:.1f}%)")
        else:
            print(f" └-> {base_name}: Final train acc: {df['train_accuracy'].iloc[-1]:.1f}%")
    
    # Add overall title
    model_folder = os.path.basename(model_dir)
    fig.suptitle(f'Training History - {model_folder}', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot in the model directory
    plot_filename = os.path.join(model_dir, "training_history_plots.png")
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
    print(f" -> Saved: {plot_filename}")
    plt.close()

def plot_single_history(csv_file, output_dir=None):
    """
    Plot a single training history CSV file
    
    Args:
        csv_file: Path to training history CSV file
        output_dir: Directory to save the plot (optional, uses CSV location if not specified)
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(df['epoch'], df['train_accuracy'], label='Training Accuracy', marker='o', linewidth=2)
    
    # Check for different validation accuracy columns
    if 'fer_val_accuracy' in df.columns and df['fer_val_accuracy'].notna().any():
        ax1.plot(df['epoch'], df['fer_val_accuracy'], label='FER Validation Accuracy', marker='s', linewidth=2)
    elif 'ckp_val_accuracy' in df.columns and df['ckp_val_accuracy'].notna().any():
        ax1.plot(df['epoch'], df['ckp_val_accuracy'], label='CKP Validation Accuracy', marker='^', linewidth=2)
    elif 'combined_val_accuracy' in df.columns and df['combined_val_accuracy'].notna().any():
        ax1.plot(df['epoch'], df['combined_val_accuracy'], label='Combined Validation Accuracy', marker='d', linewidth=2)
        
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o', linewidth=2)
    
    # Check for different validation loss columns
    if 'fer_val_loss' in df.columns and df['fer_val_loss'].notna().any():
        ax2.plot(df['epoch'], df['fer_val_loss'], label='FER Validation Loss', marker='s', linewidth=2)
    elif 'ckp_val_loss' in df.columns and df['ckp_val_loss'].notna().any():
        ax2.plot(df['epoch'], df['ckp_val_loss'], label='CKP Validation Loss', marker='^', linewidth=2)
    elif 'combined_val_loss' in df.columns and df['combined_val_loss'].notna().any():
        ax2.plot(df['epoch'], df['combined_val_loss'], label='Combined Validation Loss', marker='d', linewidth=2)
        
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    base_name = os.path.basename(csv_file).replace('.csv', '')
    fig.suptitle(f'Training History - {base_name}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    base_name = os.path.basename(csv_file).replace('.csv', '')
    plot_filename = os.path.join(output_dir, f"{base_name}_plot.png")
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
    print(f" -> Plot saved to: {plot_filename}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("\n[Plot Training History]")
        print(" -> Usage: python plot_training_history.py <model_folder_or_csv_file>")
        print(" -> Examples:")
        print("    python plot_training_history.py model_00001")
        print("    python plot_training_history.py model_00001/training_history_fer.csv")
        
        # List available model folders
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = script_dir  # models directory is the current directory
        
        model_folders = [d for d in os.listdir(models_dir) 
                        if d.startswith('model_') and os.path.isdir(os.path.join(models_dir, d))]
        
        if model_folders:
            model_folders.sort()
            print(f" -> Found {len(model_folders)} model folders")
            print(f" -> Using most recent: {model_folders[-1]}")
            model_folder = model_folders[-1]
            model_dir = os.path.join(models_dir, model_folder)
            plot_all_histories(model_dir)
        else:
            print(" XX No model folders found")
            sys.exit(1)
    else:
        arg = sys.argv[1]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check if it's a CSV file
        if arg.endswith('.csv'):
            if not os.path.exists(arg):
                # Try as relative to script dir
                arg = os.path.join(script_dir, arg)
            if os.path.exists(arg):
                print(f"\n[Plot Training History]")
                print(f" -> Plotting: {arg}")
                plot_single_history(arg)
            else:
                print(f" XX CSV file '{arg}' not found")
                sys.exit(1)
        else:
            # Assume it's a model folder
            model_dir = os.path.join(script_dir, arg)
            if not os.path.exists(model_dir):
                print(f" XX Model folder '{arg}' not found")
                sys.exit(1)
            print(f"\n[Plot Training History]")
            print(f" -> Plotting for: {arg}")
            plot_all_histories(model_dir)

if __name__ == "__main__":
    main()