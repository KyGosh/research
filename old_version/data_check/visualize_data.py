import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_single_sample(pt_path, out_dir):
    data = torch.load(pt_path).numpy()
    # Data columns:
    # kb: BACK, RIGHT, FORWARD, LEFT (0,1,2,3)
    # ms: fire, dp, dy, speed, acc, jerk (4,5,6,7,8,9)
    dp = data[:, 5]
    dy = data[:, 6]
    speed = data[:, 7]

    plt.figure(figsize=(15, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(dp, label='d_pitch (Vertical)', color='blue')
    plt.legend()
    plt.title(f'Mouse Movement Analysis - {os.path.basename(pt_path)}')
    
    plt.subplot(3, 1, 2)
    plt.plot(dy, label='d_yaw (Horizontal)', color='orange')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(speed, label='Speed', color='green')
    plt.legend()
    plt.xlabel('Ticks (640 total)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "single_sample_plot.png"))
    plt.close()
    print("Saved single_sample_plot.png")

def plot_empty_ratio_distribution(meta_path, out_dir):
    df = pd.read_csv(meta_path)
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of empty_ratio
    counts, bins, patches = plt.hist(df['empty_ratio'], bins=50, color='purple', alpha=0.7)
    plt.title('Distribution of Empty Row Ratio Across All Segments')
    plt.xlabel('Empty Ratio (0 to 1)')
    plt.ylabel('Frequency (Number of Segments)')
    
    # Calculate how many would be dropped at threshold 0.8
    threshold = 0.8
    dropped = len(df[df['empty_ratio'] > threshold])
    total = len(df)
    
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold={threshold} (Drops {dropped}/{total} segments)')
    plt.legend()
    
    plt.savefig(os.path.join(out_dir, "empty_ratio_distribution.png"))
    plt.close()
    print("Saved empty_ratio_distribution.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default=os.path.join("d:\\", "Project", "Research", "serialized_features"))
    parser.add_argument("--out-dir", type=str, default=os.path.join("d:\\", "Project", "Research", "scripts", "data_check"))
    args = parser.parse_args()

    meta_path = os.path.join(args.features_dir, "feature_metadata.csv")
    
    if not os.path.exists(meta_path):
        print(f"Metadata not found at {meta_path}. Please run serialize_features.py first.")
        return

    # 1. Plot empty ratio distribution
    plot_empty_ratio_distribution(meta_path, args.out_dir)

    # 2. Pick a random valid .pt file to plot
    df = pd.read_csv(meta_path)
    valid_samples = df[df['empty_ratio'] < 0.5]
    if len(valid_samples) > 0:
        sample_path = valid_samples.iloc[0]['filepath']
        plot_single_sample(sample_path, args.out_dir)
    else:
        print("No valid samples found with empty_ratio < 0.5 to plot.")

if __name__ == "__main__":
    main()
