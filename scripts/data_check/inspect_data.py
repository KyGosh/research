import os
import torch
import pandas as pd
import numpy as np
import random
import argparse

# 特征索引映射 (保持一致)
FEATURE_MAP = {
    "kb_back": 0, "kb_right": 1, "kb_forward": 2, "kb_left": 3,
    "fire": 4, "dp": 5, "dy": 6, "speed": 7, "acc": 8, "jerk": 9
}

def inspect_sample(pt_path):
    print(f"\n{'='*20} Inspecting Sample {'='*20}")
    print(f"File: {pt_path}")
    
    # 1. Load tensor
    x = torch.load(pt_path)
    data = x.numpy()
    
    print(f"Shape: {data.shape} (Ticks, Features)")
    
    # 2. Create a readable DataFrame for the first 15 ticks
    cols = ["kb_B", "kb_R", "kb_F", "kb_L", "fire", "dp", "dy", "speed", "acc", "jerk"]
    df_head = pd.DataFrame(data[100:115, :], columns=cols)
    
    print("\n--- First 15 Ticks (Data Preview) ---")
    print(df_head.to_string())
    
    # 3. Statistics for key features (Confirm quantization)
    print("\n--- Feature Value Statistics (Unique Values) ---")
    for feat in ["dp", "dy", "speed"]:
        idx = FEATURE_MAP[feat]
        vals = data[:, idx]
        unique_vals = np.unique(vals)
        print(f"{feat:6s} | Unique Count: {len(unique_vals):2d} | Range: [{min(vals):.1f}, {max(vals):.1f}] | Values: {sorted(unique_vals)}")

    # 4. Check for sign preservation
    dp_idx = FEATURE_MAP["dp"]
    has_negative = np.any(data[:, dp_idx] < 0)
    has_positive = np.any(data[:, dp_idx] > 0)
    print(f"\nSign Check (dp): Contains Negative? {has_negative} | Contains Positive? {has_positive}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default=os.path.join("d:\\", "Project", "Research", "serialized_features"))
    parser.add_argument("--file", type=str, help="Path to a specific .pt file")
    args = parser.parse_args()

    target_file = args.file
    
    if not target_file:
        # Randomly pick a .pt file
        all_pts = []
        for root, _, files in os.walk(args.features_dir):
            for f in files:
                if f.endswith(".pt"):
                    all_pts.append(os.path.join(root, f))
        
        if not all_pts:
            print("No .pt files found in serialized_features!")
            return
        target_file = random.choice(all_pts)

    inspect_sample(target_file)

if __name__ == "__main__":
    main()
