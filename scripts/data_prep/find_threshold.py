import os
import argparse
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# 特征索引映射 (基于 serialized_features 中的 .pt 文件结构)
# kb: 0-3 | ms: 4:fire, 5:dp, 6:dy, 7:speed, 8:acc, 9:jerk
FEATURE_MAP = {
    "kb_back": 0, "kb_right": 1, "kb_forward": 2, "kb_left": 3,
    "fire": 4, "dp": 5, "dy": 6, "speed": 7, "acc": 8, "jerk": 9
}

def calculate_feature_thresholds(meta_path, feature_names, num_classes=5, use_abs=True, sample_ratio=0.1, ignore_zeros=False):
    """
    动态计算特征阈值。
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)
    
    if sample_ratio < 1.0:
        df_meta = df_meta.sample(frac=sample_ratio, random_state=42)
    
    file_paths = df_meta['filepath'].tolist()
    print(f"Sampling {len(file_paths)} segments for threshold calculation...")

    feature_data = {name: [] for name in feature_names}
    indices = [FEATURE_MAP[name] for name in feature_names]

    for path in tqdm(file_paths, desc="Collecting data"):
        if not os.path.exists(path):
            continue
        data = torch.load(path).numpy()
        for name, idx in zip(feature_names, indices):
            vals = data[:, idx]
            if use_abs:
                vals = np.abs(vals)
            feature_data[name].extend(vals[::5].tolist())

    if ignore_zeros:
        q_steps = np.linspace(0, 1, num_classes)[1:-1]
    else:
        q_steps = np.linspace(0, 1, num_classes + 1)[1:-1]
    
    results = {}
    valid_feature_data = {}
    print(f"\n--- Calculated Thresholds {'(Ignoring Zeros)' if ignore_zeros else ''} ---")
    for name in feature_names:
        arr = np.array(feature_data[name])
        
        if ignore_zeros:
            if use_abs:
                active_data = arr[arr > 1e-6]
            else:
                active_data = arr[np.abs(arr) > 1e-6]

            if len(active_data) == 0:
                print(f"  {name}: all zero, skipped")
                continue

            quantiles = np.quantile(active_data, q_steps)
        else:
            quantiles = np.quantile(arr, q_steps)
            
        # 去重（防止塌缩）
        unique_quantiles = sorted(set(quantiles.tolist()))
        results[name] = unique_quantiles
        valid_feature_data[name] = arr
        
        print(f"Feature: {name} ({num_classes} classes requested, {len(unique_quantiles)+1} actual)")
        if ignore_zeros:
            print(f"  Class 0: Exactly 0")
        for idx, val in enumerate(unique_quantiles):
            print(f"  Threshold {idx+1}: {val:.6f}")
            
    return results, valid_feature_data

def plot_distributions(feature_data, thresholds, num_classes, use_abs, ignore_zeros, out_dir):
    num_features = len(feature_data)
    if num_features == 0: return
    
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 5 * num_features))
    if num_features == 1:
        axes = [axes]
        
    for i, (name, data) in enumerate(feature_data.items()):
        ax = axes[i]
        arr = np.array(data)
        ax.hist(arr, bins=100, color='skyblue', edgecolor='black', alpha=0.7, log=True)
        
        feat_thresholds = thresholds[name]
        for idx, t in enumerate(feat_thresholds):
            ax.axvline(x=t, color='red', linestyle='--', linewidth=1.5)
            ax.text(t, ax.get_ylim()[1]*0.5, f' T{idx+1}', color='red', fontweight='bold')
            
        title_extra = " (Ignoring Zeros)" if ignore_zeros else ""
        title_suffix = " (Absolute)" if use_abs else ""
        ax.set_title(f"Distribution of {name}{title_suffix}{title_extra}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency (Log Scale)")
        ax.grid(True, which='both', linestyle=':', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "feature_distributions.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\nDistribution plot saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Find dynamic thresholds with deduplication.")
    parser.add_argument("--meta-path", type=str, default=os.path.join("d:\\", "Project", "Research", "serialized_features", "feature_metadata.csv"))
    parser.add_argument("--features", type=str, default="speed,dp,dy,acc,jerk", help="Comma separated feature names")
    parser.add_argument("--num-classes", type=int, default=5, help="Target number of classes")
    parser.add_argument("--no-abs", action="store_false", dest="use_abs", help="Do not use absolute values")
    parser.add_argument("--sample-ratio", type=float, default=0.2, help="Ratio of segments to use")
    parser.add_argument("--ignore-zeros", action="store_true", default="True", help="Ignore zero values when calculating quantiles")
    parser.add_argument("--out-dir", type=str, default=os.path.join("d:\\", "Project", "Research", "serialized_features"))
    args = parser.parse_args()

    feature_list = [f.strip() for f in args.features.split(",")]
    
    thresholds, raw_data = calculate_feature_thresholds(
        args.meta_path, 
        feature_list, 
        num_classes=args.num_classes, 
        use_abs=args.use_abs,
        sample_ratio=args.sample_ratio,
        ignore_zeros=args.ignore_zeros
    )

    plot_distributions(raw_data, thresholds, args.num_classes, args.use_abs, args.ignore_zeros, args.out_dir)

    out_json = os.path.join(args.out_dir, "dynamic_thresholds.json")
    with open(out_json, "w") as f:
        json.dump({
            "target_num_classes": args.num_classes,
            "use_abs": args.use_abs,
            "ignore_zeros": args.ignore_zeros,
            "thresholds": thresholds
        }, f, indent=4)
    
    print(f"JSON results saved to {out_json}")

if __name__ == "__main__":
    main()
