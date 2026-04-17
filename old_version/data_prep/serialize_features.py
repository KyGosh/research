import os
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from functools import partial

# We will need the logic from datasets.py to list files and parse them.
# Adding the project root to path so we can import scripts.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.ml.datasets import list_name_dirs, build_segments_index, read_unified_sample, read_keyboard_csv, read_mouse_csv

def process_single_segment(item, processed_root, out_root):
    player_name, mk, r, s, kb_path, ms_path = item
    
    # 1. Read files
    kb_data = read_keyboard_csv(kb_path)
    ms_data = read_mouse_csv(ms_path)
    
    if kb_data is None or ms_data is None:
        return None

    # 一次清洗数据，清除所有不满足预期数据大小的样本
    if kb_data.shape[0] < 640 or ms_data.shape[0] < 640:
        return None # Skip short segments
        
    # Trim to exactly 640 ticks
    kb_data = kb_data[:640, :]
    ms_data = ms_data[:640, :]
    
    unified_data = np.concatenate([kb_data, ms_data], axis=1).astype(np.float32)
    
    # 2. Calculate empty row ratio (all features are 0)
    # Check if a row is completely 0 across all features
    zero_rows = np.all(unified_data == 0, axis=1)
    empty_ratio = np.mean(zero_rows)
    
    # 3. Create output directory structure
    # serialized_features/mapX/player_name/
    out_dir = os.path.join(out_root, mk, player_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as .pt file
    out_filename = f"{mk}_r{r}_seg{s}.pt"
    out_filepath = os.path.join(out_dir, out_filename)
    
    torch.save(torch.tensor(unified_data, dtype=torch.float32), out_filepath)
    
    # Also collect mouse change rates (speed) to compute global quantiles later
    # We will sample the speed column (index 7 in unified: kb has 4, so ms is 4-9, speed is index 4+3=7)
    # Let's be exact:
    # kb: BACK, RIGHT, FORWARD, LEFT (4 cols) -> idx 0,1,2,3
    # ms: fire, dp, dy, speed, acc, jerk (6 cols) -> idx 4,5,6,7,8,9
    speed_data = unified_data[:, 7]
    # Sample every 10th value to save memory when aggregating
    sampled_speed = speed_data[::10].tolist()
    
    return {
        "player": player_name,
        "map": mk,
        "round": r,
        "seg": s,
        "filepath": out_filepath,
        "empty_ratio": float(empty_ratio),
        "sampled_speed": sampled_speed
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=str, default=os.path.join("d:\\", "Project", "Research", "processed_data"))
    parser.add_argument("--out-root", type=str, default=os.path.join("d:\\", "Project", "Research", "serialized_features"))
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count() - 1)
    args = parser.parse_args()

    processed_root = args.processed_root
    out_root = args.out_root
    
    if not os.path.exists(processed_root):
        print(f"Error: {processed_root} does not exist.")
        return

    print("Scanning directories...")
    from scripts.ml.datasets import list_maps, list_name_dirs_in_map
    maps = list_maps(processed_root)
    
    # Build task list
    tasks = []
    for map_dir in maps:
        ndirs = list_name_dirs_in_map(processed_root, map_dir)
        idx = build_segments_index(ndirs)
        for player_name, segments in idx.items():
            for (mk, r, s), files in segments.items():
                kb_path = files.get("kb")
                ms_path = files.get("ms")
                if kb_path and ms_path:
                    tasks.append((player_name, mk, r, s, kb_path, ms_path))

    # 结合鼠标和键盘数据，此时不对样本长度做校验
    print(f"Found {len(tasks)} valid keyboard-mouse pairs.")
    
    results = []
    global_speeds = []
    
    # Process in parallel
    process_func = partial(process_single_segment, processed_root=processed_root, out_root=out_root)
    
    print(f"Processing with {max(1, args.workers)} workers...")
    with multiprocessing.Pool(max(1, args.workers)) as pool:
        for res in tqdm(pool.imap_unordered(process_func, tasks), total=len(tasks)):
            if res is not None:
                # Separate large list to avoid huge dataframe
                global_speeds.extend(res.pop("sampled_speed"))
                results.append(res)
                
    # Save metadata
    print("Saving metadata...")
    df_meta = pd.DataFrame(results)
    meta_path = os.path.join(out_root, "feature_metadata.csv")
    df_meta.to_csv(meta_path, index=False)
    
    # Calculate and save global quantiles for speed
    # 目前只找了0.2 0.4 0.6 0.8 等位置的值？没有修改实际训练数据
    print("Calculating global quantiles for mouse speed...")
    if global_speeds:
        speeds = np.array(global_speeds)
        # remove exact 0s if they dominate, but let's keep them for true distribution
        q_vals = [0.2, 0.4, 0.6, 0.8]
        quantiles = np.quantile(speeds, q_vals)
        
        print("Speed Quantiles:")
        for q, v in zip(q_vals, quantiles):
            print(f"  {int(q*100)}%: {v:.6f}")
            
        with open(os.path.join(out_root, "speed_quantiles.txt"), "w") as f:
            for q, v in zip(q_vals, quantiles):
                f.write(f"{q},{v}\n")
    
    print(f"Done! Processed {len(results)} segments.")
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()
