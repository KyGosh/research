import os
import argparse
import json
import torch
from tqdm import tqdm
import multiprocessing
from functools import partial

# 特征索引映射
FEATURE_MAP = {
    "kb_back": 0, "kb_right": 1, "kb_forward": 2, "kb_left": 3,
    "fire": 4, "dp": 5, "dy": 6, "speed": 7, "acc": 8, "jerk": 9
}

'''
更新数据，将原始的浮点数数据进行分类
'''

def finalize_single_pt(pt_path, config):
    """
    对单个 .pt 文件应用离散化逻辑
    """
    try:
        # 1. Load raw tensor
        x = torch.load(pt_path)
        
        use_abs = config.get("use_abs", True)
        ignore_zeros = config.get("ignore_zeros", True)
        thresholds_dict = config.get("thresholds", {})
        
        for feat_name, t_list in thresholds_dict.items():
            if feat_name not in FEATURE_MAP: continue
            col_idx = FEATURE_MAP[feat_name]
            raw_v = x[:, col_idx]
            
            # 记录符号
            sign = torch.sign(raw_v)
            
            # 准备比较值
            v_for_bucket = torch.abs(raw_v) if use_abs else raw_v
            q_bins = torch.tensor(t_list, dtype=torch.float32)
            
            if ignore_zeros:
                active_mask = torch.abs(raw_v) > 1e-6
                bins = torch.zeros_like(raw_v)
                if active_mask.any():
                    # 映射到 1..N
                    bins[active_mask] = (torch.bucketize(v_for_bucket[active_mask].contiguous(), q_bins) + 1).float()
            else:
                bins = torch.bucketize(v_for_bucket.contiguous(), q_bins).float()
            
            if use_abs:
                x[:, col_idx] = bins * sign
            else:
                x[:, col_idx] = bins
        
        # 2. Overwrite or save to a target path
        # 这里我们选择直接覆盖原始 pt，因为用户明确要求“序列化数据就是目标数据”
        # 或者您可以指定一个新的目录。为了稳妥，我们存入原文件名加后缀，或者直接覆盖。
        # 这里采用直接覆盖的方式。
        torch.save(x, pt_path)
        return True
    except Exception as e:
        print(f"Error processing {pt_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Finalize features by applying thresholds offline.")
    parser.add_argument("--features-dir", type=str, default=os.path.join("d:\\", "Project", "Research", "serialized_features"))
    parser.add_argument("--thresholds-json", type=str, default=os.path.join("d:\\", "Project", "Research", "serialized_features", "dynamic_thresholds.json"))
    parser.add_argument("--workers", type=int, default=os.cpu_count() - 1)
    args = parser.parse_args()

    if not os.path.exists(args.thresholds_json):
        print("Thresholds JSON not found. Please run find_threshold.py first.")
        return

    with open(args.thresholds_json, 'r') as f:
        config = json.load(f)

    # 递归查找所有 .pt 文件
    print("Scanning for .pt files...")
    all_files = []
    for root, _, files in os.walk(args.features_dir):
        for f in files:
            if f.endswith(".pt"):
                all_files.append(os.path.join(root, f))
    
    print(f"Found {len(all_files)} files. Applying thresholds...")

    # 并行处理
    func = partial(finalize_single_pt, config=config)
    
    with multiprocessing.Pool(max(1, args.workers)) as pool:
        results = list(tqdm(pool.imap_unordered(func, all_files), total=len(all_files)))
    
    success_count = sum(results)
    print(f"\nSuccessfully finalized {success_count}/{len(all_files)} feature files.")
    print("Now train_auth_v2.py will run significantly faster!")

if __name__ == "__main__":
    main()
