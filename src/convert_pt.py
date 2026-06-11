# generate .pt files
import multiprocessing
import os
import shutil
from functools import partial

import pandas as pd
import torch
from tqdm import tqdm

def process_single(csv_file: str, csv_data_root: str, pt_dir_root: str):
    try:
        # 1. 修正路径逻辑：保留相对于 csv_data_root 的子目录结构
        rel_path = os.path.relpath(csv_file, csv_data_root)
        pt_path = os.path.join(pt_dir_root, rel_path.replace(".csv", ".pt"))
        os.makedirs(os.path.dirname(pt_path), exist_ok=True)

        # 2. 读取 CSV，显式处理布尔值
        df = pd.read_csv(
            csv_file,
            true_values=["TRUE", "True"],
            false_values=["FALSE", "False"]
        ).drop(columns=["name", "steamid", "tick"])

        # 4. 显式转换为数值类型
        for col in df.columns:
            # 将布尔/字符串 True/False 转换为 1.0/0.0
            if df[col].dtype == bool or df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 5. 处理缺失值（参考 datasets.py）
        df_selected = df.bfill().fillna(0)

        # 6. 转换为 numpy 并裁剪到 640 ticks (参考 serialize_features.py)
        data_np = df_selected.values.astype("float32")
        if data_np.shape[0] > 640:
            data_np = data_np[:640, :]
        elif data_np.shape[0] < 640:
            pass

        # 7. 保存为原始 Tensor（ManifestDataset 期望的格式）
        tensor = torch.from_numpy(data_np)
        torch.save(tensor, pt_path)

        return pt_path

    except Exception as e:
        print(f"[ERROR] {csv_file}: {e}")
        return None


import yaml
with open("../setting.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

EXTRACTED_DIR = cfg["path"]["extract"]
PT_DIR = cfg["path"]["pt"]

PROCESSORS = multiprocessing.cpu_count() - 1

def main():
    if not os.path.exists(EXTRACTED_DIR):
        print(f"Error: {EXTRACTED_DIR} does not exist.")
        return
    print(f"Scanning directories: {EXTRACTED_DIR}")

    csv_files = []
    for root, _, files in os.walk(EXTRACTED_DIR):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    print(f"Found {len(csv_files)} csv files")

    if os.path.isdir(PT_DIR):
        shutil.rmtree(PT_DIR)

    # Process in parallel
    process_func = partial(process_single, csv_data_root=EXTRACTED_DIR, pt_dir_root=PT_DIR)

    print(f"Processing with {max(1, PROCESSORS)} workers...")
    with multiprocessing.Pool(max(1, PROCESSORS)) as pool:
        for _ in tqdm(pool.imap_unordered(process_func, csv_files), total=len(csv_files)):
            pass

    print(f"Done! Files saved to {PT_DIR}")
    return


if __name__ == '__main__':
    main()