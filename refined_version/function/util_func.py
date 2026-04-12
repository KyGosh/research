import json
import os
import re

import numpy as np
import pandas as pd


# 确认原始数据从map1开始连续不间断
def get_maps(directory: str) -> int:
    max_map = -1
    pattern = re.compile(r"_m(\d+)\.csv$")
    for _, _, files in os.walk(directory):
        for f in files:
            if m := pattern.search(f):
                max_map = max(max_map, int(m.group(1)))
    return max_map

# 按地图处理数据
def process_map_data(map_idx, mouse_dir, time_dir):
    mouse_file = os.path.join(mouse_dir, f"mouse_m{map_idx}.csv")
    time_file = os.path.join(time_dir, f"time_m{map_idx}.csv")

    if not os.path.exists(mouse_file) or not os.path.exists(time_file):
        print(f"Warning: Missing data for map{map_idx}")
        return pd.DataFrame()

    # 读取数据
    df_mouse = pd.read_csv(mouse_file)
    df_time = pd.read_csv(time_file)

    # 方案A：先按玩家分组计算特征
    # 确保数据按玩家和时间排序，以便 diff 正确执行
    df_mouse = df_mouse.sort_values(["name", "tick"])

    print(f"Calculating features for map{map_idx}...")
    # 使用 groupby().diff() 替代 apply() 以消除警告并提升性能
    mouse_gb = df_mouse.groupby("name")
    df_mouse["dp"] = mouse_gb["pitch"].diff().abs()
    dy_raw = mouse_gb["yaw"].diff()
    df_mouse["dy"] = ((dy_raw + 180) % 360 - 180).abs()
    df_mouse["speed"] = np.sqrt(df_mouse["dp"] ** 2 + df_mouse["dy"] ** 2)

    # 重新计算后续依赖列，同样取绝对值
    df_mouse["acc"] = df_mouse.groupby("name")["speed"].diff().abs()
    df_mouse["jerk"] = df_mouse.groupby("name")["acc"].diff().abs()

    # 根据 time_data 过滤时间窗口 [freeze_end, end]
    filtered_chunks = []
    for _, row in df_time.iterrows():
        mask = (df_mouse["tick"] >= row["freeze_end"]) & (df_mouse["tick"] <= row["end"])
        filtered_chunks.append(df_mouse[mask])

    if not filtered_chunks:
        return pd.DataFrame()

    return pd.concat(filtered_chunks, axis=0)

def discretize_mouse_data(df: pd.DataFrame) -> pd.DataFrame:
    with open("d:\\Project\Research\output\discretized_level.json", "r") as f:
        config = json.load(f)

    for feature, level in config.items():
        if feature not in df.columns:
            print(f"Discretizing {feature} err, not exist...")
            continue

        raw_v = df[feature]
        sign = np.sign(raw_v)
        raw_v = raw_v.abs()
        q_bins = np.array(level)

        inf_mask = raw_v > 1e-6
        bins = pd.Series(0.0, index=df.index)

        if inf_mask.any():
            bins.loc[inf_mask] = np.digitize(raw_v[inf_mask], q_bins) + 1

        df[feature] = bins * sign

    return df


def list_maps(processed_root: str) -> list[str]:
    maps = []
    for d in os.listdir(processed_root):
        p = os.path.join(processed_root, d)
        if not os.path.isdir(p):
            continue
        if re.match(r"^map\d+$", d):
            maps.append(d)
    maps.sort(key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0)
    return maps