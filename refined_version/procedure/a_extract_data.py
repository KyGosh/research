import argparse
import os
import re
import shutil
from typing import Tuple

import numpy as np
import pandas as pd

'''
功能：提取数据，将原始的多名玩家糅合的数据转换为每个独立玩家的数据
处理后的结果：
-name
--map1
---keyboard
---mouse
---combined
并确保每个csv文件的大小均满足目标窗口大小
ps. 可以在后续添加增量功能，而不是每次生成都要删除原始文件
'''

# [map*, [(kb, ms, time), directory]], [map*, [playerA, ...]]
def sort_by_map(directory: str) -> Tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    map_data: dict[str, dict[str, str]] = {}
    map_players: dict[str, list[str]] = {}

    keyboard = os.path.join(directory, "keystroke_data")
    mouse = os.path.join(directory, "mouse_data")
    time = os.path.join(directory, "time_data")

    def collect(target: str, key_name: str):
        for root, _, files in os.walk(target):
            for f in files:
                m = re.search(r"_m(\d+)\.csv$", f) # 原始数据命名要求
                if not m:
                    continue
                map_key = f"map{m.group(1)}"
                map_data.setdefault(map_key, {})
                file_path = os.path.join(root, f)
                map_data[map_key][key_name] = file_path

                if (key_name == "kb" or key_name == "ms") and (map_key not in map_players):
                    try:
                        df = pd.read_csv(file_path)
                        if "name" in df.columns:
                            names = df["name"].dropna().unique()
                            map_players[map_key] = names
                    except Exception as e:
                        print(f"read error: {file_path}, {e}")

    collect(keyboard, "kb")
    collect(mouse, "ms")
    collect(time, "time")

    return map_data, map_players

# name / map / keyboard | mouse | keyboard+mouse
def create_dir(player_in_map: list[str], map_key: str, out_dir: str) -> dict[str, dict[str, str]]:
    ans: dict[str, dict[str, str]] = {}

    for player in player_in_map:
        base = os.path.join(out_dir, player, map_key)
        kb_dir = os.path.join(base, "keyboard")
        ms_dir = os.path.join(base, "mouse")
        combined_dir = os.path.join(base, "combined")

        os.makedirs(kb_dir, exist_ok=True)
        os.makedirs(ms_dir, exist_ok=True)
        os.makedirs(combined_dir, exist_ok=True)

        ans[player] = {
            "kb": kb_dir,
            "ms": ms_dir,
            "combined": combined_dir
        }

    return ans

def get_windows(time_file: str, window_size: int, overlap_ratio: float) -> dict[int, list[tuple[int, int]]]:
    df = pd.read_csv(time_file)
    windows: dict[int, list[tuple[int, int]]] = {}

    # step gets less as overlap raises
    step = int(window_size * (1 - overlap_ratio))

    for _, row in df.iterrows():
        round_num = int(row["round_num"])
        freeze_end = int(row["freeze_end"])
        end = int(row["end"])

        segments: list[tuple[int, int]] = []
        start = freeze_end

        while start + window_size - 1 < end:
            segments.append((start, start + window_size - 1))
            start += step

        if segments:
            windows[round_num] = segments

    return windows

def save_player_segments(
        windows: dict[int, list[tuple[int, int]]],
        out_dir: dict[str, str],
        player_dfs: dict[str, pd.DataFrame],
        window_size: int
) -> None:
    for category, df in player_dfs.items():
        tick_set = set(df["tick"].values)
        seg_idx = 1

        for r in sorted(windows.keys()):
            for start, end in windows[r]:
                # only record avatar live time, not necessary, size check do the same thing
                if end not in tick_set:
                    break

                mask = (df["tick"] >= start) & (df["tick"] <= end)
                part = df.loc[mask]

                if part.empty:
                    break
                # make sure all the data correspond to window's size
                if part["tick"].nunique() != window_size:
                    continue

                f_name = f"r{r}_seg{seg_idx}_{category}.csv"
                part.to_csv(os.path.join(out_dir[category], f_name), index=False)
                seg_idx += 1

# add more characteristics into mouse data
def extract_mouse_characteristics(raw_file: str) -> pd.DataFrame:
    df_raw = pd.read_csv(raw_file)

    df_raw["dp"] = df_raw.groupby("name")["pitch"].diff()
    df_raw["dy"] = df_raw.groupby("name")["yaw"].diff()
    df_raw["speed"] = np.sqrt(df_raw["dp"] ** 2 + df_raw["dy"] ** 2)
    df_raw["acc"] = df_raw.groupby("name")["speed"].diff()
    df_raw["jerk"] = df_raw.groupby("name")["acc"].diff()

    df = df_raw.bfill().fillna(0)

    return df


def extract_data(origin_dir: str, out_dir: str, segment_ticks: int, overlap_ratio: float) -> None:
    # 1. categorize origin data into respective maps
    data_dict, players_dict = sort_by_map(origin_dir)

    # 2. handle
    for map_k in sorted(data_dict.keys()):
        # get all data about this map
        time_f = data_dict[map_k]["time"]
        kb_f = data_dict[map_k]["kb"]
        ms_f = data_dict[map_k]["ms"]

        name_ls = players_dict[map_k]

        if not all([time_f, kb_f, ms_f, name_ls is not None]):
            print(f"Skipping {map_k} due to missing data.")
            continue

        print(f"Processing {map_k}")

        # establish directory, iterate through all maps
        directory = create_dir(name_ls, map_k, out_dir)

        # get target time windows
        segments_by_round = get_windows(time_f, segment_ticks, overlap_ratio)

        # read current map data into dataframe
        try:
            df_kb = pd.read_csv(kb_f)
            df_ms = extract_mouse_characteristics(ms_f)
            df_combined = pd.merge(df_kb, df_ms, on=["tick", "steamid", "name"], how="inner")

            kb_groups = df_kb.groupby("name")
            ms_groups = df_ms.groupby("name")
            combined_groups = df_combined.groupby("name")
        except Exception as e:
            print(f"Error loading data for {map_k}: {e}")
            continue

        # write files into directory following this map's players
        print("Start writing files...")
        for name in name_ls:
            player_dfs = {
                "kb": kb_groups.get_group(name),
                "ms": ms_groups.get_group(name),
                "combined": combined_groups.get_group(name)
            }
            save_player_segments(segments_by_round, directory[name], player_dfs, segment_ticks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-dir", type=str,
                        default=os.path.join("d:\\", "Project", "Research", "origin_data"))
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join("d:\\", "Project", "Research", "test_data"))
    # the size of time window, 1 s = 64 tick
    parser.add_argument("--segment-ticks", type=int, default=640)
    # 0.0 ==> 1 ~ 640, 640 ~ 1280 || 0.5 ==> 1 ~ 640, 320 ~ 960
    parser.add_argument("--overlap-ratio", type=float, default=0.0)
    args = parser.parse_args()

    # parameter validation
    if not os.path.exists(args.origin_dir):
        print(f"{args.origin_dir} does not exist")
        return
    if args.overlap_ratio < 0.0 or args.overlap_ratio > 1.0:
        print(f"{args.overlap_ratio} is outside of [0, 1]")
        return

    # delete out_dir each time to avoid redundant data for different arguments
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)

    extract_data(args.origin_dir, args.out_dir, args.segment_ticks, args.overlap_ratio)

    print("Extract data process finished.")
    return

if __name__ == "__main__":
    main()