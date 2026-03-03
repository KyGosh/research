import os
import argparse
import re
import shutil
from typing import Dict, List, Tuple, Iterable, Set

import pandas as pd

# ------------------------------------------------------------
# 1. 找到每个 map 的 keyboard / mouse / time 文件
# ------------------------------------------------------------
def find_files(origin_dir: str) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}

    ks_dir = os.path.join(origin_dir, "keystroke_data")
    ms_dir = os.path.join(origin_dir, "mouse_data")
    tm_dir = os.path.join(origin_dir, "time_data")

    def collect(dir_path: str, key_name: str):
        for root, _, files in os.walk(dir_path):
            for f in files:
                m = re.search(r"_m(\d+)\.csv$", f)
                if not m:
                    continue
                map_key = f"map{m.group(1)}"
                result.setdefault(map_key, {})
                result[map_key][key_name] = os.path.join(root, f)

    collect(ks_dir, "keyboard")
    collect(ms_dir, "mouse")
    collect(tm_dir, "time")

    return result

# ------------------------------------------------------------
# 2. 根据 time.csv 生成每一轮的时间段
# 支持 overlap
# ------------------------------------------------------------
def load_time_segments(time_csv: str, segment_ticks: int, overlap_ratio: float = 0.0) -> Dict[int, List[Tuple[int, int]]]:

    df = pd.read_csv(time_csv)
    segments: Dict[int, List[Tuple[int, int]]] = {}

    step = int(segment_ticks * (1 - overlap_ratio))
    if step <= 0:
        raise ValueError("overlap_ratio 太大")

    for _, row in df.iterrows():
        freeze_end = int(row["freeze_end"])
        end = int(row["end"])
        round_num = int(row["round_num"])

        segs: List[Tuple[int, int]] = []
        start = freeze_end

        while start + segment_ticks - 1 <= end:
            segs.append((start, start + segment_ticks - 1))
            start += step

        if segs:
            segments[round_num] = segs

    return segments

# ------------------------------------------------------------
# 3. 收集所有玩家名字
# ------------------------------------------------------------
def collect_names(files: Iterable[str]) -> Set[str]:
    names: Set[str] = set()

    for path in files:
        if not os.path.exists(path):
            continue

        for chunk in pd.read_csv(path, usecols=["name"], chunksize=200000):
            names.update(chunk["name"].astype(str).unique())

    return names

# ------------------------------------------------------------
# 4. 建立目录结构：map/player/keyboard mouse
# ------------------------------------------------------------
def ensure_dirs(out_root: str, map_key: str, names: Iterable[str]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}

    for name in names:
        base = os.path.join(out_root, map_key, name)
        kb_dir = os.path.join(base, "keyboard")
        ms_dir = os.path.join(base, "mouse")

        os.makedirs(kb_dir, exist_ok=True)
        os.makedirs(ms_dir, exist_ok=True)

        mapping[name] = {
            "keyboard": kb_dir,
            "mouse": ms_dir,
        }

    return mapping

# ------------------------------------------------------------
# 5. 按 segment 切分并写入文件
# ------------------------------------------------------------
def filter_and_write(
    data_csv: str,
    name: str,
    segments_by_round: Dict[int, List[Tuple[int, int]]],
    out_dir: str,
    tag: str,
    map_key: str,
) -> None:

    df = pd.read_csv(data_csv)
    sub = df[df["name"] == name]

    if sub.empty:
        return

    tick_set = set(sub["tick"].values)
    seg_index = 1

    for r in sorted(segments_by_round.keys()):
        for start, end in segments_by_round[r]:

            if end not in tick_set:
                break

            mask = (sub["tick"] >= start) & (sub["tick"] <= end)
            part = sub.loc[mask]

            if part.empty:
                continue

            fname = f"{map_key}_r{r}_seg{seg_index:03d}_{tag}.csv"
            part.to_csv(os.path.join(out_dir, fname), index=False)

            seg_index += 1

# ------------------------------------------------------------
# 6. 主处理流程
# ------------------------------------------------------------
def process(origin_dir: str, out_dir: str, segment_ticks: int, overlap_ratio: float) -> None:
    files = find_files(origin_dir)

    # 收集所有玩家
    names = collect_names([v for m in files.values() for v in [m.get("keyboard"), m.get("mouse")] if v])

    for map_key in sorted(files.keys()):
        time_file = files[map_key].get("time")
        keyboard_file = files[map_key].get("keyboard")
        mouse_file = files[map_key].get("mouse")

        if not time_file or not keyboard_file or not mouse_file:
            continue

        print(f"Processing {map_key}...")

        # 建立目录
        name_dirs = ensure_dirs(out_dir, map_key, names)

        # 生成时间段
        segments_by_round = load_time_segments(time_file, segment_ticks, overlap_ratio)

        for name in names:
            filter_and_write(
                keyboard_file,
                name,
                segments_by_round,
                name_dirs[name]["keyboard"],
                "kb",
                map_key,
            )

            filter_and_write(
                mouse_file,
                name,
                segments_by_round,
                name_dirs[name]["mouse"],
                "ms",
                map_key,
            )

# ------------------------------------------------------------
# 7. main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--origin-dir",
        type=str,
        default=os.path.join("d:\\", "Project", "Research", "origin_data"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("d:\\", "Project", "Research", "processed_data"),
    )
    parser.add_argument("--segment-ticks", type=int, default=640)
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.0,  # 0 表示不重叠
        help="0.5 表示 50% overlap",
    )
    args = parser.parse_args()

    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    process(args.origin_dir, args.out_dir, args.segment_ticks, args.overlap_ratio)

if __name__ == "__main__":
    main()