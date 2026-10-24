import os
from collections import defaultdict
import yaml
import pytest
import torch
import matplotlib.pyplot as plt
from demoparser2 import DemoParser, WantedPropState

def test_map_func():
    from function.util_func import get_maps
    assert get_maps("d:\\Project\\Research\\origin_data\\mouse_data") == 10


def test_pt_file():
    data = torch.load("D:\Project\Research\pt_data\FalleN\map1\keyboard\\r1_seg1_kb.pt")
    print(type(data))
    print(data)
    data = torch.load("D:\Project\Research\pt_data\FalleN\map1\mouse\\r1_seg1_ms.pt")
    print(type(data))
    print(data)


def test_file_name():
    from function.util_func import players_in_dataset
    players_in_dataset()


def test_make_graph():
    from function.util_func import make_graph
    make_graph("d:\\Project\\Research\\output\\total_performance.json")

PLAYERS = ['apEX', 'FalleN', 'flameZ', 'KSCERATO', 'mezii', 'molodoy', 'ropz', 'YEKINDAR', 'yuurih', 'ZywOo']

def test_get_total_file():
    csv_files = []
    for root, _, files in os.walk("d:\\Project\\Research\\dataset\\extracted_data\\ZywOo"):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    print(f"Found {len(csv_files)} csv files")

import seaborn as sns

def test_graph_size():
    # 海报风格（只影响视觉，不改变数据）
    sns.set_context("poster", font_scale=1.4)

    plt.figure(figsize=(12, 6))

    # 全局字体（和你40pt正文匹配）
    plt.rcParams.update({
        'font.size': 36,
        'axes.titlesize': 40,
        'axes.labelsize': 36,
        'xtick.labelsize': 34,
        'ytick.labelsize': 34,
        'legend.fontsize': 32
    })
    a = [1,1,1,1,1]
    plt.plot(a)

    plt.title("10-fold Cross Validation Summary", fontweight='bold', pad=20)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")

    plt.legend()
    plt.tight_layout()
    plt.savefig(("test.png"), dpi=300)
    plt.close()


def test_new_pt_file():
    pt_root = "D:\\Project\\Research\\dataset\\pt_data\\ Lucyyyyy"
    if not os.path.exists(pt_root):
        print(f"\n[ERROR] pt_root directory does not exist: {pt_root}")
        return

    pt_files = []
    for root, _, files in os.walk(pt_root):
        for f in files:
            if f.endswith(".pt"):
                pt_files.append(os.path.join(root, f))

    if not pt_files:
        print("\n[WARNING] No .pt files found in the dataset.")
        return

    print(f"\n[INFO] Found {len(pt_files)} total .pt files in {pt_root}.")
    # Inspect the first 3 files
    for path in pt_files[:3]:
        rel_path = os.path.relpath(path, pt_root)
        print(f"\nInspecting: {rel_path}")
        try:
            tensor = torch.load(path)
            print(f"  Type:  {type(tensor)}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype: {tensor.dtype}")
            print("  Content (First 5 ticks):")
            print(tensor[:5])
        except Exception as e:
            print(f"  Error loading file: {e}")


def test_count_player_demos():
    """测试检索所有 demo 文件中每名玩家对应的 demo 数量"""
    demo_dir = "d:\\Project\\demo_file\\leisure"

    if not os.path.exists(demo_dir):
        print(f"[ERROR] Demos directory does not exist: {demo_dir}")
        return

    # 1. 递归查找所有 .dem 文件（与 parse_demo_to_pt.py 保持一致）
    demo_files = []
    for root, _, files in os.walk(demo_dir):
        for file in files:
            if file.endswith(".dem"):
                demo_files.append(os.path.join(root, file))

    print(f"\nFound {len(demo_files)} CS2 demo files in {demo_dir}.")
    if not demo_files:
        return

    player_counts = defaultdict(int)

    # 2. 遍历每个 demo 文件，获取该 demo 中的玩家
    for idx, demo_path in enumerate(demo_files, 1):
        filename = os.path.basename(demo_path)
        print(f"[{idx}/{len(demo_files)}] Parsing demo: {filename}")
        try:
            parser = DemoParser(demo_path)
            # 优先使用高效的 parse_player_info()
            player_info = parser.parse_player_info()
            if player_info is not None and not player_info.empty and "name" in player_info.columns:
                demo_players = set(player_info["name"].dropna().unique())
            else:
                # 备用方案：按 parse_demo_to_pt.py 的 WantedPropState("is_alive", True) 提取
                df_telemetry = parser.parse_ticks(["name"], prop_states=[WantedPropState("is_alive", True)])
                demo_players = set(df_telemetry["name"].dropna().unique()) if not df_telemetry.empty else set()
        except Exception as e:
            print(f"[WARNING] Failed parsing demo {filename}: {e}")
            continue

        for player_name in demo_players:
            if player_name and isinstance(player_name, str) and player_name.strip():
                player_counts[player_name] += 1

    # 打印排版整齐的统计结果
    print("\n" + "=" * 40)
    print(f"{'Player Name':<25} | {'Demo Count':<10}")
    print("=" * 40)
    for player, count in sorted(player_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{player:<25} | {count:<10}")
    print("=" * 40 + "\n")