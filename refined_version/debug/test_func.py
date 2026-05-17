import os

import pytest
import torch
import matplotlib.pyplot as plt

def test_map_func():
    from refined_version.function.util_func import get_maps
    assert get_maps("d:\\Project\\Research\\origin_data\\mouse_data") == 10


def test_pt_file():
    data = torch.load("D:\Project\Research\pt_data_origin\FalleN\map1\keyboard\\r1_seg1_kb.pt")
    print(type(data))
    print(data)
    data = torch.load("D:\Project\Research\pt_data_origin\FalleN\map1\mouse\\r1_seg1_ms.pt")
    print(type(data))
    print(data)


def test_file_name():
    from refined_version.function.util_func import players_in_dataset
    players_in_dataset()


def test_make_graph():
    from refined_version.function.util_func import make_graph
    make_graph("d:\\Project\\Research\\output\\total_performance.json")

PLAYERS = ['apEX', 'FalleN', 'flameZ', 'KSCERATO', 'mezii', 'molodoy', 'ropz', 'YEKINDAR', 'yuurih', 'ZywOo']

def test_get_total_file():
    csv_files = []
    for root, _, files in os.walk("d:\\Project\\Research\\test_data\\ZywOo"):
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