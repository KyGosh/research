import os

import pytest
import torch

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


def test_get_total_file():
    csv_files = []
    for root, _, files in os.walk("d:\\Project\\Research\\test_data\\apEX"):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    print(f"Found {len(csv_files)} csv files")
