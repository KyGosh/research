import pytest
import torch


def test_map_func():
    from refined_version.function.util_func import get_maps
    assert get_maps("d:\\Project\\Research\\origin_data\\mouse_data") == 10


def test_pt_file():
    data = torch.load("D:\Project\Research\pt_data\FalleN\map1\keyboard\\r1_seg1_kb.pt")
    print(type(data))
    print(data)
    data = torch.load("D:\Project\Research\pt_data\FalleN\map1\mouse\\r1_seg1_ms.pt")
    print(type(data))
    print(data)