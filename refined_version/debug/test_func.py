import pytest

def test_map_func():
    from refined_version.function.util_func import get_maps
    assert get_maps("d:\\Project\\Research\\origin_data\\mouse_data") == 10