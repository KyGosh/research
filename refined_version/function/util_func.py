import os
import re

# 确认原始数据从map1开始连续不间断
def get_maps(directory: str) -> int:
    max_map = -1
    pattern = re.compile(r"_m(\d+)\.csv$")
    for _, _, files in os.walk(directory):
        for f in files:
            if m := pattern.search(f):
                max_map = max(max_map, int(m.group(1)))
    return max_map