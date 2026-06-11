"""调试冻结期提取"""
from demoparser2 import DemoParser
import pandas as pd

parser = DemoParser("./data/demo_file/m1.dem")

# 获取第一轮的信息
round_starts = parser.parse_event("round_start", other=["tick", "game_time"])
print("=== Round Starts ===")
print(round_starts.head(3))
print()

# 获取第一轮的冻结期数据
first_round_start = int(round_starts.iloc[0]['tick'])
print(f"First round starts at tick: {first_round_start}")

# 获取从回合开始到 +500 的 ticks 中 is_freeze_period 的信息
search_ticks = list(range(first_round_start, first_round_start + 300, 5))
print(f"Searching ticks: {search_ticks[:10]}...")

df = parser.parse_ticks(["is_freeze_period"], ticks=search_ticks)
print("\n=== Freeze Period Data ===")
print(df.head(20))

# 按 tick 排序并去重
df_sorted = df.drop_duplicates(subset=['tick']).sort_values('tick')
print("\n=== After Sorting and Dedup ===")
print(df_sorted.head(20))

# 检查冻结期的变化
print("\n=== Freeze Status Changes ===")
prev_frozen = None
for idx, row in df_sorted.iterrows():
    is_frozen = row['is_freeze_period']
    tick = int(row['tick'])
    if prev_frozen != is_frozen:
        print(f"Tick {tick}: {prev_frozen} -> {is_frozen}")
        if prev_frozen is True and is_frozen is False:
            print(f"  *** FREEZE END at tick {tick} ***")
    prev_frozen = is_frozen
