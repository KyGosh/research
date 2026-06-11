"""探查 demoparser 中可用的事件和字段"""
from demoparser2 import DemoParser
import os

demo_path = "./data/demo_file/m1.dem"

if os.path.exists(demo_path):
    parser = DemoParser(demo_path)
    
    print("=== round_start 事件 ===")
    try:
        df = parser.parse_event("round_start", other=["tick", "game_time"])
        print(df)
        print(f"共 {len(df)} 个事件\n")
    except Exception as e:
        print(f"错误: {e}\n")
    
    print("=== round_end 事件 ===")
    try:
        df = parser.parse_event("round_end", other=["tick", "game_time"])
        print(df)
        print(f"共 {len(df)} 个事件\n")
    except Exception as e:
        print(f"错误: {e}\n")
    
    print("=== 尝试从 ticks 提取冻结期变化 ===")
    try:
        # 获取少量 tick 来看 is_freeze_period 的值
        df = parser.parse_ticks(["is_freeze_period", "tick"], ticks=list(range(0, 100, 10)))
        print(df)
    except Exception as e:
        print(f"错误: {e}")
else:
    print(f"找不到 demo 文件: {demo_path}")
