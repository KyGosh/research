"""
批量解析 demo 文件，提取玩家操作和游戏事件时间信息

输出两类 CSV 文件：
1. {demo_name}_player_ticks.csv - 玩家每 tick 的操作和视角信息
2. {demo_name}_game_events.csv - 游戏回合事件时间点（tick 形式）
"""
import os
import glob
import pandas as pd
from demoparser2 import DemoParser
from pathlib import Path

def extract_freeze_end_ticks(parser, round_starts):
    """
    通过监测 is_freeze_period 从 True 变为 False 来提取冻结期结束的 tick
    """
    freeze_ends = []
    
    # 获取每个 round 开始后的一些 ticks 来找冻结期结束
    for idx, row in round_starts.iterrows():
        start_tick = int(row['tick'])
        round_num = int(row['round'])
        
        # 在回合开始后的 1000 ticks 内查找冻结期结束
        search_ticks = list(range(start_tick, start_tick + 1000, 2))
        
        try:
            df = parser.parse_ticks(["is_freeze_period"], ticks=search_ticks)
            # 按 tick 排序，并去重
            df = df.drop_duplicates(subset=['tick']).sort_values('tick')
            
            # 找到从 True 变为 False 的位置
            prev_frozen = None
            for _, tick_row in df.iterrows():
                is_frozen = tick_row['is_freeze_period']
                current_tick = int(tick_row['tick'])
                
                if prev_frozen is True and is_frozen is False:
                    freeze_ends.append({
                        'round': round_num,
                        'freeze_end_tick': current_tick
                    })
                    break
                prev_frozen = is_frozen
        except Exception as e:
            pass
    
    return freeze_ends

def process_demo(demo_path, output_dir="./data/output"):
    """处理单个 demo 文件"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    demo_name = Path(demo_path).stem
    
    try:
        print(f"处理: {demo_name}")
        parser = DemoParser(demo_path)
        
        # ====== 文件 1: 玩家操作和视角 ======
        wanted_props = ["FORWARD", "BACK", "LEFT", "RIGHT", "FIRE", "pitch", "yaw", "tick"]
        player_ticks = parser.parse_ticks(wanted_props=wanted_props)
        
        # 重新排列列顺序并保留必要的列
        player_ticks = player_ticks[wanted_props]
        
        # 保存玩家操作数据
        player_output = os.path.join(output_dir, f"{demo_name}_player_ticks.csv")
        player_ticks.to_csv(player_output, index=False)
        print(f"  ✓ 玩家操作: {player_output} ({len(player_ticks)} 行)")
        
        # ====== 文件 2: 游戏事件时间 ======
        
        # round_start 事件
        round_starts = parser.parse_event("round_start", other=["tick", "game_time"])
        
        # round_end 事件
        round_ends = parser.parse_event("round_end", other=["tick", "game_time"])
        
        # 提取冻结期结束的 tick
        freeze_ends_list = extract_freeze_end_ticks(parser, round_starts)
        freeze_ends_df = pd.DataFrame(freeze_ends_list) if freeze_ends_list else pd.DataFrame(columns=['round', 'freeze_end_tick'])
        
        # 组合所有数据
        game_events_df = pd.DataFrame({
            'round': round_starts['round'].values,
            'round_start_tick': round_starts['tick'].values,
        })
        
        # 添加 round_end_tick
        round_end_dict = dict(zip(round_ends['round'], round_ends['tick']))
        game_events_df['round_end_tick'] = game_events_df['round'].map(round_end_dict)
        
        # 添加 freeze_end_tick
        if not freeze_ends_df.empty:
            freeze_dict = dict(zip(freeze_ends_df['round'], freeze_ends_df['freeze_end_tick']))
            game_events_df['freeze_end_tick'] = game_events_df['round'].map(freeze_dict)
        
        # 添加 official_end_tick (整个比赛的最后 tick)
        if not round_ends.empty:
            official_end = int(round_ends['tick'].max())
        else:
            official_end = game_events_df['round_end_tick'].max()
        
        game_events_df['official_end_tick'] = official_end
        
        # 重新排列列顺序
        cols = ['round', 'round_start_tick']
        if 'freeze_end_tick' in game_events_df.columns:
            cols.append('freeze_end_tick')
        cols.extend(['round_end_tick', 'official_end_tick'])
        game_events_df = game_events_df[cols]
        
        # 保存游戏事件数据
        events_output = os.path.join(output_dir, f"{demo_name}_game_events.csv")
        game_events_df.to_csv(events_output, index=False)
        print(f"  ✓ 游戏事件: {events_output} ({len(game_events_df)} 行)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False

def main():
    # 查找所有 dem 文件
    demo_files = glob.glob("./data/demo_file/*.dem")
    
    if not demo_files:
        print("未找到 .dem 文件，请检查 ./data/demo_file 目录")
        return
    
    print(f"找到 {len(demo_files)} 个 demo 文件")
    print("-" * 60)
    
    success_count = 0
    for demo_path in sorted(demo_files):
        if process_demo(demo_path):
            success_count += 1
        print()
    
    print("-" * 60)
    print(f"处理完成: {success_count}/{len(demo_files)} 个文件成功")
    print(f"输出目录: ./data/output")

if __name__ == "__main__":
    main()