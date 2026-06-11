# Demo 文件批量解析工具

## 功能说明

该脚本用于批量处理 Counter-Strike 2 demo 文件，提取玩家操作信息和游戏事件时间点。

对于每个 demo 文件，会生成两个 CSV 文件：

### 1. 玩家操作数据 (`{demo_name}_player_ticks.csv`)
包含每个 tick 时刻所有玩家的操作和视角信息：
- **FORWARD**: 玩家是否按下前进键（布尔值）
- **BACK**: 玩家是否按下后退键（布尔值）
- **LEFT**: 玩家是否按下左键（布尔值）
- **RIGHT**: 玩家是否按下右键（布尔值）
- **FIRE**: 玩家是否按下开火键（布尔值）
- **pitch**: 玩家的俯仰角（浮点数，度数）
- **yaw**: 玩家的偏航角（浮点数，度数）
- **tick**: 当前 tick 编号（整数）

示例行：
```
FORWARD,BACK,LEFT,RIGHT,FIRE,pitch,yaw,tick
False,False,False,False,False,0.0,47.999954,0
False,False,False,False,False,9.202423,-6.455841,0
True,False,False,False,False,-1.234,120.567,100
```

### 2. 游戏事件数据 (`{demo_name}_game_events.csv`)
包含每个回合的关键时间点（以 tick 形式表达）：
- **round**: 回合编号（整数）
- **round_start_tick**: 回合开始时的 tick
- **round_end_tick**: 回合结束时的 tick
- **official_end_tick**: 整个比赛的最后 tick（所有回合的最大 tick）

示例行：
```
round,round_start_tick,round_end_tick,official_end_tick
1,0,9008,230788
2,9328,17968,230788
3,18288,26946,230788
```

## 使用方法

### 前置条件
- Python 3.8+
- 已安装 `demoparser2` 和 `pandas` 包
- demo 文件位于 `./data/demo_file/` 目录

### 运行脚本

```bash
cd /home/nitanglei
python code/research/function/parser.py
```

### 输出

所有生成的 CSV 文件将保存到 `./data/output/` 目录。

运行示例：
```
找到 11 个 demo 文件
------------------------------------------------------------
处理: m1
  ✓ 玩家操作: ./data/output/m1_player_ticks.csv (2445650 行)
  ✓ 游戏事件: ./data/output/m1_game_events.csv (25 行)

处理: m2
  ✓ 玩家操作: ./data/output/m2_player_ticks.csv (2035080 行)
  ✓ 游戏事件: ./data/output/m2_game_events.csv (24 行)

...

处理完成: 11/11 个文件成功
输出目录: ./data/output
```

## 数据说明

### Tick 单位
- 1 tick = 1/64 秒（CS2 默认的时间步长为 64 ticks/秒）
- 例如：64 ticks = 1 秒，128 ticks = 2 秒

### 视角角度（pitch/yaw）
- **pitch**: 俯仰角，范围 -90 ~ 90 度，-90 表示向下看，90 表示向上看
- **yaw**: 偏航角，范围 -180 ~ 180 度，0 表示向北，90 表示向西，-90 表示向东

### 按钮按下状态
- **True**: 按键被按下
- **False**: 按键未被按下

## 文件处理流程

1. 遍历 `./data/demo_file/` 目录中的所有 `.dem` 文件
2. 对每个文件：
   - 使用 `demoparser2` 的 `parse_ticks()` 提取玩家每个 tick 的操作和视角
   - 使用 `parse_event()` 获取 `round_start` 和 `round_end` 事件
   - 合并数据并保存为 CSV
3. 所有成功的文件都会在输出目录生成对应的文件

## 常见问题

### Q: 为什么某些 CSV 行数很多？
A: 这是正常的。每个 tick 每个活跃玩家都会有一行数据。一个完整的比赛可能包含数百万行（多个回合 × 多个玩家 × 高 tick 频率）。

### Q: 如何处理这么大的 CSV 文件？
A: 可以使用 pandas 分块读取：
```python
import pandas as pd
for chunk in pd.read_csv('m1_player_ticks.csv', chunksize=100000):
    # 处理每个块
    process(chunk)
```

### Q: 如何根据玩家 ID 过滤数据？
A: 玩家信息（steamid、name）在更完整的解析中可用，可以修改脚本添加这些字段：
```python
wanted_props = ["FORWARD", "BACK", "LEFT", "RIGHT", "FIRE", "pitch", "yaw", "tick", "steamid", "player_name"]
```

## 脚本位置

- 主脚本: `/home/nitanglei/code/research/function/parser.py`
- 输入目录: `/home/nitanglei/data/demo_file/`
- 输出目录: `/home/nitanglei/data/output/`
