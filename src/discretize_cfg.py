import json
import numpy as np
import pandas as pd
import os
import yaml

DISCRETIZED_LEVELS = {
    "dp": 5,
    "dy": 5,
    "speed": 5,
    "acc": 5,
    "jerk": 5
}

with open("../setting.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MOUSE_DIR = cfg["path"]["mouse"]
TIME_DIR = cfg["path"]["time"]
OUTPUT_DIR = cfg["path"]["output"]

def main():
    all_data = []
    from function.util_func import process_map_data
    for i in range(1, 11):
        df_map = process_map_data(i, MOUSE_DIR, TIME_DIR)
        if not df_map.empty:
            all_data.append(df_map)

    if not all_data:
        print("No data collected.")
        return

    # 购买时间结束--回合结束时间段内的数据(abs)，全部玩家（10），全部地图（10）
    df_total = pd.concat(all_data, axis=0)
    df_total = df_total.dropna(subset=DISCRETIZED_LEVELS.keys())

    # 是否随机取样
    # df_total = df_total.sample(frac=1, random_state=42)

    results = {}
    # quantile切分
    # for type, level in DISCRETIZED_LEVELS.items():
    #     q_steps = np.linspace(0, 1, level)[1:-1]
    #
    #     # 0作为level 0分布
    #     non_zero = df_total[df_total[type] != 0]
    #
    #     quantiles = np.quantile(non_zero[type], q_steps)
    #     results[type] = sorted(quantiles)

    # log后均分
    for type, level in DISCRETIZED_LEVELS.items():
        df_total[type] = np.log1p(df_total[type])
        bins = np.linspace(df_total[type].min(), df_total[type].max(), level)[1: -1]
        results[type] = sorted(bins)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_json = os.path.join(OUTPUT_DIR, "discretized_level_log.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"JSON results saved to {output_json}")
    return


if __name__ == '__main__':
    main()