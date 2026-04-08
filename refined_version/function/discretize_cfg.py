# 输出序列化的配置文件，确认不同等级的threshold值
import argparse
import json

import numpy as np
import pandas as pd
import os

DISCRETIZED_LEVELS = {
    "dp": 5,
    "dy": 5,
    "speed": 5,
    "acc": 5,
    "jerk": 5
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_dir", type=str, default=os.path.join("d:\\", "Project", "Research", "origin_data", "mouse_data"))
    parser.add_argument("--time_dir", type=str, default=os.path.join("d:\\", "Project", "Research", "origin_data", "time_data"))
    parser.add_argument("--output_dir", type=str, default="d:\\Project\\Research\\output")
    args = parser.parse_args()

    all_data = []
    from util_func import process_map_data
    for i in range(1, 11):
        df_map = process_map_data(i, args.mouse_dir, args.time_dir)
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
    for type, level in DISCRETIZED_LEVELS.items():
        q_steps = np.linspace(0, 1, level)[1:-1]
        quantiles = np.quantile(df_total[type], q_steps)
        unique_quantiles = sorted(set(quantiles))
        results[type] = unique_quantiles

    output_json = os.path.join(args.output_dir, "discretized_level.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"JSON results saved to {output_json}")
    return


if __name__ == '__main__':
    main()