import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
功能：
1. 读取 mouse_data(原始文件) 和 time_data (map1-map10)
2. 方案A：先按玩家分组计算 dp, dy, speed, acc, jerk
3. 根据 time_data 的 [freeze_end, end] 过滤数据
4. 汇总所有地图数据，删除 NaN
5. 统计 0 值数量，并生成非 0 值的分布图
'''

DISCRETIZED_TYPE = ["dp", "dy", "speed", "acc", "jerk"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_dir", type=str, default=os.path.join("d:\\", "Project","Research", "origin_data", "mouse_data"))
    parser.add_argument("--time_dir", type=str, default=os.path.join("d:\\", "Project","Research", "origin_data", "time_data"))
    parser.add_argument("--output_img_dir", type=str, default="plots")
    args = parser.parse_args()

    if not os.path.exists(args.output_img_dir):
        os.makedirs(args.output_img_dir)

    all_data = []
    from util_func import process_map_data
    for i in range(1, 11):
        df_map = process_map_data(i, args.mouse_dir, args.time_dir)
        if not df_map.empty:
            all_data.append(df_map)
    
    if not all_data:
        print("No data collected.")
        return

    df_total = pd.concat(all_data, axis=0)
    
    # 3. 删除 NaN (每个玩家每回合的前几行 diff 产生的 NaN)
    df_total = df_total.dropna(subset=DISCRETIZED_TYPE)
    
    print("\n--- Statistics for 0 values ---")
    for col in DISCRETIZED_TYPE:
        zero_count = (df_total[col] == 0).sum()
        total_count = len(df_total)
        print(f"{col}: {zero_count} zeros (out of {total_count} rows, {zero_count/total_count:.2%})")

    print("\n--- Percentile Report (Non-zero values) ---")
    percentile_data = {}
    for col in DISCRETIZED_TYPE:
        non_zero = df_total[df_total[col] != 0][col]
        if non_zero.empty:
            continue
        
        # 计算百分位数
        qs = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
        res = non_zero.quantile(qs)
        percentile_data[col] = res
        print(f"\n[{col}] Percentiles:")
        for q, val in res.items():
            print(f"  {q*100:>2.0f}%: {val:.4f}")

    # 4. 生成辅助离散化的图表
    print("\nGenerating enhanced visualization for discretization...")
    
    for col in DISCRETIZED_TYPE:
        non_zero_data = df_total[df_total[col] != 0][col]
        if non_zero_data.empty:
            continue
            
        # 创建画布：左图直方图+KDE，右图ECDF（累积分布）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 限制绘图范围在 99% 分位数以内，避免被异常值挤压
        upper_limit = non_zero_data.quantile(0.99)
        plot_data = non_zero_data[non_zero_data <= upper_limit]
        
        # 左图：分布密度
        sns.histplot(plot_data, kde=True, ax=ax1)
        ax1.set_title(f"{col} Distribution (Up to 99th Percentile)")
        ax1.set_xlabel("Value")
        
        # 右图：累积分布曲线 (ECDF) - 最利于决定离散化阈值
        sns.ecdfplot(non_zero_data, ax=ax2)
        ax2.set_title(f"{col} Cumulative Distribution (ECDF)")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Percentage of Data Covered")
        ax2.grid(True, linestyle='--', alpha=0.6)
        # 在ECDF上标出常用的分位数线
        for q in [0.25, 0.5, 0.75]:
            val = non_zero_data.quantile(q)
            if val <= non_zero_data.quantile(0.999): # 避免线条画得太远
                ax2.axhline(q, color='r', linestyle=':', alpha=0.5)
                ax2.axvline(val, color='r', linestyle=':', alpha=0.5)
                ax2.text(val, q, f"{q:.2f}\n{val:.2f}", fontsize=8)

        plt.suptitle(f"Discretization Analysis: {col}", fontsize=16)
        plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
        plt.savefig(os.path.join(args.output_img_dir, f"{col}_discretize_analysis.png"))
        plt.close()

    print(f"Enhanced plots saved to '{args.output_img_dir}' folder.")


if __name__ == '__main__':
    main()