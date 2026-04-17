import os
import json
import random
import argparse
from pathlib import Path

'''
当前实现没有处理空行的情况，需要注意
'''

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def freeze_dataset(input_dir, output_dir, target_player, data_type, total_folds=10):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 固定种子以确保实验可复现性
    random.seed(42)

    # 1. 搜集所有玩家的 manifest
    player_manifests = {}
    for p_dir in input_root.iterdir():
        manifest_path = p_dir / "manifest.json"
        if p_dir.is_dir() and manifest_path.exists():
            all_samples = load_json(manifest_path)
            # 过滤指定类型的样本
            typed_samples = [s for s in all_samples if s['type'] == data_type]
            if typed_samples:
                player_manifests[p_dir.name] = typed_samples

    if target_player not in player_manifests:
        print(f"Error: Target player '{target_player}' has no data for type '{data_type}'.")
        return

    # 2. 确定该玩家可用的地图列表
    pos_samples = player_manifests[target_player]
    available_maps = sorted(list(set([s['map'] for s in pos_samples])), key=lambda x: int(x[3:]))

    # 如果要求的 fold 大于现有 map 数，自动调整或报错
    num_available_maps = len(available_maps)
    if total_folds > num_available_maps:
        print(
            f"Warning: requested {total_folds} folds but only {num_available_maps} maps available. Reducing to {num_available_maps}.")
        total_folds = num_available_maps

    experiment_data = {}

    # 3. LOMO (Leave-One-Map-Out) 循环生成 Folds
    for f_idx in range(total_folds):
        fold_key = f"fold_{f_idx + 1}"

        # 定义地图划分逻辑
        # Test: 当前索引对应的地图
        # Valid: 前一个索引对应的地图 (循环偏移)
        test_map = available_maps[f_idx]
        valid_map = available_maps[(f_idx + 1) % num_available_maps]
        train_maps = [m for m in available_maps if m != test_map and m != valid_map]

        # --- 正样本 (POS) 划分 ---
        pos_train = [s['path'] for s in pos_samples if s['map'] in train_maps]
        pos_valid = [s['path'] for s in pos_samples if s['map'] == valid_map]
        pos_test = [s['path'] for s in pos_samples if s['map'] == test_map]

        # --- 负样本 (NEG) 划分与采样 ---
        neg_players = [p for p in player_manifests.keys() if p != target_player]
        neg_train_all = []
        neg_valid_all = []
        neg_test_all = []

        # 负样本训练集采样逻辑：POS 训练集总数 / 负样本玩家数
        num_pos_train = len(pos_train)
        if len(neg_players) > 0 and num_pos_train > 0:
            num_per_neg_player = max(1, num_pos_train // len(neg_players))
        else:
            num_per_neg_player = 0

        for p_name in neg_players:
            p_samples = player_manifests[p_name]

            # NEG Train: 从训练地图池中均分采样
            # todo: 修改train_map逻辑，当前正负样本集涉及到的map相同，需要修改为map不同情况时也能处理
            p_train_pool = [s['path'] for s in p_samples if s['map'] in train_maps]
            if p_train_pool:
                count = min(len(p_train_pool), num_per_neg_player)
                neg_train_all.extend(random.sample(p_train_pool, count))

            # NEG Valid/Test: 从对应地图全量提取
            neg_valid_all.extend([s['path'] for s in p_samples if s['map'] == valid_map])
            neg_test_all.extend([s['path'] for s in p_samples if s['map'] == test_map])

        if len(neg_train_all) < num_pos_train:
            pos_train = pos_train[:len(neg_train_all)]
            print(f"balance neg & pos samples. delete {num_pos_train} - {len(neg_train_all)} samples")

        # 封装到 Fold
        experiment_data[fold_key] = {
            "test_map": test_map,
            "valid_map": valid_map,
            "train": {"pos": pos_train, "neg": neg_train_all},
            "valid": {"pos": pos_valid, "neg": neg_valid_all},
            "test": {"pos": pos_test, "neg": neg_test_all}
        }

        print(f"Processed {fold_key}: Test={test_map}, Train POS={len(pos_train)} / NEG={len(neg_train_all)}")

    # 4. 保存 JSON
    output_filename = f"{target_player}_{data_type}_folds.json"
    output_path = output_root / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=4, ensure_ascii=False)

    print(f"\n[Success] Final manifest saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze LOMO dataset for a specific player and type.")
    parser.add_argument("--input_dir", default=os.path.join("d:\\", "Project", "Research", "pt_data"), help="Directory containing player manifest.json files.")
    parser.add_argument("--output_dir", default=os.path.join("d:\\", "Project", "Research", "output"), help="Directory to save the final experiment JSON.")
    parser.add_argument("--player", default="apEX", help="The target player name (POS class).")
    parser.add_argument("--type", choices=["mouse", "keyboard", "combined"], default="combined",
                        help="Data type to process.")
    parser.add_argument("--folds", type=int, default=10, help="Number of LOMO folds (limited by available maps).")

    args = parser.parse_args()
    freeze_dataset(args.input_dir, args.output_dir, args.player, args.type, args.folds)
