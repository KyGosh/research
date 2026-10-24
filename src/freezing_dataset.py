import json
import random
import re
import yaml
from pathlib import Path

def extract_sort_key(sample, player_name=""):
    path_str = sample.get("path", "")
    match_id = sample.get("match_id", sample.get("map", ""))
    filename = Path(path_str).stem
    m = re.search(r"r(\d+)_seg(\d+)", filename)
    if m:
        return (player_name, str(match_id), int(m.group(1)), int(m.group(2)), filename)
    return (player_name, str(match_id), 0, 0, filename)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def freeze_dataset(input_dir, output_dir, target_player, data_type, total_folds=10):
    """
    Original Leave-One-Map-Out (LOMO) binary splitting logic for authentication.
    """
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    # 1. Gather all manifests
    player_manifests = {}
    for p_dir in sorted(list(input_root.iterdir())):
        manifest_path = p_dir / "manifest.json"
        if p_dir.is_dir() and manifest_path.exists():
            all_samples = load_json(manifest_path)
            # Filter type (mouse/keyboard/combined)
            typed_samples = [s for s in all_samples if s['type'] == data_type]
            if typed_samples:
                typed_samples.sort(key=lambda s: extract_sort_key(s, p_dir.name))
                player_manifests[p_dir.name] = typed_samples

    if target_player not in player_manifests:
        print(f"Error: Target player '{target_player}' has no data for type '{data_type}'.")
        return

    # 2. Determine maps
    pos_samples = player_manifests[target_player]
    
    # Check if we have map/match_id in manifest
    # (Backward compatibility with older 'map' key in manifest)
    map_key_name = 'match_id' if 'match_id' in pos_samples[0] else 'map'
    available_maps = sorted(list(set([s[map_key_name] for s in pos_samples])))

    num_available_maps = len(available_maps)
    if total_folds > num_available_maps:
        print(f"Warning: requested {total_folds} folds but only {num_available_maps} sessions available. Reducing folds.")
        total_folds = num_available_maps

    experiment_data = {}

    # 3. LOMO Loop
    for f_idx in range(total_folds):
        fold_key = f"fold_{f_idx + 1}"

        test_map = available_maps[f_idx]
        valid_map = available_maps[(f_idx + 1) % num_available_maps]
        train_maps = [m for m in available_maps if m != test_map and m != valid_map]

        pos_train = [s['path'] for s in pos_samples if s[map_key_name] in train_maps]
        pos_valid = [s['path'] for s in pos_samples if s[map_key_name] == valid_map]
        pos_test = [s['path'] for s in pos_samples if s[map_key_name] == test_map]

        neg_players = sorted([p for p in player_manifests.keys() if p != target_player])
        neg_train_all = []
        neg_valid_all = []
        neg_test_all = []

        num_pos_train = len(pos_train)
        if len(neg_players) > 0 and num_pos_train > 0:
            num_per_neg_player = max(1, num_pos_train // len(neg_players))
        else:
            num_per_neg_player = 0

        for p_name in neg_players:
            p_samples = player_manifests[p_name]
            p_train_pool = [s['path'] for s in p_samples if s[map_key_name] in train_maps]
            if p_train_pool:
                count = min(len(p_train_pool), num_per_neg_player)
                neg_train_all.extend(random.sample(p_train_pool, count))

            neg_valid_all.extend([s['path'] for s in p_samples if s[map_key_name] == valid_map])
            neg_test_all.extend([s['path'] for s in p_samples if s[map_key_name] == test_map])

        if len(neg_train_all) < num_pos_train:
            pos_train = pos_train[:len(neg_train_all)]

        experiment_data[fold_key] = {
            "test_map": test_map,
            "valid_map": valid_map,
            "train": {"pos": pos_train, "neg": neg_train_all},
            "valid": {"pos": pos_valid, "neg": neg_valid_all},
            "test": {"pos": pos_test, "neg": neg_test_all}
        }
        print(f"Processed {fold_key}: Test={test_map}, Train POS={len(pos_train)} / NEG={len(neg_train_all)}")

    output_filename = f"{target_player}_{data_type}_folds.json"
    output_path = output_root / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=4, ensure_ascii=False)
    print(f"[Success] Manifest saved to: {output_path}")

def freeze_dataset_id(input_dir, output_dir, data_type, total_folds=10):
    """
    Match-Based multi-class split logic for player identification.
    Split ratio:
      - n_matches >= 10: 8 : 1 : x (8 train, 1 valid, remaining n_matches - 9 to test)
      - n_matches < 10:  x : 1 : 1 (n_matches - 2 to train, 1 valid, 1 test)
      - Insufficient matches (< 3): Skipped (retaining original behavior)
    """
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    # 1. Gather all manifests, skipping players with insufficient matches (< 5)
    player_manifests = {}
    for p_dir in sorted(list(input_root.iterdir())):
        manifest_path = p_dir / "manifest.json"
        if p_dir.is_dir() and manifest_path.exists():
            all_samples = load_json(manifest_path)
            typed_samples = [s for s in all_samples if s['type'] == data_type]
            if typed_samples:
                # Check unique match count
                map_key_name = 'match_id' if 'match_id' in typed_samples[0] else 'map'
                match_keys = set([s[map_key_name] for s in typed_samples])
                if len(match_keys) < 3:
                    print(f"[WARNING] Player '{p_dir.name}' only has {len(match_keys)} match(es). Insufficient data to perform cross-validation. Skipping player.")
                    continue
                typed_samples.sort(key=lambda s: extract_sort_key(s, p_dir.name))
                player_manifests[p_dir.name] = typed_samples

    # Dynamically build player list from the actual folders in the dataset
    players_list = sorted(list(player_manifests.keys()))
    player_to_label = {p_name: i for i, p_name in enumerate(players_list)}
    print(f"Dynamically mapped {len(players_list)} players to class labels: {player_to_label}")

    experiment_data = {
        "class_mapping": player_to_label
    }

    # Pre-extract sorted unique match keys for each player
    player_match_keys = {}
    for p_name, samples in player_manifests.items():
        map_key_name = 'match_id' if 'match_id' in samples[0] else 'map'
        match_keys = sorted(list(set([s[map_key_name] for s in samples])))
        player_match_keys[p_name] = match_keys

    # 2. Compile Folds
    for f_idx in range(total_folds):
        fold_key = f"fold_{f_idx + 1}"
        
        train_paths, train_labels = [], []
        valid_paths, valid_labels = [], []
        test_paths, test_labels = [], []

        for p_name in players_list:
            label = player_to_label[p_name]
            samples = player_manifests[p_name]
            map_key_name = 'match_id' if 'match_id' in samples[0] else 'map'
            match_keys = player_match_keys[p_name]
            n_matches = len(match_keys)

            if n_matches >= 10:
                # 8 : 1 : x 策略 (>= 10 场：8 场训练，1 场验证，剩余全部放入测试集)
                valid_idx = (f_idx + 1) % n_matches
                valid_set = {match_keys[valid_idx]}

                train_indices = [(f_idx + 2 + i) % n_matches for i in range(8)]
                train_set = {match_keys[i] for i in train_indices}

                test_set = set(match_keys) - train_set - valid_set
            else:
                # x : 1 : 1 策略 (< 10 场：1 场测试，1 场验证，剩余全部作为训练集)
                test_idx = f_idx % n_matches
                valid_idx = (f_idx + 1) % n_matches
                test_set = {match_keys[test_idx]}
                valid_set = {match_keys[valid_idx]}

                train_set = set(match_keys) - test_set - valid_set

            # Group all samples of this player
            for s in samples:
                m_id = s[map_key_name]
                if m_id in test_set:
                    test_paths.append(s['path'])
                    test_labels.append(label)
                elif m_id in valid_set:
                    valid_paths.append(s['path'])
                    valid_labels.append(label)
                elif m_id in train_set:
                    train_paths.append(s['path'])
                    train_labels.append(label)

        experiment_data[fold_key] = {
            "train": {"paths": train_paths, "labels": train_labels},
            "valid": {"paths": valid_paths, "labels": valid_labels},
            "test": {"paths": test_paths, "labels": test_labels}
        }
        print(f"Processed {fold_key}: Train={len(train_paths)}, Valid={len(valid_paths)}, Test={len(test_paths)}")

    # 4. Save JSON
    output_filename = f"identification_{data_type}_folds.json"
    output_path = output_root / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=4, ensure_ascii=False)

    print(f"\n[Success] Identification fold manifest saved to: {output_path}")

# Load configuration
with open("../setting.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PT_DIR = cfg["path"]["pt"]
OUTPUT_DIR = cfg["path"]["output"]
PLAYERS = cfg["players"]
TYPES = cfg["types"]
FOLDS = cfg["fold"]
TASK = cfg.get("task", "authentication")

if __name__ == "__main__":
    if TASK == "identification":
        print(f"Running DATASET FREEZE for task: {TASK.upper()} (Multi-Class Identification)")
        for type_ in TYPES:
            freeze_dataset_id(PT_DIR, OUTPUT_DIR, type_, FOLDS)
    else:
        print(f"Running DATASET FREEZE for task: {TASK.upper()} (Binary Authentication)")
        for player in PLAYERS:
            for type_ in TYPES:
                freeze_dataset(PT_DIR, OUTPUT_DIR, player, type_, FOLDS)