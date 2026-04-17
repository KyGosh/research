import os
import argparse
import pandas as pd
import json
import random

def generate_manifest(meta_path, target_name, out_path, empty_ratio_threshold=0.8, seed=42):
    random.seed(seed)
    
    # Load metadata
    df = pd.read_csv(meta_path)
    
    # Clean data based on empty ratio
    initial_len = len(df)
    df = df[df['empty_ratio'] <= empty_ratio_threshold]
    print(f"Filtered out {initial_len - len(df)} segments with empty_ratio > {empty_ratio_threshold}.")
    
    # Identify maps
    all_maps = sorted(df['map'].unique())
    kfold = len(all_maps)
    
    if kfold == 0:
        print("No maps found in metadata!")
        return
        
    print(f"Found {kfold} maps: {all_maps}")
    
    all_players = sorted(df['player'].unique())
    if target_name not in all_players:
        print(f"Target player {target_name} not found in data!")
        return
        
    imp_players = [p for p in all_players if p != target_name]
    
    manifest = {
        "target_name": target_name,
        "empty_ratio_threshold": empty_ratio_threshold,
        "seed": seed,
        "folds": []
    }
    
    for k in range(kfold):
        test_map = all_maps[k]
        valid_map = all_maps[(k + 1) % kfold]
        train_maps = [m for m in all_maps if m not in [test_map, valid_map]]
        
        fold_data = {
            "fold_idx": k,
            "test_map": test_map,
            "valid_map": valid_map,
            "train_maps": train_maps,
            "train": {"pos": [], "neg": []},
            "val": {"pos": [], "neg": []},
            "test": {"pos": [], "neg": []}
        }
        
        # Helper to get files
        def get_files(maps, player):
            return df[(df['map'].isin(maps)) & (df['player'] == player)]['filepath'].tolist()
            
        # -- TRAIN --
        pos_train = get_files(train_maps, target_name)
        
        neg_train = []
        # We want to balance neg_train to match pos_train.
        # We sample equally from each impostor.
        num_pos_train = len(pos_train)
        if num_pos_train > 0 and len(imp_players) > 0:
            per_imp = max(1, num_pos_train // len(imp_players))
            neg_pool_files = {imp: get_files(train_maps, imp) for imp in imp_players}
            
            rest_pool = []
            for imp in imp_players:
                c = neg_pool_files[imp].copy()
                random.shuffle(c)
                neg_train.extend(c[:per_imp])
                rest_pool.extend(c[per_imp:])
                
            # If we need more to match exactly
            if len(neg_train) < num_pos_train:
                random.shuffle(rest_pool)
                need = num_pos_train - len(neg_train)
                neg_train.extend(rest_pool[:need])
                
            # If we have too many
            neg_train = neg_train[:num_pos_train]
            
        random.shuffle(pos_train)
        random.shuffle(neg_train)
        
        fold_data["train"]["pos"] = pos_train
        fold_data["train"]["neg"] = neg_train
        
        # -- VAL --
        # For val and test, we usually use all available data to get true AUC.
        pos_val = get_files([valid_map], target_name)
        neg_val = []
        for imp in imp_players:
            neg_val.extend(get_files([valid_map], imp))
            
        fold_data["val"]["pos"] = pos_val
        fold_data["val"]["neg"] = neg_val
        
        # -- TEST --
        pos_test = get_files([test_map], target_name)
        neg_test = []
        for imp in imp_players:
            neg_test.extend(get_files([test_map], imp))
            
        fold_data["test"]["pos"] = pos_test
        fold_data["test"]["neg"] = neg_test
        
        manifest["folds"].append(fold_data)
        
        print(f"Fold {k} ({test_map}): Train Pos={len(pos_train)}, Train Neg={len(neg_train)}")
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=str, default=os.path.join("d:\\", "Project", "Research", "serialized_features", "feature_metadata.csv"))
    parser.add_argument("--target-name", type=str, default="apEX")
    parser.add_argument("--out-path", type=str, default=os.path.join("d:\\", "Project", "Research", "experiments", "apEX_auth_folds.json"))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_manifest(args.meta_path, args.target_name, args.out_path, args.threshold, args.seed)

if __name__ == "__main__":
    main()
