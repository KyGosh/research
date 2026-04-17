import json
from pathlib import Path

def generate_player_manifests(pt_root="d:\\Project\\Research\\pt_data"):
    """
    扫描 pt_root 目录，为每个玩家生成一个 manifest.json。
    结构：pt_data/{player}/manifest.json
    内容：平铺式的样本列表，包含 path, map, type。
    """
    pt_root = Path(pt_root)
    if not pt_root.exists():
        print(f"Error: {pt_root} does not exist.")
        return

    # 获取所有玩家目录
    players = [d for d in pt_root.iterdir() if d.is_dir()]
    
    for player_dir in players:
        player_name = player_dir.name
        manifest = []
        
        # 递归查找所有 .pt 文件
        # 预期结构: pt_data/{player}/{map}/{type}/*.pt
        for pt_file in player_dir.rglob("*.pt"):
            # 获取相对于玩家目录的路径部分
            try:
                relative_parts = pt_file.relative_to(player_dir).parts
                if len(relative_parts) >= 2:
                    # 假设路径为: map1/mouse/r1_seg1.pt
                    mapping_info = {
                        "path": str(pt_file.as_posix()),
                        "map": relative_parts[0],
                        "type": relative_parts[1]
                    }
                    manifest.append(mapping_info)
            except Exception as e:
                print(f"Skipping {pt_file}: {e}")
        
        # 将该玩家的索引写入该玩家目录
        output_path = player_dir / "manifest.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
        
        print(f"Done: {player_name} -> {len(manifest)} samples recorded in {output_path}")

if __name__ == "__main__":
    generate_player_manifests()
