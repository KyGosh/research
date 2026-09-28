import os
import re
import datetime
import yaml
import numpy as np
import torch
from demoparser2 import DemoParser, WantedPropState
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function.util_func import discretize_mouse_data

# Load configuration
with open("../setting.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DEMO_DIR = cfg["path"].get("demos", "d:\\Project\\cs2_parser\\demoparser\\demo_file")
PT_DIR_ROOT = cfg["path"]["pt"]
WINDOW_SIZE = cfg["time-window"]
OVERLAP_RATIO = cfg["overlap-ratio"]


def parse_single_demo(demo_path):
    filename = os.path.basename(demo_path)
    print(f"\n==================================================")
    print(f"Processing Demo: {filename}")
    print(f"==================================================")

    # 1. Initialize Parser
    parser = DemoParser(demo_path)

    # 2. Extract Metadata from Demo Header & Filename
    header = parser.parse_header()
    map_name = header.get("map_name", "unknown_map")

    # Date from file system modification time
    mtime = os.path.getmtime(demo_path)
    date_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y%m%d")

    # vitality-vs-natus-vincere-m1-inferno.dem
    match = re.search(r"^(?P<team1>[\w-]+)-vs-(?P<team2>[\w-]+)-m(?P<match_num>\d+)", filename)
    if match:
        team1 = match.group("team1").replace("-", "_")
        team2 = match.group("team2").replace("-", "_")
        match_num = f"m{match.group('match_num')}"
        match_key = f"match_{date_str}_{team1}_vs_{team2}_{match_num}_{map_name}"
    else:
        fallback_name = os.path.splitext(filename)[0].replace("-", "_")
        match_key = f"match_{date_str}_{fallback_name}_{map_name}"

    # 3. Parse Telemetry in a single pass
    print("Parsing telemetry ticks...")
    fields = [
        "pitch", "yaw", "FIRE",
        "FORWARD", "BACK", "LEFT", "RIGHT", "is_walking",
        "steamid", "name"
    ]
    try:
        df_telemetry = parser.parse_ticks(fields, prop_states=[WantedPropState("is_alive", True)])
    except Exception as e:
        print(f"[ERROR] Failed parsing ticks: {e}")
        return

    if df_telemetry.empty:
        print("[WARNING] Empty telemetry data. Skipping demo.")
        return

    # 4. Retrieve Round Events in Memory with Robust User-Defined Lifecycle
    print("Parsing round events...")
    tick_rate = header.get("tick_rate", 64.0)
    if tick_rate <= 0:
        tick_rate = 64.0

    df_start = parser.parse_event("round_start")
    df_freeze = parser.parse_event("round_freeze_end")
    df_end = parser.parse_event("round_end")
    df_official = parser.parse_event("round_officially_ended")

    # Extract ticks as sorted lists of integers
    start_ticks = sorted([int(t) for t in df_start["tick"].values]) if not df_start.empty else []
    freeze_ticks = sorted([int(t) for t in df_freeze["tick"].values]) if not df_freeze.empty else []
    end_ticks = sorted([int(t) for t in df_end["tick"].values]) if not df_end.empty else []
    official_ticks = sorted([int(t) for t in df_official["tick"].values]) if not df_official.empty else []

    rounds = []
    current_round_num = 1

    # find all valid rounds in this demo file
    for s in start_ticks:
        # 1. Find the next official_end tick. If missing, discard this round.
        official_candidates = [t for t in official_ticks if t > s]
        if not official_candidates:
            continue
        official = official_candidates[0]

        # 2. Check if another round_start occurs between s and official.
        # If so, this round was restarted/aborted; discard it.
        between_starts = [t for t in start_ticks if s < t < official]
        if between_starts:
            continue

        # 3. Find freeze_end (live) tick within the round [s, official]
        live_candidates = [t for t in freeze_ticks if s < t < official]
        if live_candidates:
            live = live_candidates[0]
        else:
            # Estimate: start + 20s
            live = int(s + 20 * tick_rate)

        # 4. Find round_end tick within the round [live, official]
        end_candidates = [t for t in end_ticks if live < t < official]
        if end_candidates:
            end = end_candidates[0]
        else:
            # Estimate: official_end - 5s
            end = int(official - 5 * tick_rate)

        # 5. Sanity checks: Ensure valid order and duration
        if live < end and (end - live) >= WINDOW_SIZE:
            rounds.append((current_round_num, live, end))
            current_round_num += 1

    print(f"Found {len(rounds)} valid rounds.")

    # 5. Calculate Mouse Dynamics and Discretize
    print("Calculating mouse dynamics and discretizing...")
    df_telemetry = df_telemetry.sort_values(["name", "tick"])
    mouse_gb = df_telemetry.groupby("name")
    df_telemetry["dp"] = mouse_gb["pitch"].diff()
    dy_raw = mouse_gb["yaw"].diff()
    df_telemetry["dy"] = (dy_raw + 180) % 360 - 180
    df_telemetry["speed"] = np.sqrt(df_telemetry["dp"] ** 2 + df_telemetry["dy"] ** 2)
    df_telemetry["acc"] = df_telemetry.groupby("name")["speed"].diff()
    df_telemetry["jerk"] = df_telemetry.groupby("name")["acc"].diff()

    df_telemetry = df_telemetry.bfill().fillna(0)
    df_telemetry = discretize_mouse_data(df_telemetry)

    # 6. Convert Bool/Obj Columns to Numeric
    for col in ["FORWARD", "BACK", "LEFT", "RIGHT", "is_walking", "FIRE"]:
        if col in df_telemetry.columns:
            df_telemetry[col] = df_telemetry[col].astype(float)

    # Define feature selections
    keyboard_cols = ["is_walking", "BACK", "RIGHT", "FORWARD", "LEFT"]
    mouse_cols = ["FIRE", "dp", "dy", "speed", "acc", "jerk"]
    combined_cols = keyboard_cols + mouse_cols

    # 7. Group and slice sliding windows in memory
    print("Slicing windows and saving directly to .pt...")
    player_groups = df_telemetry.groupby("name")
    step = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

    for player_name, df_player in player_groups:
        player_tick_set = set(df_player["tick"].values)
        
        # Avoid static typing warnings for Hashable by using type narrowing
        player_name_str = player_name if isinstance(player_name, str) else "unknown"

        # Setup directories for each type
        dirs = {
            "keyboard": os.path.join(PT_DIR_ROOT, player_name_str, match_key, "keyboard"),
            "mouse": os.path.join(PT_DIR_ROOT, player_name_str, match_key, "mouse"),
            "combined": os.path.join(PT_DIR_ROOT, player_name_str, match_key, "combined")
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        seg_idx = 1
        for round_num, live, end in rounds:
            mask = (df_player["tick"] >= live) & (df_player["tick"] <= end)
            df_round = df_player.loc[mask]
            if df_round.empty:
                continue

            start_w = live
            while start_w + WINDOW_SIZE - 1 < end:
                end_w = start_w + WINDOW_SIZE - 1
                if end_w not in player_tick_set:
                    start_w += step
                    break

                df_win = df_round[(df_round["tick"] >= start_w) & (df_round["tick"] <= end_w)]
                if df_win["tick"].nunique() == WINDOW_SIZE:
                    # Keyboard Tensors
                    kb_data = df_win[keyboard_cols].values.astype(np.float32)
                    torch.save(torch.from_numpy(kb_data), os.path.join(dirs["keyboard"], f"r{round_num}_seg{seg_idx}.pt"))

                    # Mouse Tensors
                    ms_data = df_win[mouse_cols].values.astype(np.float32)
                    torch.save(torch.from_numpy(ms_data), os.path.join(dirs["mouse"], f"r{round_num}_seg{seg_idx}.pt"))

                    # Combined Tensors
                    comb_data = df_win[combined_cols].values.astype(np.float32)
                    torch.save(torch.from_numpy(comb_data), os.path.join(dirs["combined"], f"r{round_num}_seg{seg_idx}.pt"))

                    seg_idx += 1
                start_w += step
        print(f"  Processed {seg_idx - 1} segments for player: {player_name}")

def main():
    if not os.path.exists(DEMO_DIR):
        print(f"[ERROR] Demos directory does not exist: {DEMO_DIR}")
        return

    # Find all .dem files
    demo_files = []
    for root, _, files in os.walk(DEMO_DIR):
        for file in files:
            if file.endswith(".dem"):
                demo_files.append(os.path.join(root, file))

    print(f"Found {len(demo_files)} CS2 demo files in {DEMO_DIR}.")
    for demo in demo_files:
        parse_single_demo(demo)
    print("\n[SUCCESS] Direct-to-Tensor parsing pipeline completed successfully.")

if __name__ == "__main__":
    main()
