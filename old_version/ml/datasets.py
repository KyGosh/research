import os
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from joblib.parallel import default_parallel_config


def list_name_dirs(processed_root: str) -> Dict[str, Dict[str, str]]:
    res: Dict[str, Dict[str, str]] = {}
    for n in os.listdir(processed_root):
        base = os.path.join(processed_root, n)
        if not os.path.isdir(base):
            continue
        kb = os.path.join(base, "keyboard")
        ms = os.path.join(base, "mouse")
        if os.path.isdir(kb) or os.path.isdir(ms):
            res[n] = {"keyboard": kb, "mouse": ms}
    return res


def list_maps(processed_root: str) -> List[str]:
    maps = []
    for d in os.listdir(processed_root):
        p = os.path.join(processed_root, d)
        if not os.path.isdir(p):
            continue
        if re.match(r"^map\d+$", d):
            maps.append(d)
    maps.sort(key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0)
    return maps


def list_name_dirs_in_map(processed_root: str, map_dir: str) -> Dict[str, Dict[str, str]]:
    res: Dict[str, Dict[str, str]] = {}
    base_map = os.path.join(processed_root, map_dir)
    if not os.path.isdir(base_map):
        return res
    for n in os.listdir(base_map):
        base = os.path.join(base_map, n)
        if not os.path.isdir(base):
            continue
        kb = os.path.join(base, "keyboard")
        ms = os.path.join(base, "mouse")
        if os.path.isdir(kb) or os.path.isdir(ms):
            res[n] = {"keyboard": kb, "mouse": ms}
    return res


def parse_segment_name(fname: str) -> Optional[Tuple[str, int, int, str]]:
    m = re.match(r"(map\d+)_r(\d+)_seg(\d+)_(kb|ms)\.csv$", fname)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)

def build_segments_index(name_dirs: Dict[str, Dict[str, str]]) -> Dict[str, Dict[Tuple[str, int, int], Dict[str, str]]]:
    idx: Dict[str, Dict[Tuple[str, int, int], Dict[str, str]]] = {}
    for n, d in name_dirs.items():
        idx[n] = {}
        for tag in ["keyboard", "mouse"]:
            p = d.get(tag)
            if not p or not os.path.isdir(p):
                continue
            for f in os.listdir(p):
                meta = parse_segment_name(f)
                if not meta:
                    continue
                mk, r, s, mtag = meta
                key = (mk, r, s)
                idx[n].setdefault(key, {})
                idx[n][key][mtag] = os.path.join(p, f)
    return idx


def build_pairs_in_map(processed_root: str, map_dir: str, name: str) -> List[Tuple[str, str, Tuple[str, int, int]]]:
    nd = list_name_dirs_in_map(processed_root, map_dir)
    if name not in nd:
        return []
    kb_dir = nd[name].get("keyboard")
    ms_dir = nd[name].get("mouse")
    keys: Dict[Tuple[int, int], Dict[str, str]] = {}
    if kb_dir and os.path.isdir(kb_dir):
        for f in os.listdir(kb_dir):
            meta = parse_segment_name(f)
            if not meta:
                continue
            _, r, s, tag = meta
            if tag != "kb":
                continue
            keys.setdefault((r, s), {})
            keys[(r, s)]["kb"] = os.path.join(kb_dir, f)
    if ms_dir and os.path.isdir(ms_dir):
        for f in os.listdir(ms_dir):
            meta = parse_segment_name(f)
            if not meta:
                continue
            _, r, s, tag = meta
            if tag != "ms":
                continue
            keys.setdefault((r, s), {})
            keys[(r, s)]["ms"] = os.path.join(ms_dir, f)
    res: List[Tuple[str, str, Tuple[str, int, int]]] = []
    for (r, s), vv in keys.items():
        kb = vv.get("kb")
        ms = vv.get("ms")
        if kb and ms:
            res.append((kb, ms, (map_dir, r, s)))
    return res


def list_pairs_for_name_in_maps(processed_root: str, name: str, maps: List[str]) -> List[Tuple[str, str, Tuple[str, int, int]]]:
    res: List[Tuple[str, str, Tuple[str, int, int]]] = []
    for mdir in maps:
        res.extend(build_pairs_in_map(processed_root, mdir, name))
    res.sort(key=lambda x: (x[2][0], x[2][1], x[2][2]))
    return res


def read_keyboard_csv(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = ["BACK", "RIGHT", "FORWARD", "LEFT"]
    if not set(cols).issubset(df.columns):
        return None
    arr = df[cols].astype(str).values
    res = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[1]):
        c = arr[:, i].astype(str)
        c = np.where(c == "True", "1.0", c)
        c = np.where(c == "False", "0.0", c)
        v = c.astype(np.float32)
        res[:, i] = v
    return res

def read_mouse_csv(path: str, standardize: bool = False) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = ["FIRE", "pitch", "yaw"]
    if not set(cols).issubset(df.columns):
        return None

    # 处理数据，使用pandas/dataframe
    df["dp"] = df["pitch"].diff()
    df["dy"] = df["yaw"].diff()
    df["speed"] = np.sqrt(df["dp"] ** 2 + df["dy"] ** 2)
    df["acc"] = df["speed"].diff()
    df["jerk"] = df["acc"].diff()

    # 处理df中的NaN数据
    df = df.bfill().fillna(0)

    # 特征，转成numpy
    fire = df["FIRE"].map({True: 1.0, False: 0.0}).astype(np.float32).values
    pitch = df["pitch"].astype(np.float32).values
    yaw = df["yaw"].astype(np.float32).values
    dp = df["dp"].astype(np.float32).values
    dy = df["dy"].astype(np.float32).values
    speed = df["speed"].astype(np.float32).values
    acc = df["acc"].astype(np.float32).values
    jerk = df["jerk"].astype(np.float32).values

    # 标准化
    if standardize:
        for v in (pitch, yaw):
            m = float(np.mean(v))
            s = float(np.std(v)) if float(np.std(v)) > 1e-8 else 1.0
            v -= m
            v /= s

    # res = np.stack([fire, pitch, yaw, dp, dy, speed, acc, jerk], axis=1).astype(np.float32)
    res = np.stack([fire, dp, dy, speed, acc, jerk], axis=1).astype(np.float32)

    if np.isnan(res).any():
        print("np has NaN value!")
        return None

    return res

def read_unified_sample(kb_path: str, ms_path: str) -> Optional[np.ndarray]:
    kb = read_keyboard_csv(kb_path)
    ms = read_mouse_csv(ms_path)
    if kb is None or ms is None:
        return None
    return np.concatenate([kb, ms], axis=1).astype(np.float32)


def list_pairs_for_name(processed_root: str, name: str) -> List[Tuple[str, str, Tuple[str, int, int]]]:
    ndirs = list_name_dirs(processed_root)
    if name not in ndirs:
        return []
    idx = build_segments_index({name: ndirs[name]})
    pairs: List[Tuple[str, str, Tuple[str, int, int]]] = []
    for key, files in idx[name].items():
        kb = files.get("kb")
        ms = files.get("ms")
        if kb and ms:
            pairs.append((kb, ms, key))
    return sorted(pairs, key=lambda x: (x[2][0], x[2][1], x[2][2]))


def list_single_for_name(processed_root: str, name: str, mode: str) -> List[str]:
    ndirs = list_name_dirs(processed_root)
    if name not in ndirs:
        return []
    p = ndirs[name].get("keyboard" if mode == "keyboard" else "mouse")
    if not p or not os.path.isdir(p):
        return []
    files = []
    for f in os.listdir(p):
        meta = parse_segment_name(f)
        if not meta:
            continue
        files.append(os.path.join(p, f))
    files.sort()
    return files


def list_single_for_name_in_maps(processed_root: str, name: str, mode: str, maps: List[str]) -> List[str]:
    res: List[str] = []
    for m in maps:
        nd = list_name_dirs_in_map(processed_root, m)
        if name not in nd:
            continue
        p = nd[name].get("keyboard" if mode == "keyboard" else "mouse")
        if not p or not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            meta = parse_segment_name(f)
            if not meta:
                continue
            res.append(os.path.join(p, f))
    res.sort()
    return res


def read_sample(path: str, mode: str) -> Optional[np.ndarray]:
    if mode == "keyboard":
        return read_keyboard_csv(path)
    if mode == "mouse":
        return read_mouse_csv(path)
    return None
