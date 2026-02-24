import os
import argparse
import re
from typing import Dict, List, Tuple, Iterable, Set

import pandas as pd


def find_files(origin_dir: str) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    ks_dir = os.path.join(origin_dir, "keystroke_button")
    ms_dir = os.path.join(origin_dir, "mouse_data")
    tm_dir = os.path.join(origin_dir, "time_data")
    for root, _, files in os.walk(ks_dir):
        for f in files:
            m = re.search(r"_m(\d+)\.csv$", f)
            if not m:
                continue
            k = f"m{m.group(1)}"
            result.setdefault(k, {})
            result[k]["keyboard"] = os.path.join(root, f)
    for root, _, files in os.walk(ms_dir):
        for f in files:
            m = re.search(r"_m(\d+)\.csv$", f)
            if not m:
                continue
            k = f"m{m.group(1)}"
            result.setdefault(k, {})
            result[k]["mouse"] = os.path.join(root, f)
    for root, _, files in os.walk(tm_dir):
        for f in files:
            m = re.search(r"_m(\d+)\.csv$", f)
            if not m:
                continue
            k = f"m{m.group(1)}"
            result.setdefault(k, {})
            result[k]["time"] = os.path.join(root, f)
    return result


def load_time_segments(time_csv: str, segment_ticks: int) -> Dict[int, List[Tuple[int, int]]]:
    df = pd.read_csv(time_csv)
    segments: Dict[int, List[Tuple[int, int]]] = {}
    for _, row in df.iterrows():
        fe = int(row["freeze_end"])
        end = int(row["end"])
        r = int(row["round_num"])
        segs: List[Tuple[int, int]] = []
        start = fe
        while start + segment_ticks - 1 <= end:
            segs.append((start, start + segment_ticks - 1))
            start += segment_ticks
        if segs:
            segments[r] = segs
    return segments


def collect_names(files: Iterable[str]) -> Set[str]:
    names: Set[str] = set()
    for path in files:
        if not os.path.exists(path):
            continue
        try:
            for chunk in pd.read_csv(path, usecols=["name"], chunksize=200000):
                names.update(chunk["name"].astype(str).unique().tolist())
        except ValueError:
            df = pd.read_csv(path)
            if "name" in df.columns:
                names.update(df["name"].astype(str).unique().tolist())
    return names


def ensure_dirs(out_root: str, names: Iterable[str]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    for n in names:
        base = os.path.join(out_root, n)
        kb = os.path.join(base, "keyboard")
        ms = os.path.join(base, "mouse")
        os.makedirs(kb, exist_ok=True)
        os.makedirs(ms, exist_ok=True)
        mapping[n] = {"keyboard": kb, "mouse": ms}
    return mapping


def filter_and_write_all_rounds(
    data_csv: str,
    name: str,
    segments_by_round: Dict[int, List[Tuple[int, int]]],
    out_dir: str,
    tag: str,
    map_key: str,
    start_index: int,
    dtypes: Dict[str, str] = None,
    encoding: str = "utf-8",
) -> int:
    idx = start_index
    usecols = None
    if dtypes is None:
        dtypes = {}
    try:
        header_df = pd.read_csv(data_csv, nrows=0)
        cols = header_df.columns.tolist()
        if "tick" in cols and "name" in cols:
            usecols = cols
    except Exception:
        pass
    for chunk in pd.read_csv(data_csv, chunksize=500000, usecols=usecols, dtype=dtypes, encoding=encoding):
        sub = chunk[chunk["name"] == name]
        if sub.empty:
            continue
        for r in sorted(segments_by_round.keys()):
            segs = segments_by_round[r]
            for s, e in segs:
                mask = (sub["tick"] >= s) & (sub["tick"] <= e)
                part = sub.loc[mask]
                if part.empty:
                    continue
                fname = f"{map_key}_r{r}_seg{idx:03d}_{tag}.csv"
                fpath = os.path.join(out_dir, fname)
                part.to_csv(fpath, index=False)
                idx += 1
    return idx


def process(
    origin_dir: str,
    out_dir: str,
    segment_ticks: int,
    expected_names: int = None,
) -> None:
    files = find_files(origin_dir)
    names = collect_names(
        [v for m in files.values() for v in [m.get("keyboard"), m.get("mouse")] if v]
    )
    if expected_names is not None and len(names) != expected_names:
        pass
    name_dirs = ensure_dirs(out_dir, names)
    counters_kb: Dict[str, int] = {n: 1 for n in names}
    counters_ms: Dict[str, int] = {n: 1 for n in names}
    for map_key in sorted(files.keys()):
        tfile = files[map_key].get("time")
        kfile = files[map_key].get("keyboard")
        mfile = files[map_key].get("mouse")
        if not tfile or not kfile or not mfile:
            continue
        segments_by_round = load_time_segments(tfile, segment_ticks)
        for n in names:
            counters_kb[n] = filter_and_write_all_rounds(
                kfile,
                n,
                segments_by_round,
                name_dirs[n]["keyboard"],
                "kb",
                map_key,
                counters_kb[n],
                dtypes={"tick": "int64", "steamid": "string", "name": "string"},
            )
            counters_ms[n] = filter_and_write_all_rounds(
                mfile,
                n,
                segments_by_round,
                name_dirs[n]["mouse"],
                "ms",
                map_key,
                counters_ms[n],
                dtypes={"tick": "int64", "steamid": "string", "name": "string"},
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin-dir",
        type=str,
        default=os.path.join("d:\\", "Project", "Research", "origin_data"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("d:\\", "Project", "Research", "processed_data"),
    )
    parser.add_argument("--segment-ticks", type=int, default=1920)
    parser.add_argument("--expected-names", type=int, default=None)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    process(args.origin_dir, args.out_dir, args.segment_ticks, args.expected_names)


if __name__ == "__main__":
    main()
