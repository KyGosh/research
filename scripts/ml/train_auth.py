import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import TensorDataset, DataLoader
from scripts.ml.datasets import list_maps, list_name_dirs_in_map, list_single_for_name_in_maps, list_pairs_for_name_in_maps, read_sample, read_unified_sample
from scripts.ml.models import UnifiedModel, FusionModel


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_tensors(files, mode):
    xs = []
    for f in files:
        x = read_sample(f, mode)
        if x is None:
            continue
        xs.append(x)
    if not xs:
        return None
    x = np.stack(xs, axis=0).astype(np.float32)
    return torch.tensor(x, dtype=torch.float32)


def build_tensors_unified_from_pairs(pairs):
    xs = []
    for kb, ms, _ in pairs:
        x = read_unified_sample(kb, ms)
        if x is None:
            continue
        xs.append(x)
    if not xs:
        return None
    x = np.stack(xs, axis=0).astype(np.float32)
    return torch.tensor(x, dtype=torch.float32)


def build_tensors_fusion(pairs):
    kb_xs = []
    ms_xs = []
    for kb, ms, _ in pairs:
        k = read_sample(kb, "keyboard")
        m = read_sample(ms, "mouse")
        if k is None or m is None:
            continue
        kb_xs.append(k)
        ms_xs.append(m)
    if not kb_xs:
        return None, None
    kb = np.stack(kb_xs, axis=0).astype(np.float32)
    ms = np.stack(ms_xs, axis=0).astype(np.float32)
    return torch.tensor(kb, dtype=torch.float32), torch.tensor(ms, dtype=torch.float32)


def train_binary_unified(x_train, y_train, x_val, y_val, input_dim, device, epochs=30, lr=1e-3, bsz=32, patience=5):
    model = UnifiedModel(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bsz)
    best_val = float("inf")
    best_state = None
    bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        vs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                vs.append(loss_fn(model(xb), yb).item())
        v = float(np.mean(vs)) if vs else 0.0
        if v < best_val:
            best_val = v
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_binary_fusion(kb_train, ms_train, y_train, kb_val, ms_val, y_val, kb_dim, ms_dim, device, epochs=30, lr=1e-3, bsz=32, patience=5):
    model = FusionModel(kb_input_dim=kb_dim, ms_input_dim=ms_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(kb_train, ms_train, y_train)
    val_ds = TensorDataset(kb_val, ms_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bsz)
    best_val = float("inf")
    best_state = None
    bad = 0
    for _ in range(epochs):
        model.train()
        for kbx, msx, yb in train_loader:
            kbx = kbx.to(device)
            msx = msx.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(kbx, msx)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        vs = []
        with torch.no_grad():
            for kbx, msx, yb in val_loader:
                kbx = kbx.to(device)
                msx = msx.to(device)
                yb = yb.to(device)
                vs.append(loss_fn(model(kbx, msx), yb).item())
        v = float(np.mean(vs)) if vs else 0.0
        if v < best_val:
            best_val = v
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def compute_eer(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fnr = 1 - tpr
        i = int(np.argmin(np.abs(fpr - fnr)))
        return float((fpr[i] + fnr[i]) / 2.0)
    except Exception:
        return 0.0


def evaluate_unified(model, x, y, device, bsz=64):
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=bsz)
    ps = []
    ys = []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            p = torch.sigmoid(model(xb))
            ps.extend(p.cpu().numpy().tolist())
            ys.extend(yb.cpu().numpy().tolist())
    pr = np.array(ps).reshape(-1)
    gt = np.array(ys).reshape(-1)
    pred = (pr > 0.5).astype(np.float32)
    acc = float(np.mean(pred == gt)) if gt.size > 0 else 0.0
    try:
        auc = roc_auc_score(gt, pr) if len(np.unique(gt)) > 1 else 0.0
    except Exception:
        auc = 0.0
    eer = compute_eer(gt, pr)
    return acc, auc, eer


def evaluate_fusion(model, kb_x, ms_x, y, device, bsz=64):
    ds = TensorDataset(kb_x, ms_x, y)
    dl = DataLoader(ds, batch_size=bsz)
    ps = []
    ys = []
    model.eval()
    with torch.no_grad():
        for kbx, msx, yb in dl:
            kbx = kbx.to(device)
            msx = msx.to(device)
            yb = yb.to(device)
            p = torch.sigmoid(model(kbx, msx))
            ps.extend(p.cpu().numpy().tolist())
            ys.extend(yb.cpu().numpy().tolist())
    pr = np.array(ps).reshape(-1)
    gt = np.array(ys).reshape(-1)
    pred = (pr > 0.5).astype(np.float32)
    acc = float(np.mean(pred == gt)) if gt.size > 0 else 0.0
    try:
        auc = roc_auc_score(gt, pr) if len(np.unique(gt)) > 1 else 0.0
    except Exception:
        auc = 0.0
    eer = compute_eer(gt, pr)
    return acc, auc, eer


def get_all_names_in_maps(processed_root: str, maps_list):
    s = set()
    for m in maps_list:
        nd = list_name_dirs_in_map(processed_root, m)
        for n in nd.keys():
            s.add(n)
    return sorted(list(s))


def run_fold(processed_root: str, target_name: str, mode: str, train_maps, valid_map, test_map, names, device, epochs=30, lr=1e-3, bsz=32, seed=42):
    random.seed(seed)
    if mode in ["keyboard", "mouse"]:
        pos_train = list_single_for_name_in_maps(processed_root, target_name, mode, train_maps)
        neg_pool = {imp: list_single_for_name_in_maps(processed_root, imp, mode, train_maps) for imp in names}
        pos_val = list_single_for_name_in_maps(processed_root, target_name, mode, [valid_map])
        neg_val = []
        for imp in names:
            neg_val.extend(list_single_for_name_in_maps(processed_root, imp, mode, [valid_map]))
        pos_test = list_single_for_name_in_maps(processed_root, target_name, mode, [test_map])
        neg_test = []
        for imp in names:
            neg_test.extend(list_single_for_name_in_maps(processed_root, imp, mode, [test_map]))
        random.shuffle(pos_train)
        per_imp = max(1, len(pos_train) // 9) if len(names) >= 9 else max(1, len(pos_train) // max(1, len(names)))
        neg_train = []
        rest_pool = []
        base_imps = names[:9] if len(names) >= 9 else names
        for imp in base_imps:
            c = neg_pool.get(imp, [])
            random.shuffle(c)
            take = c[:per_imp]
            neg_train.extend(take)
            rest_pool.extend(c[per_imp:])
        if len(neg_train) < len(pos_train):
            random.shuffle(rest_pool)
            need = len(pos_train) - len(neg_train)
            neg_train.extend(rest_pool[:need])
        neg_train = neg_train[:len(pos_train)]
        x_pos_train = build_tensors(pos_train, mode)
        x_neg_train = build_tensors(neg_train, mode)
        if x_pos_train is None or x_neg_train is None:
            return None
        y_pos_train = torch.ones(x_pos_train.shape[0], dtype=torch.float32)
        y_neg_train = torch.zeros(x_neg_train.shape[0], dtype=torch.float32)
        x_train = torch.cat([x_pos_train, x_neg_train], dim=0)
        y_train = torch.cat([y_pos_train, y_neg_train], dim=0)
        idx = torch.randperm(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]
        x_val_pos = build_tensors(pos_val, mode)
        x_val_neg = build_tensors(neg_val, mode)
        x_test_pos = build_tensors(pos_test, mode)
        x_test_neg = build_tensors(neg_test, mode)
        if x_val_pos is None or x_val_neg is None or x_test_pos is None or x_test_neg is None:
            return None
        y_val_pos = torch.ones(x_val_pos.shape[0], dtype=torch.float32)
        y_val_neg = torch.zeros(x_val_neg.shape[0], dtype=torch.float32)
        y_test_pos = torch.ones(x_test_pos.shape[0], dtype=torch.float32)
        y_test_neg = torch.zeros(x_test_neg.shape[0], dtype=torch.float32)
        x_val = torch.cat([x_val_pos, x_val_neg], dim=0)
        y_val = torch.cat([y_val_pos, y_val_neg], dim=0)
        x_test = torch.cat([x_test_pos, x_test_neg], dim=0)
        y_test = torch.cat([y_test_pos, y_test_neg], dim=0)
        input_dim = x_train.shape[-1]
        model = train_binary_unified(x_train, y_train, x_val, y_val, input_dim, device, epochs=epochs, lr=lr, bsz=bsz)
        vacc, vauc, veer = evaluate_unified(model, x_val, y_val, device)
        tacc, tauc, teer = evaluate_unified(model, x_test, y_test, device)
        return {"val": {"acc": vacc, "auc": vauc, "eer": veer}, "test": {"acc": tacc, "auc": tauc, "eer": teer}}
    elif mode == "unified":
        pos_train_pairs = list_pairs_for_name_in_maps(processed_root, target_name, train_maps)
        neg_pool_pairs = {imp: list_pairs_for_name_in_maps(processed_root, imp, train_maps) for imp in names}
        pos_val_pairs = list_pairs_for_name_in_maps(processed_root, target_name, [valid_map])
        neg_val_pairs = []
        for imp in names:
            neg_val_pairs.extend(list_pairs_for_name_in_maps(processed_root, imp, [valid_map]))
        pos_test_pairs = list_pairs_for_name_in_maps(processed_root, target_name, [test_map])
        neg_test_pairs = []
        for imp in names:
            neg_test_pairs.extend(list_pairs_for_name_in_maps(processed_root, imp, [test_map]))
        random.shuffle(pos_train_pairs)
        per_imp = max(1, len(pos_train_pairs) // 9) if len(names) >= 9 else max(1, len(pos_train_pairs) // max(1, len(names)))
        neg_train_pairs = []
        rest_pool = []
        base_imps = names[:9] if len(names) >= 9 else names
        for imp in base_imps:
            c = neg_pool_pairs.get(imp, [])
            random.shuffle(c)
            take = c[:per_imp]
            neg_train_pairs.extend(take)
            rest_pool.extend(c[per_imp:])
        if len(neg_train_pairs) < len(pos_train_pairs):
            random.shuffle(rest_pool)
            need = len(pos_train_pairs) - len(neg_train_pairs)
            neg_train_pairs.extend(rest_pool[:need])
        neg_train_pairs = neg_train_pairs[:len(pos_train_pairs)]
        x_pos_train = build_tensors_unified_from_pairs(pos_train_pairs)
        x_neg_train = build_tensors_unified_from_pairs(neg_train_pairs)
        if x_pos_train is None or x_neg_train is None:
            return None
        y_pos_train = torch.ones(x_pos_train.shape[0], dtype=torch.float32)
        y_neg_train = torch.zeros(x_neg_train.shape[0], dtype=torch.float32)
        x_train = torch.cat([x_pos_train, x_neg_train], dim=0)
        y_train = torch.cat([y_pos_train, y_neg_train], dim=0)
        idx = torch.randperm(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]
        x_val_pos = build_tensors_unified_from_pairs(pos_val_pairs)
        x_val_neg = build_tensors_unified_from_pairs(neg_val_pairs)
        x_test_pos = build_tensors_unified_from_pairs(pos_test_pairs)
        x_test_neg = build_tensors_unified_from_pairs(neg_test_pairs)
        if x_val_pos is None or x_val_neg is None or x_test_pos is None or x_test_neg is None:
            return None
        y_val_pos = torch.ones(x_val_pos.shape[0], dtype=torch.float32)
        y_val_neg = torch.zeros(x_val_neg.shape[0], dtype=torch.float32)
        y_test_pos = torch.ones(x_test_pos.shape[0], dtype=torch.float32)
        y_test_neg = torch.zeros(x_test_neg.shape[0], dtype=torch.float32)
        x_val = torch.cat([x_val_pos, x_val_neg], dim=0)
        y_val = torch.cat([y_val_pos, y_val_neg], dim=0)
        x_test = torch.cat([x_test_pos, x_test_neg], dim=0)
        y_test = torch.cat([y_test_pos, y_test_neg], dim=0)
        input_dim = x_train.shape[-1]
        model = train_binary_unified(x_train, y_train, x_val, y_val, input_dim, device, epochs=epochs, lr=lr, bsz=bsz)
        vacc, vauc, veer = evaluate_unified(model, x_val, y_val, device)
        tacc, tauc, teer = evaluate_unified(model, x_test, y_test, device)
        return {"val": {"acc": vacc, "auc": vauc, "eer": veer}, "test": {"acc": tacc, "auc": tauc, "eer": teer}}
    else:
        pos_train_pairs = list_pairs_for_name_in_maps(processed_root, target_name, train_maps)
        neg_pool_pairs = {imp: list_pairs_for_name_in_maps(processed_root, imp, train_maps) for imp in names}
        pos_val_pairs = list_pairs_for_name_in_maps(processed_root, target_name, [valid_map])
        neg_val_pairs = []
        for imp in names:
            neg_val_pairs.extend(list_pairs_for_name_in_maps(processed_root, imp, [valid_map]))
        pos_test_pairs = list_pairs_for_name_in_maps(processed_root, target_name, [test_map])
        neg_test_pairs = []
        for imp in names:
            neg_test_pairs.extend(list_pairs_for_name_in_maps(processed_root, imp, [test_map]))
        random.shuffle(pos_train_pairs)
        per_imp = max(1, len(pos_train_pairs) // 9) if len(names) >= 9 else max(1, len(pos_train_pairs) // max(1, len(names)))
        neg_train_pairs = []
        rest_pool = []
        base_imps = names[:9] if len(names) >= 9 else names
        for imp in base_imps:
            c = neg_pool_pairs.get(imp, [])
            random.shuffle(c)
            take = c[:per_imp]
            neg_train_pairs.extend(take)
            rest_pool.extend(c[per_imp:])
        if len(neg_train_pairs) < len(pos_train_pairs):
            random.shuffle(rest_pool)
            need = len(pos_train_pairs) - len(neg_train_pairs)
            neg_train_pairs.extend(rest_pool[:need])
        neg_train_pairs = neg_train_pairs[:len(pos_train_pairs)]
        kb_pos_train, ms_pos_train = build_tensors_fusion(pos_train_pairs)
        kb_neg_train, ms_neg_train = build_tensors_fusion(neg_train_pairs)
        if kb_pos_train is None or kb_neg_train is None:
            return None
        y_pos_train = torch.ones(kb_pos_train.shape[0], dtype=torch.float32)
        y_neg_train = torch.zeros(kb_neg_train.shape[0], dtype=torch.float32)
        kb_train = torch.cat([kb_pos_train, kb_neg_train], dim=0)
        ms_train = torch.cat([ms_pos_train, ms_neg_train], dim=0)
        y_train = torch.cat([y_pos_train, y_neg_train], dim=0)
        idx = torch.randperm(kb_train.shape[0])
        kb_train = kb_train[idx]
        ms_train = ms_train[idx]
        y_train = y_train[idx]
        kb_val_pos, ms_val_pos = build_tensors_fusion(pos_val_pairs)
        kb_val_neg, ms_val_neg = build_tensors_fusion(neg_val_pairs)
        kb_test_pos, ms_test_pos = build_tensors_fusion(pos_test_pairs)
        kb_test_neg, ms_test_neg = build_tensors_fusion(neg_test_pairs)
        if kb_val_pos is None or kb_val_neg is None or kb_test_pos is None or kb_test_neg is None:
            return None
        y_val_pos = torch.ones(kb_val_pos.shape[0], dtype=torch.float32)
        y_val_neg = torch.zeros(kb_val_neg.shape[0], dtype=torch.float32)
        y_test_pos = torch.ones(kb_test_pos.shape[0], dtype=torch.float32)
        y_test_neg = torch.zeros(kb_test_neg.shape[0], dtype=torch.float32)
        kb_val = torch.cat([kb_val_pos, kb_val_neg], dim=0)
        ms_val = torch.cat([ms_val_pos, ms_val_neg], dim=0)
        y_val = torch.cat([y_val_pos, y_val_neg], dim=0)
        kb_test = torch.cat([kb_test_pos, kb_test_neg], dim=0)
        ms_test = torch.cat([ms_test_pos, ms_test_neg], dim=0)
        y_test = torch.cat([y_test_pos, y_test_neg], dim=0)
        kb_dim = kb_train.shape[-1]
        ms_dim = ms_train.shape[-1]
        model = train_binary_fusion(kb_train, ms_train, y_train, kb_val, ms_val, y_val, kb_dim, ms_dim, device, epochs=epochs, lr=lr, bsz=bsz)
        vacc, vauc, veer = evaluate_fusion(model, kb_val, ms_val, y_val, device)
        tacc, tauc, teer = evaluate_fusion(model, kb_test, ms_test, y_test, device)
        return {"val": {"acc": vacc, "auc": vauc, "eer": veer}, "test": {"acc": tacc, "auc": tauc, "eer": teer}}


def run_mapcv(processed_root: str, target_name: str, mode: str, device, epochs=30, lr=1e-3, bsz=32, seed=42):
    maps_list = list_maps(processed_root)
    if not maps_list:
        return []
    names = get_all_names_in_maps(processed_root, maps_list)
    names = [n for n in names if n != target_name]
    kfold = len(maps_list)
    results = []
    for k in range(kfold):
        test_map = maps_list[k]
        valid_map = maps_list[(k + 1) % kfold]
        train_maps = [m for i, m in enumerate(maps_list) if i not in [k, (k + 1) % kfold]]
        r = run_fold(processed_root, target_name, mode, train_maps, valid_map, test_map, names, device, epochs=epochs, lr=lr, bsz=bsz, seed=seed)
        if r is None:
            continue
        results.append({"fold": k, "test_map": test_map, "valid_map": valid_map, **r})
        print(f"[{mode}] fold {k+1}/{kfold} val acc={r['val']['acc']:.4f} auc={r['val']['auc']:.4f} eer={r['val']['eer']:.4f} | test acc={r['test']['acc']:.4f} auc={r['test']['auc']:.4f} eer={r['test']['eer']:.4f}")
    if results:
        v_acc = float(np.mean([x["val"]["acc"] for x in results]))
        v_auc = float(np.mean([x["val"]["auc"] for x in results]))
        v_eer = float(np.mean([x["val"]["eer"] for x in results]))
        t_acc = float(np.mean([x["test"]["acc"] for x in results]))
        t_auc = float(np.mean([x["test"]["auc"] for x in results]))
        t_eer = float(np.mean([x["test"]["eer"] for x in results]))
        print(f"[{mode}] avg val acc={v_acc:.4f} auc={v_auc:.4f} eer={v_eer:.4f}")
        print(f"[{mode}] avg test acc={t_acc:.4f} auc={t_auc:.4f} eer={t_eer:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=str, default=os.path.join("d:\\", "Project", "Research", "processed_data"))
    parser.add_argument("--target-name", type=str, default="apEX")
    parser.add_argument("--mode", type=str, choices=["keyboard", "mouse", "unified", "fusion", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bsz", type=int, default=32)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes = ["keyboard", "mouse", "unified", "fusion"] if args.mode == "all" else [args.mode]
    for m in modes:
        run_mapcv(args.processed_root, args.target_name, m, device, epochs=args.epochs, lr=args.lr, bsz=args.bsz, seed=args.seed)


if __name__ == "__main__":
    main()
