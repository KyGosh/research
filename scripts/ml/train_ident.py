import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets import list_name_dirs, list_single_for_name, list_pairs_for_name, read_sample, read_unified_sample
from .models import UnifiedModel, FusionModel


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_ident_sets(processed_root: str, mode: str, per_name_n: int = 100):
    names = list(list_name_dirs(processed_root).keys())
    data = {}
    for n in names:
        if mode in ["keyboard", "mouse"]:
            files = list_single_for_name(processed_root, n, mode)
        elif mode == "unified":
            files = list_single_for_name(processed_root, n, "keyboard")
        else:
            files = list_pairs_for_name(processed_root, n)
        random.shuffle(files)
        sel = files[:per_name_n]
        rest = files[per_name_n:]
        data[n] = {"selected": sel, "rest": rest}
    return names, data


def build_tensors_multiclass(names, data, mode):
    xs = []
    ys = []
    for i, n in enumerate(names):
        for f in data[n]["selected"]:
            if mode in ["keyboard", "mouse"]:
                x = read_sample(f, mode)
            elif mode == "unified":
                x = read_unified_sample(f, f.replace("_kb.csv", "_ms.csv"))
            else:
                x = None
            if x is None or x.shape[0] != 1920:
                continue
            xs.append(x)
            ys.append(i)
    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def build_tensors_multiclass_fusion(names, data):
    kb_xs = []
    ms_xs = []
    ys = []
    for i, n in enumerate(names):
        for kb, ms, key in data[n]["selected"]:
            k = read_sample(kb, "keyboard")
            m = read_sample(ms, "mouse")
            if k is None or m is None or k.shape[0] != 1920 or m.shape[0] != 1920:
                continue
            kb_xs.append(k)
            ms_xs.append(m)
            ys.append(i)
    kb = np.stack(kb_xs, axis=0).astype(np.float32)
    ms = np.stack(ms_xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return torch.tensor(kb, dtype=torch.float32), torch.tensor(ms, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def split_multi(x, y):
    idx = torch.arange(x.shape[0])
    idx = idx[torch.randperm(idx.shape[0])]
    x = x[idx]
    y = y[idx]
    n = x.shape[0]
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]
    return x_train, y_train, x_val, y_val, x_test, y_test


def split_multi_fusion(kb, ms, y):
    idx = torch.arange(kb.shape[0])
    idx = idx[torch.randperm(idx.shape[0])]
    kb = kb[idx]
    ms = ms[idx]
    y = y[idx]
    n = kb.shape[0]
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    kb_train, ms_train, y_train = kb[:n_train], ms[:n_train], y[:n_train]
    kb_val, ms_val, y_val = kb[n_train:n_train + n_val], ms[n_train:n_train + n_val], y[n_train:n_train + n_val]
    kb_test, ms_test, y_test = kb[n_train + n_val:], ms[n_train + n_val:], y[n_train + n_val:]
    return kb_train, ms_train, y_train, kb_val, ms_val, y_val, kb_test, ms_test, y_test


def train_multi_unified(x_train, y_train, x_val, y_val, x_test, y_test, input_dim, num_classes, device, epochs=50, lr=1e-3, bsz=32):
    model = UnifiedModel(input_dim=input_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(x_train.to(device), y_train.to(device))
    val_ds = TensorDataset(x_val.to(device), y_val.to(device))
    test_ds = TensorDataset(x_test.to(device), y_test.to(device))
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bsz)
    best_val = None
    best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_losses.append(loss.item())
        v = float(np.mean(val_losses)) if val_losses else None
        if best_val is None or (v is not None and v < best_val):
            best_val = v
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(test_ds, batch_size=bsz)
        preds = []
        gts = []
        for xb, yb in test_loader:
            logits = model(xb)
            p = torch.argmax(logits, dim=-1)
            preds.extend(p.cpu().numpy().tolist())
            gts.extend(yb.cpu().numpy().tolist())
        acc = float(np.mean(np.array(preds) == np.array(gts))) if preds else 0.0
    return model, acc


def train_multi_fusion(kb_train, ms_train, y_train, kb_val, ms_val, y_val, kb_test, ms_test, y_test, kb_dim, ms_dim, num_classes, device, epochs=50, lr=1e-3, bsz=32):
    model = FusionModel(kb_input_dim=kb_dim, ms_input_dim=ms_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(kb_train.to(device), ms_train.to(device), y_train.to(device))
    val_ds = TensorDataset(kb_val.to(device), ms_val.to(device), y_val.to(device))
    test_ds = TensorDataset(kb_test.to(device), ms_test.to(device), y_test.to(device))
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bsz)
    best_val = None
    best_state = None
    for ep in range(epochs):
        model.train()
        for kb_x, ms_x, yb in train_loader:
            opt.zero_grad()
            logits = model(kb_x, ms_x)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            for kb_x, ms_x, yb in val_loader:
                logits = model(kb_x, ms_x)
                loss = loss_fn(logits, yb)
                val_losses.append(loss.item())
        v = float(np.mean(val_losses)) if val_losses else None
        if best_val is None or (v is not None and v < best_val):
            best_val = v
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(test_ds, batch_size=bsz)
        preds = []
        gts = []
        for kb_x, ms_x, yb in test_loader:
            logits = model(kb_x, ms_x)
            p = torch.argmax(logits, dim=-1)
            preds.extend(p.cpu().numpy().tolist())
            gts.extend(yb.cpu().numpy().tolist())
        acc = float(np.mean(np.array(preds) == np.array(gts))) if preds else 0.0
    return model, acc


def sample_holdout(processed_root: str, names, data, mode):
    items = []
    for i, n in enumerate(names):
        rest = data[n]["rest"]
        if not rest:
            continue
        f = random.choice(rest)
        items.append((i, f))
    return items


def eval_holdout_unified(model, processed_root, items):
    preds = []
    gts = []
    for i, f in items:
        x = read_unified_sample(f, f.replace("_kb.csv", "_ms.csv"))
        if x is None or x.shape[0] != 1920:
            continue
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        logits = model(x)
        p = torch.argmax(logits, dim=-1).item()
        preds.append(p)
        gts.append(i)
    acc = float(np.mean(np.array(preds) == np.array(gts))) if preds else 0.0
    return acc


def eval_holdout_fusion(model, items):
    preds = []
    gts = []
    for i, pair in items:
        kb, ms, key = pair
        k = read_sample(kb, "keyboard")
        m = read_sample(ms, "mouse")
        if k is None or m is None or k.shape[0] != 1920 or m.shape[0] != 1920:
            continue
        kb_t = torch.tensor(k, dtype=torch.float32).unsqueeze(0)
        ms_t = torch.tensor(m, dtype=torch.float32).unsqueeze(0)
        logits = model(kb_t, ms_t)
        p = torch.argmax(logits, dim=-1).item()
        preds.append(p)
        gts.append(i)
    acc = float(np.mean(np.array(preds) == np.array(gts))) if preds else 0.0
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=str, default=os.path.join("d:\\", "Project", "Research", "processed_data"))
    parser.add_argument("--mode", type=str, choices=["keyboard", "mouse", "unified", "fusion"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bsz", type=int, default=32)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, data = build_ident_sets(args.processed_root, args.mode, per_name_n=100)
    if args.mode in ["keyboard", "mouse", "unified"]:
        x, y = build_tensors_multiclass(names, data, args.mode if args.mode != "unified" else "unified")
        x_train, y_train, x_val, y_val, x_test, y_test = split_multi(x, y)
        input_dim = x_train.shape[-1]
        num_classes = len(names)
        model, acc = train_multi_unified(x_train, y_train, x_val, y_val, x_test, y_test, input_dim, num_classes, device, epochs=args.epochs, lr=args.lr, bsz=args.bsz)
        print(f"acc={acc:.4f}")
        items = sample_holdout(args.processed_root, names, data, args.mode)
        hacc = eval_holdout_unified(model, args.processed_root, items)
        print(f"holdout_acc={hacc:.4f}")
    else:
        kb, ms, y = build_tensors_multiclass_fusion(names, data)
        kb_train, ms_train, y_train, kb_val, ms_val, y_val, kb_test, ms_test, y_test = split_multi_fusion(kb, ms, y)
        kb_dim = kb_train.shape[-1]
        ms_dim = ms_train.shape[-1]
        num_classes = len(names)
        model, acc = train_multi_fusion(kb_train, ms_train, y_train, kb_val, ms_val, y_val, kb_test, ms_test, y_test, kb_dim, ms_dim, num_classes, device, epochs=args.epochs, lr=args.lr, bsz=args.bsz)
        print(f"acc={acc:.4f}")
        items = []
        for i, n in enumerate(names):
            rest = data[n]["rest"]
            pairs = [p for p in rest]
            if not pairs:
                continue
            items.append((i, random.choice(pairs)))
        hacc = eval_holdout_fusion(model, items)
        print(f"holdout_acc={hacc:.4f}")


if __name__ == "__main__":
    main()

