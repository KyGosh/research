import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

from scripts.ml.datasets import list_name_dirs, list_single_for_name, list_pairs_for_name, read_sample, read_unified_sample
from scripts.ml.models import UnifiedModel, FusionModel


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# 怎么处理unified情况--未知--list_single_for_name没有处理mode为unified时的情况
def sample_auth_sets(processed_root: str, target_name: str, mode: str, pos_n: int = 90, neg_n: int = 90):
    names = list(list_name_dirs(processed_root).keys())
    names = [n for n in names if n != target_name]
    random.shuffle(names)
    impostors = names[:9] if len(names) >= 9 else names
    pos_files = []
    neg_files = []
    if mode in ["keyboard", "mouse", "unified"]:
        pos_all = list_single_for_name(processed_root, target_name, "keyboard" if mode == "keyboard" else "mouse")
        random.shuffle(pos_all)
        pos_files = pos_all[:pos_n]
        for imp in impostors:
            imp_all = list_single_for_name(processed_root, imp, "keyboard" if mode == "keyboard" else "mouse")
            random.shuffle(imp_all)
            neg_files.extend(imp_all[:10])
        neg_files = neg_files[:neg_n]
    elif mode == "fusion":
        pos_pairs = list_pairs_for_name(processed_root, target_name)
        random.shuffle(pos_pairs)
        pos_files = pos_pairs[:pos_n]
        for imp in impostors:
            imp_pairs = list_pairs_for_name(processed_root, imp)
            random.shuffle(imp_pairs)
            neg_files.extend(imp_pairs[:10])
        neg_files = neg_files[:neg_n]
    return pos_files, neg_files


def build_tensors(files, mode):
    xs = []
    for f in files:
        if mode in ["keyboard", "mouse"]:
            x = read_sample(f, mode)
        elif mode == "unified":
            x = read_unified_sample(f, f.replace("_kb.csv", "_ms.csv"))
        else:
            x = None
        if x is None or x.shape[0] != 1920:
            continue
        xs.append(x)
    if not xs:
        return None
    x = np.stack(xs, axis=0).astype(np.float32)
    return torch.tensor(x, dtype=torch.float32)


def build_tensors_fusion(pairs):
    kb_xs = []
    ms_xs = []
    for kb, ms, key in pairs:
        k = read_sample(kb, "keyboard")
        m = read_sample(ms, "mouse")
        if k is None or m is None or k.shape[0] != 1920 or m.shape[0] != 1920:
            continue
        kb_xs.append(k)
        ms_xs.append(m)
    if not kb_xs:
        return None, None
    kb = np.stack(kb_xs, axis=0).astype(np.float32)
    ms = np.stack(ms_xs, axis=0).astype(np.float32)
    return torch.tensor(kb, dtype=torch.float32), torch.tensor(ms, dtype=torch.float32)


def split_data(x_pos, x_neg):
    y_pos = torch.ones(x_pos.shape[0], dtype=torch.float32)
    y_neg = torch.zeros(x_neg.shape[0], dtype=torch.float32)
    x = torch.cat([x_pos, x_neg], dim=0)
    y = torch.cat([y_pos, y_neg], dim=0)
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


def train_binary_unified(x_train, y_train, x_val, y_val, x_test, y_test, input_dim, device, epochs=50, lr=1e-3, bsz=32, patience=5):
    model = UnifiedModel(input_dim=input_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # ✅ 不提前放 GPU
    train_ds = TensorDataset(x_train, y_train)
    val_ds   = TensorDataset(x_val,   y_val)
    test_ds  = TensorDataset(x_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=bsz)
    test_loader  = DataLoader(test_ds, batch_size=bsz)

    best_val_loss = float("inf")
    best_state = None
    early_stop_counter = 0

    # =========================
    # Training Loop
    # =========================
    for ep in range(epochs):

        # ===== Train =====
        model.train()
        train_loss_list = []
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()

            logits = model(xb)
            loss = loss_fn(logits, yb)

            loss.backward()
            opt.step()

            train_loss_list.append(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            train_correct += (preds == yb).sum().item()
            train_total += yb.numel()

        train_loss = float(np.mean(train_loss_list))
        train_acc = train_correct / train_total


        # ===== Validation =====
        model.eval()
        val_loss_list = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = loss_fn(logits, yb)

                val_loss_list.append(loss.item())

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                val_correct += (preds == yb).sum().item()
                val_total += yb.numel()

        val_loss = float(np.mean(val_loss_list))
        val_acc = val_correct / val_total

        print(f"Epoch [{ep+1:02d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ===== Early Stopping =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break


    # =========================
    # Load Best Model
    # =========================
    if best_state is not None:
        model.load_state_dict(best_state)

    # =========================
    # Test Evaluation
    # =========================
    model.eval()
    preds = []
    gts = []
    prob_list = []

    with torch.no_grad():
        for xb, yb in test_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            probs = torch.sigmoid(logits)

            preds.extend((probs > 0.5).cpu().numpy().tolist())
            prob_list.extend(probs.cpu().numpy().tolist())
            gts.extend(yb.cpu().numpy().tolist())

    preds = np.array(preds)
    gts = np.array(gts)
    prob_list = np.array(prob_list)

    test_acc = float(np.mean(preds == gts))

    try:
        test_auc = roc_auc_score(gts, prob_list)
    except:
        test_auc = 0.0

    print(f"\nFinal Test Acc: {test_acc:.4f}")
    print(f"Final Test AUC: {test_auc:.4f}")

    return model, test_acc


def train_binary_fusion(kb_train, ms_train, y_train, kb_val, ms_val, y_val, kb_test, ms_test, y_test, kb_dim, ms_dim, device, epochs=50, lr=1e-3, bsz=32, patience=5):
    model = FusionModel(kb_input_dim=kb_dim, ms_input_dim=ms_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # ✅ 不提前放到 GPU
    train_ds = TensorDataset(kb_train, ms_train, y_train)
    val_ds   = TensorDataset(kb_val,   ms_val,   y_val)
    test_ds  = TensorDataset(kb_test,  ms_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=bsz)
    test_loader  = DataLoader(test_ds, batch_size=bsz)

    best_val_loss = float("inf")
    best_state = None
    early_stop_counter = 0

    # =========================
    # Training Loop
    # =========================
    for ep in range(epochs):

        # ===== Train =====
        model.train()
        train_loss_list = []
        train_correct = 0
        train_total = 0

        for kb_x, ms_x, yb in train_loader:

            kb_x = kb_x.to(device)
            ms_x = ms_x.to(device)
            yb   = yb.to(device)

            opt.zero_grad()

            logits = model(kb_x, ms_x)
            loss = loss_fn(logits, yb)

            loss.backward()
            opt.step()

            train_loss_list.append(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            train_correct += (preds == yb).sum().item()
            train_total += yb.numel()

        train_loss = float(np.mean(train_loss_list))
        train_acc = train_correct / train_total


        # ===== Validation =====
        model.eval()
        val_loss_list = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for kb_x, ms_x, yb in val_loader:

                kb_x = kb_x.to(device)
                ms_x = ms_x.to(device)
                yb   = yb.to(device)

                logits = model(kb_x, ms_x)
                loss = loss_fn(logits, yb)

                val_loss_list.append(loss.item())

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                val_correct += (preds == yb).sum().item()
                val_total += yb.numel()

        val_loss = float(np.mean(val_loss_list))
        val_acc = val_correct / val_total

        print(f"Epoch [{ep+1:02d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ===== Early Stopping =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    # =========================
    # Load Best Model
    # =========================
    if best_state is not None:
        model.load_state_dict(best_state)

    # =========================
    # Test Evaluation
    # =========================
    model.eval()
    preds = []
    gts = []
    prob_list = []

    with torch.no_grad():
        for kb_x, ms_x, yb in test_loader:

            kb_x = kb_x.to(device)
            ms_x = ms_x.to(device)
            yb   = yb.to(device)

            logits = model(kb_x, ms_x)
            probs = torch.sigmoid(logits)

            preds.extend((probs > 0.5).cpu().numpy().tolist())
            prob_list.extend(probs.cpu().numpy().tolist())
            gts.extend(yb.cpu().numpy().tolist())

    preds = np.array(preds)
    gts = np.array(gts)
    prob_list = np.array(prob_list)

    test_acc = float(np.mean(preds == gts))

    # AUC（更适合 authentication）
    try:
        test_auc = roc_auc_score(gts, prob_list)
    except:
        test_auc = 0.0

    print(f"\nFinal Test Acc: {test_acc:.4f}")
    print(f"Final Test AUC: {test_auc:.4f}")

    return model, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=str, default=os.path.join("d:\\", "Project", "Research", "processed_data"))
    # parser.add_argument("--target-name", type=str, required=True)
    parser.add_argument("--target-name", type=str, default="ropz")
    # parser.add_argument("--mode", type=str, choices=["keyboard", "mouse", "unified", "fusion"], required=True)
    parser.add_argument("--mode", type=str, default="keyboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bsz", type=int, default=32)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos, neg = sample_auth_sets(args.processed_root, args.target_name, args.mode)
    if args.mode in ["keyboard", "mouse", "unified"]:
        x_pos = build_tensors(pos, args.mode)
        x_neg = build_tensors(neg, args.mode)
        if x_pos is None or x_neg is None:
            return
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_pos, x_neg)
        input_dim = x_train.shape[-1]
        _, acc = train_binary_unified(x_train, y_train, x_val, y_val, x_test, y_test, input_dim, device, epochs=args.epochs, lr=args.lr, bsz=args.bsz)
        print(f"acc={acc:.4f}")
    else:
        kb_pos, ms_pos = build_tensors_fusion(pos)
        kb_neg, ms_neg = build_tensors_fusion(neg)
        if kb_pos is None or kb_neg is None:
            return
        y_pos = torch.ones(kb_pos.shape[0], dtype=torch.float32)
        y_neg = torch.zeros(kb_neg.shape[0], dtype=torch.float32)
        kb = torch.cat([kb_pos, kb_neg], dim=0)
        ms = torch.cat([ms_pos, ms_neg], dim=0)
        y = torch.cat([y_pos, y_neg], dim=0)
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
        kb_dim = kb_train.shape[-1]
        ms_dim = ms_train.shape[-1]
        _, acc = train_binary_fusion(kb_train, ms_train, y_train, kb_val, ms_val, y_val, kb_test, ms_test, y_test, kb_dim, ms_dim, device, epochs=args.epochs, lr=args.lr, bsz=args.bsz)
        print(f"acc={acc:.4f}")


if __name__ == "__main__":
    main()

