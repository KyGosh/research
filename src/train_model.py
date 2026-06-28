import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function.models import UnifiedModel

import yaml
with open("../setting.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PT_DIR = cfg["path"]["pt"]
OUTPUT_DIR = cfg["path"]["output"]
TARGET = cfg["model"]["player"]
TYPE = cfg["model"]["type"]
BSZ = cfg["model"]["bsz"]
LEARNING_RATE = cfg["model"]["lr"]
DECAY = cfg["model"]["weight-decay"]
PATIENCE = cfg["model"]["patience"]
HIDDEN_DIM = cfg["model"]["hidden-dim"]
DROPOUT = cfg["model"]["dropout"]
USE_ATTENTION = cfg["model"]["use-attention"]
NUM_LAYERS = cfg["model"]["num-layers"]
EPOCH = cfg["model"]["epochs"]
ARCH = cfg["model"].get("arch", "lstm")
TASK = cfg.get("task", "authentication")

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

class ManifestDataset(Dataset):
    """
    Original binary dataset loading.
    """
    def __init__(self, pos_paths, neg_paths, desc="Loading Data"):
        self.paths = pos_paths + neg_paths
        self.labels = [1.0] * len(pos_paths) + [0.0] * len(neg_paths)

        self.data_cache = []
        print(f"[{desc}] Pre-loading {len(self.paths)} samples into RAM...")
        for p in tqdm(self.paths, desc=desc, leave=False):
            self.data_cache.append(torch.load(p).clone())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = self.data_cache[idx]
        label = self.labels[idx]
        return x, torch.tensor(label, dtype=torch.float32)


class MultiClassManifestDataset(Dataset):
    """
    New multi-class dataset loading for player identification.
    """
    def __init__(self, paths, labels, desc="Loading Data"):
        self.paths = paths
        self.labels = labels

        self.data_cache = []
        print(f"[{desc}] Pre-loading {len(self.paths)} samples into RAM...")
        for p in tqdm(self.paths, desc=desc, leave=False):
            self.data_cache.append(torch.load(p).clone())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = self.data_cache[idx]
        label = self.labels[idx]
        return x, torch.tensor(label, dtype=torch.long)


def compute_eer(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fnr = 1 - tpr
        i = int(np.argmin(np.abs(fpr - fnr)))
        return float((fpr[i] + fnr[i]) / 2.0)
    except Exception:
        return 0.0

def evaluate(model, loader, device):
    """
    Evaluation for binary authentication.
    """
    model.eval()
    ps = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p = torch.sigmoid(model(xb))
            ps.extend(p.cpu().numpy().tolist())
            ys.extend(yb.numpy().tolist())

    pr = np.array(ps).reshape(-1)
    gt = np.array(ys).reshape(-1)
    pred = (pr > 0.5).astype(np.float32)
    acc = float(np.mean(pred == gt)) if gt.size > 0 else 0.0
    try:
        auc = roc_auc_score(gt, pr) if len(np.unique(gt)) > 1 else 0.0
    except Exception:
        auc = 0.0
    eer = compute_eer(gt, pr)
    return acc, auc, eer, pr, gt


def evaluate_id(model, loader, device):
    """
    Evaluation for multi-class identification.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.numpy().tolist())

    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)
    
    acc = float(np.mean(preds_np == labels_np)) if labels_np.size > 0 else 0.0
    f1 = float(f1_score(labels_np, preds_np, average='macro')) if labels_np.size > 0 else 0.0
    
    return acc, f1, preds_np, labels_np


def plot_far_frr(y_true, y_prob, save_path, title="FAR-FRR Curve"):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    frr = 1 - tpr

    plt.figure(figsize=(14, 7))
    plt.plot(thresholds, fpr, color='blue', label='FAR (False Acceptance Rate)')
    plt.plot(thresholds, frr, color='red', label='FRR (False Rejection Rate)')

    eer_idx = np.argmin(np.abs(fpr - frr))
    eer_val = (fpr[eer_idx] + frr[eer_idx]) / 2

    plt.plot(thresholds[eer_idx], eer_val, 'ko', markersize=8, label=f'EER: {eer_val:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title(title)
    plt.legend(loc='upper center')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train_fold(fold_idx, fold_data, device, plot_dir):
    print(f"\n--- Fold {fold_idx} (Test Session/Map) ---")

    train_ds = ManifestDataset(fold_data["train"]["pos"], fold_data["train"]["neg"], desc=f"Fold {fold_idx} Train")
    val_ds = ManifestDataset(fold_data["valid"]["pos"], fold_data["valid"]["neg"], desc=f"Fold {fold_idx} Val")
    test_ds = ManifestDataset(fold_data["test"]["pos"], fold_data["test"]["neg"], desc=f"Fold {fold_idx} Test")

    train_loader = DataLoader(train_ds, batch_size=BSZ, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BSZ, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BSZ, shuffle=False, num_workers=0)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]

    model = UnifiedModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_attention=USE_ATTENTION,
        arch=ARCH
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state = None
    bad = 0
    history = {"val_auc": []}

    for epoch in range(EPOCH):
        model.train()
        train_losses = []
        train_ps = []
        train_ys = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()

            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            p = torch.sigmoid(logits)
            train_ps.extend(p.detach().cpu().numpy().tolist())
            train_ys.extend(yb.detach().cpu().numpy().tolist())
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        pr = np.array(train_ps).reshape(-1)
        gt = np.array(train_ys).reshape(-1)

        try:
            tauc = roc_auc_score(gt, pr) if len(np.unique(gt)) > 1 else 0.0
        except Exception:
            tauc = 0.0

        vacc, vauc, veer, _, _ = evaluate(model, val_loader, device)
        history["val_auc"].append(vauc)

        print(f"Epoch [{epoch + 1:02d}/{EPOCH}] Loss: {train_loss:.4f}, Train AUC: {tauc:.4f} | Val AUC: {vauc:.4f}")

        if vauc > best_val_auc:
            best_val_auc = vauc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history["val_auc"]) + 1), history["val_auc"], marker='o', color='blue', label='Validation AUC')
    plt.title(f"{fold_idx} Training Trend (Map: {fold_data.get('test_map', 'unknown')})")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{fold_idx}_history.png"))
    plt.close()

    tacc, tauc, teer, pr, gt = evaluate(model, test_loader, device)
    far_frr_path = os.path.join(plot_dir, f"fold_{fold_idx}_far_frr.png")
    plot_far_frr(gt, pr, far_frr_path, title=f"Fold {fold_idx} (Map: {fold_data.get('test_map', 'unknown')}) FAR-FRR")

    print(f"{fold_idx} Test Final -> AUC: {tauc:.4f}, EER: {teer:.4f}")
    return {"val": {"auc": best_val_auc}, "test": {"acc": tacc, "auc": tauc, "eer": teer}}


def train_fold_id(fold_idx, fold_data, device, plot_dir, class_names):
    print(f"\n--- Fold {fold_idx} (Test Session/Map) ---")
    num_classes = len(class_names)

    train_ds = MultiClassManifestDataset(fold_data["train"]["paths"], fold_data["train"]["labels"], desc=f"Fold {fold_idx} Train")
    val_ds = MultiClassManifestDataset(fold_data["valid"]["paths"], fold_data["valid"]["labels"], desc=f"Fold {fold_idx} Val")
    test_ds = MultiClassManifestDataset(fold_data["test"]["paths"], fold_data["test"]["labels"], desc=f"Fold {fold_idx} Test")

    train_loader = DataLoader(train_ds, batch_size=BSZ, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BSZ, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BSZ, shuffle=False, num_workers=0)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]

    # Instantiate model with multi-class output head
    model = UnifiedModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_attention=USE_ATTENTION,
        num_classes=num_classes,
        arch=ARCH
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    bad = 0
    history = {"val_acc": []}

    for epoch in range(EPOCH):
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()

            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.detach().cpu().numpy().tolist())
            train_labels.extend(yb.detach().cpu().numpy().tolist())
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(np.array(train_preds) == np.array(train_labels)))

        vacc, vf1, _, _ = evaluate_id(model, val_loader, device)
        history["val_acc"].append(vacc)

        print(f"Epoch [{epoch + 1:02d}/{EPOCH}] Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Acc: {vacc:.4f}")

        if vacc > best_val_acc:
            best_val_acc = vacc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history["val_acc"]) + 1), history["val_acc"], marker='o', color='blue', label='Validation Acc')
    plt.title(f"{fold_idx} Training Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{fold_idx}_history.png"))
    plt.close()

    tacc, tf1, preds_np, labels_np = evaluate_id(model, test_loader, device)
    cm_path = os.path.join(plot_dir, f"{fold_idx}_confusion_matrix.png")
    plot_confusion_matrix(labels_np, preds_np, class_names, cm_path, title=f"{fold_idx} Confusion Matrix")

    print(f"Fold {fold_idx} Test Final -> Accuracy: {tacc:.4f}, Macro-F1: {tf1:.4f}")
    return {"val": {"acc": best_val_acc}, "test": {"acc": tacc, "f1": tf1}}


def main():
    if TASK == "identification":
        manifest_file = os.path.join(OUTPUT_DIR, f"identification_{TYPE}_folds.json")
    else:
        manifest_file = os.path.join(OUTPUT_DIR, f"{TARGET}_{TYPE}_folds.json")

    import datetime
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = cfg["path"]["result"]
    
    if TASK == "identification":
        run_folder = f"identification_{TYPE}_{time_str}"
    else:
        run_folder = f"authentication_{TARGET}_{TYPE}_{time_str}"
        
    plot_dir = os.path.join(exp_dir, run_folder)
    os.makedirs(plot_dir, exist_ok=True)

    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Resolve seed
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running task: {TASK.upper()} (Model-type: {TYPE})")
    print(f"Using device: {device}")

    # Remove metadata keys if present when collecting fold keys
    fold_keys = sorted([k for k in manifest.keys() if k.startswith("fold_")], key=lambda x: int(x.split("_")[1]))

    results = []

    if TASK == "identification":
        class_mapping = manifest["class_mapping"]
        # Sorted class names based on label indices
        class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]
        
        for key in fold_keys:
            fold_data = manifest[key]
            res = train_fold_id(key, fold_data, device, plot_dir, class_names)
            results.append(res)

        t_accs = [x["test"]["acc"] for x in results]
        t_f1s = [x["test"]["f1"] for x in results]

        avg_acc = float(np.mean(t_accs))
        avg_f1 = float(np.mean(t_f1s))

        _res = {
            "task": TASK,
            "type": TYPE,
            "min_acc": min(t_accs),
            "max_acc": max(t_accs),
            "avg_acc": avg_acc,
            "min_f1": min(t_f1s),
            "max_f1": max(t_f1s),
            "avg_f1": avg_f1,
        }
    else:
        for key in fold_keys:
            fold_data = manifest[key]
            res = train_fold(key, fold_data, device, plot_dir)
            results.append(res)

        t_aucs = [x["test"]["auc"] for x in results]
        t_eers = [x["test"]["eer"] for x in results]

        avg_auc = float(np.mean(t_aucs))
        avg_eer = float(np.mean(t_eers))

        _res = {
            "name": TARGET,
            "type": TYPE,
            "min_auc": min(t_aucs),
            "max_auc": max(t_aucs),
            "avg_auc": avg_auc,
            "min_eer": min(t_eers),
            "max_eer": max(t_eers),
            "avg_eer": avg_eer,
        }

    total_performance = os.path.join(exp_dir, "total_performance.json")
    if os.path.exists(total_performance):
        try:
            with open(total_performance, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"experiments": []}
    else:
        data = {"experiments": []}

    data["experiments"].append(_res)

    with open(total_performance, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Save a copy of the run results directly inside the timestamped experiment directory
    local_perf_path = os.path.join(plot_dir, "performance.json")
    with open(local_perf_path, "w", encoding="utf-8") as f:
        json.dump(_res, f, indent=4, ensure_ascii=False)

    if TASK == "identification":
        print(f"\n================== FINAL RESULTS ==================\nAvg Test Acc: {avg_acc:.4f}, Avg Test Macro-F1: {avg_f1:.4f}")
        # Summary Plot
        plt.figure(figsize=(12, 6))
        folds = range(len(results))
        plt.plot(folds, t_accs, marker='D', markersize=8, color='forestgreen', label='Test Acc')
        plt.plot(folds, t_f1s, marker='X', markersize=8, color='crimson', label='Test Macro-F1')
        plt.axhline(y=avg_acc, color='forestgreen', linestyle='--', alpha=0.5, label=f'Avg Acc ({avg_acc:.3f})')
        plt.axhline(y=avg_f1, color='crimson', linestyle='--', alpha=0.5, label=f'Avg Macro-F1 ({avg_f1:.3f})')
        plt.title(f"10-Fold Identification Summary")
        plt.xlabel("Fold Index")
        plt.ylabel("Score")
        plt.xticks(folds)
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "summary_performance.png"))
        plt.close()
    else:
        print(f"\n================== FINAL RESULTS ==================\nAvg Test AUC: {avg_auc:.4f}, Avg Test EER: {avg_eer:.4f}")
        # Summary Plot
        plt.figure(figsize=(12, 6))
        folds = range(len(results))
        plt.plot(folds, t_aucs, marker='D', markersize=8, color='forestgreen', label='Test AUC')
        plt.plot(folds, t_eers, marker='X', markersize=8, color='crimson', label='Test EER')
        plt.axhline(y=avg_auc, color='forestgreen', linestyle='--', alpha=0.5, label=f'Avg AUC ({avg_auc:.3f})')
        plt.axhline(y=avg_eer, color='crimson', linestyle='--', alpha=0.5, label=f'Avg EER ({avg_eer:.3f})')
        plt.title(f"10-Fold Cross Validation Summary")
        plt.xlabel("Fold Index")
        plt.ylabel("Score")
        plt.xticks(folds)
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "summary_performance.png"))
        plt.close()

    print(f"All plots saved to {plot_dir}")

if __name__ == "__main__":
    main()
