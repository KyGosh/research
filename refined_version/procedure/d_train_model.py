import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from refined_version.function.models import UnifiedModel


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class ManifestDataset(Dataset):
    def __init__(self, pos_paths, neg_paths, desc="Loading Data"):
        self.paths = pos_paths + neg_paths
        self.labels = [1.0] * len(pos_paths) + [0.0] * len(neg_paths)

        # 🔥 内存预加载逻辑
        self.data_cache = []
        print(f"[{desc}] Pre-loading {len(self.paths)} samples into RAM...")
        for p in tqdm(self.paths, desc=desc, leave=False):
            # 将 Tensor 加载到 CPU 内存中缓存
            self.data_cache.append(torch.load(p).clone())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 🚀 直接从内存返回，不再产生磁盘 IO
        x = self.data_cache[idx]
        label = self.labels[idx]
        return x, torch.tensor(label, dtype=torch.float32)


def compute_eer(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fnr = 1 - tpr
        i = int(np.argmin(np.abs(fpr - fnr)))
        return float((fpr[i] + fnr[i]) / 2.0)
    except Exception:
        return 0.0


def evaluate(model, loader, device):
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


def plot_far_frr(y_true, y_prob, save_path, title="FAR-FRR Curve"):
    """绘制 FAR (FPR) 和 FRR (1-TPR) 随阈值变化的曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    frr = 1 - tpr

    # 海报风格
    sns.set_context("poster", font_scale=1.4)

    plt.figure(figsize=(14, 7))

    # 全局字体（匹配40pt正文）
    plt.rcParams.update({
        'font.size': 36,
        'axes.titlesize': 40,
        'axes.labelsize': 36,
        'xtick.labelsize': 34,
        'ytick.labelsize': 34,
        'legend.fontsize': 32
    })

    plt.plot(thresholds, fpr, color='blue', label='FAR (False Acceptance Rate)', linewidth=2)
    plt.plot(thresholds, frr, color='red', label='FRR (False Rejection Rate)', linewidth=2)

    # 寻找 EER 的交点
    eer_idx = np.argmin(np.abs(fpr - frr))
    eer_val = (fpr[eer_idx] + frr[eer_idx]) / 2

    plt.plot(thresholds[eer_idx], eer_val, 'ko', markersize=8, label=f'EER: {eer_val:.4f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title("FAR-FRR Curve")
    plt.legend(loc='upper center')
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_fold(fold_idx, fold_data, args, device, plot_dir):
    print(f"\n--- Fold {fold_idx} (Test: {fold_data['test_map']}) ---")

    # 创建数据集（触发内存预加载）
    train_ds = ManifestDataset(fold_data["train"]["pos"], fold_data["train"]["neg"], desc=f"Fold {fold_idx} Train")
    val_ds = ManifestDataset(fold_data["valid"]["pos"], fold_data["valid"]["neg"], desc=f"Fold {fold_idx} Val")
    test_ds = ManifestDataset(fold_data["test"]["pos"], fold_data["test"]["neg"], desc=f"Fold {fold_idx} Test")

    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bsz, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.bsz, shuffle=False, num_workers=0)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]

    model = UnifiedModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_attention=args.use_attention
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state = None
    bad = 0

    history = {"val_auc": []}

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        vacc, vauc, veer, _, _ = evaluate(model, val_loader, device)
        history["val_auc"].append(vauc)

        print(f"Epoch [{epoch + 1:02d}/{args.epochs}] Loss: {train_loss:.4f} | Val AUC: {vauc:.4f}")

        if vauc > best_val_auc:
            best_val_auc = vauc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 海报风格（只影响视觉，不改变数据）
    sns.set_context("poster", font_scale=1.4)

    plt.figure(figsize=(12, 6))

    # 全局字体（和你40pt正文匹配）
    plt.rcParams.update({
        'font.size': 36,
        'axes.titlesize': 40,
        'axes.labelsize': 36,
        'xtick.labelsize': 34,
        'ytick.labelsize': 34,
        'legend.fontsize': 32
    })

    # 绘制曲线（增强可读性）
    epochs = range(1, len(history["val_auc"]) + 1)
    plt.plot(
        epochs,
        history["val_auc"],
        marker='o',
        markersize=8,  # 原来默认太小
        linewidth=3,  # 线更粗，适合海报
        color='blue',
        label='Validation AUC'
    )

    # 标题与标签（统一用rcParams控制大小）
    plt.title("Training Trend (Fold 0, Player_1)")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")

    # 网格稍微加粗
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)

    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"fold_{fold_idx}_history.png"), dpi=300)
    plt.close()

    # 最终评估与绘制 FAR/FRR 曲线
    tacc, tauc, teer, pr, gt = evaluate(model, test_loader, device)

    far_frr_path = os.path.join(plot_dir, f"fold_{fold_idx}_far_frr.png")
    plot_far_frr(gt, pr, far_frr_path, title=f"Fold {fold_idx} (Map: {fold_data['test_map']}) FAR-FRR")

    print(f"Fold {fold_idx} Test Final -> AUC: {tauc:.4f}, EER: {teer:.4f}")

    return {"val": {"auc": vauc}, "test": {"acc": tacc, "auc": tauc, "eer": teer}}

PLAYERS = ['apEX', 'FalleN', 'flameZ', 'KSCERATO', 'mezii', 'molodoy', 'ropz', 'YEKINDAR', 'yuurih', 'ZywOo']
TYPES = ["mouse", "keyboard", "combined"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", type=str, default="apEX")
    parser.add_argument("--type", type=str, default="combined")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use-attention", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=0, help="Set to 0 when using RAM cache")
    args = parser.parse_args()

    manifest_file = os.path.join("d:\\", "Project", "Research", "output_eer", f"{args.player}_{args.type}_folds.json")
    exp_dir = os.path.dirname(manifest_file)
    plot_dir = os.path.join(exp_dir, "plots", f"{args.player}", f"{args.type}")
    os.makedirs(plot_dir, exist_ok=True)

    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    set_seed(manifest.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    fold_keys = sorted(manifest.keys(), key=lambda x: int(x.split("_")[1]))
    for key in fold_keys:
        fold_data = manifest[key]
        res = train_fold(key, fold_data, args, device, plot_dir)
        results.append(res)

    t_aucs = [x["test"]["auc"] for x in results]
    t_eers = [x["test"]["eer"] for x in results]

    min_auc = min(t_aucs)
    max_auc = max(t_aucs)
    min_eer = min(t_eers)
    max_eer = max(t_eers)
    avg_auc = float(np.mean(t_aucs))
    avg_eer = float(np.mean(t_eers))

    _res = {
        "name" : args.player,
        "type" : args.type,
        "min_auc": min_auc,
        "max_auc": max_auc,
        "avg_auc": avg_auc,
        "min_eer": min_eer,
        "max_eer": max_eer,
        "avg_eer": avg_eer,
    }

    total_performance = os.path.join(exp_dir, "total_performance.json")
    # 读取旧数据
    if os.path.exists(total_performance):
        with open(total_performance, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"experiments": []}

    # 追加
    data["experiments"].append(_res)

    # 写回
    with open(total_performance, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(
        f"\n================== FINAL RESULTS ==================\nAvg Test AUC: {avg_auc:.4f}, Avg Test EER: {avg_eer:.4f}")

    # 汇总图
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
