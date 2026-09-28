"""
Export a served identification model for the webapp.

Retrains UnifiedModel on the full identification dataset (all folds' train+valid,
deduplicated) with early stopping, then writes a self-contained artifact set to
webapp/inference/models/ so the FastAPI inference service can load it.

Run from src/:  python export_model.py
"""
import os
import sys
import json
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function.models import UnifiedModel  # noqa: E402

with open("../setting.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

OUTPUT_DIR = cfg["path"]["output"]
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
USE_ARCFACE = cfg["model"].get("use-arcface", False)
USE_SE = cfg["model"].get("use-se", False)
WINDOW_SIZE = cfg["time-window"]
SEED = 42

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WEB_MODELS_DIR = os.path.join(REPO_ROOT, "webapp", "inference", "models")
RESULT_COPY_DIR = os.path.join(cfg["path"]["result"], "served_model", TYPE)

FEATURE_NAMES = {
    "keyboard": ["is_walking", "BACK", "RIGHT", "FORWARD", "LEFT"],
    "mouse": ["FIRE", "dp", "dy", "speed", "acc", "jerk"],
    "combined": ["is_walking", "BACK", "RIGHT", "FORWARD", "LEFT", "FIRE", "dp", "dy", "speed", "acc", "jerk"],
}


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class ManifestDataset(Dataset):
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
        return self.data_cache[idx], torch.tensor(self.labels[idx], dtype=torch.long)


def collect_all_data(manifest):
    """Gather train+valid paths/labels from every fold, deduplicated by path."""
    seen = {}
    for key in manifest.keys():
        if not key.startswith("fold_"):
            continue
        for split in ("train", "valid"):
            for p, lab in zip(manifest[key][split]["paths"], manifest[key][split]["labels"]):
                seen[p] = int(lab)
    paths = list(seen.keys())
    labels = [seen[p] for p in paths]
    return paths, labels


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting identification model  arch={ARCH} type={TYPE} device={device}")

    manifest_file = os.path.join(OUTPUT_DIR, f"identification_{TYPE}_folds.json")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    class_mapping = manifest["class_mapping"]
    class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    print(f"num_classes={num_classes} classes={class_names}")

    paths, labels = collect_all_data(manifest)
    print(f"Collected {len(paths)} unique windows")

    # 90/10 train/val split for early stopping
    idx = list(range(len(paths)))
    random.shuffle(idx)
    val_n = max(1, int(0.1 * len(idx)))
    val_idx = set(idx[:val_n])
    train_paths = [paths[i] for i in idx if i not in val_idx]
    train_labels = [labels[i] for i in idx if i not in val_idx]
    val_paths = [paths[i] for i in idx if i in val_idx]
    val_labels = [labels[i] for i in idx if i in val_idx]
    print(f"train={len(train_paths)} val={len(val_paths)}")

    train_ds = ManifestDataset(train_paths, train_labels, desc="Train")
    val_ds = ManifestDataset(val_paths, val_labels, desc="Val")
    train_loader = DataLoader(train_ds, batch_size=BSZ, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BSZ, shuffle=False, num_workers=0)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]

    model = UnifiedModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_attention=USE_ATTENTION,
        num_classes=num_classes,
        arch=ARCH,
        use_arcface=USE_ARCFACE,
        use_se=USE_SE,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    bad = 0

    for epoch in range(EPOCH):
        model.train()
        train_losses = []
        train_preds = []
        train_labels_ep = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()

            outputs = model(xb, yb)
            if isinstance(outputs, tuple):
                logits, clean_logits = outputs
            else:
                logits = clean_logits = outputs

            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            preds = torch.argmax(clean_logits, dim=1)
            train_preds.extend(preds.detach().cpu().numpy().tolist())
            train_labels_ep.extend(yb.detach().cpu().numpy().tolist())
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(np.array(train_preds) == np.array(train_labels_ep)))

        # Validation
        model.eval()
        val_preds, val_labels_ep = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                val_labels_ep.extend(yb.numpy().tolist())
        vacc = float(np.mean(np.array(val_preds) == np.array(val_labels_ep)))

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

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    print(f"Best val acc: {best_val_acc:.4f}")

    # ---- Write artifacts ----
    for d in (WEB_MODELS_DIR, RESULT_COPY_DIR):
        os.makedirs(d, exist_ok=True)

    model_config = {
        "arch": ARCH,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "use_attention": USE_ATTENTION,
        "use_arcface": USE_ARCFACE,
        "use_se": USE_SE,
        "bidirectional": False,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "window_size": WINDOW_SIZE,
        "type": TYPE,
        "feature_names": FEATURE_NAMES[TYPE],
        "class_mapping": class_mapping,
        "seed": SEED,
    }

    ckpt_name = "identification_model.pt"
    torch.save(model.state_dict(), os.path.join(WEB_MODELS_DIR, ckpt_name))
    torch.save(model.state_dict(), os.path.join(RESULT_COPY_DIR, ckpt_name))
    with open(os.path.join(WEB_MODELS_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=4, ensure_ascii=False)
    with open(os.path.join(WEB_MODELS_DIR, "class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=4, ensure_ascii=False)
    with open(os.path.join(RESULT_COPY_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=4, ensure_ascii=False)

    src_discretize = os.path.join(OUTPUT_DIR, "discretized_level_log.json")
    for d in (WEB_MODELS_DIR, RESULT_COPY_DIR):
        shutil.copy2(src_discretize, os.path.join(d, "discretized_level_log.json"))

    print(f"Artifacts written to {WEB_MODELS_DIR}")
    print("[SUCCESS] Model export complete.")


if __name__ == "__main__":
    main()
