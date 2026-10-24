import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
import warnings
import yaml
from pathlib import Path
import datetime

# Import components from train_model incrementally
from train_model import (
    train_fold,
    train_fold_id,
    set_seed,
    compute_eer,
    plot_far_frr,
    plot_confusion_matrix,
    OUTPUT_DIR,
    TARGET,
    ARCH,
    TASK,
    cfg
)

ENSEMBLE_CFG = cfg.get("ensemble", {})
MODALITIES = ENSEMBLE_CFG.get("modalities", ["keyboard", "mouse", "combined"])
GRID_STEP = ENSEMBLE_CFG.get("soft_vote_grid_step", 0.05)
DEMO_DIR = cfg.get("path", {}).get("demos", "d:\\Project\\demo_file")
PT_DIR = cfg.get("path", {}).get("pt", "d:\\Project\\Research\\dataset\\pt_data")


def load_player_groups(demo_dir=DEMO_DIR, pt_dir=PT_DIR, custom_json_path=None):
    """
    Loads or extracts player groups ('pro_junior' vs 'top_20').
    1. Checks if custom_json_path or 'player_groups.json' in workspace exists.
    2. Otherwise scans demo_dir (pro_junior/top_20) and pt_dir to build groups dynamically.
    Returns: dict {"pro_junior": list_of_players, "top_20": list_of_players}
    """
    candidates = [
        custom_json_path,
        os.path.join(os.path.dirname(__file__), "..", "player_groups.json"),
        os.path.join(OUTPUT_DIR, "player_groups.json"),
        "player_groups.json"
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    groups = json.load(f)
                if "pro_junior" in groups and "top_20" in groups:
                    print(f"[INFO] Loaded player groups from: {p}")
                    return groups
            except Exception as e:
                print(f"[WARNING] Failed reading {p}: {e}")

    # Fallback: scan demo_dir and pt_dir
    print(f"[INFO] Scanning {demo_dir} to build player groups...")
    demo_to_group = {}
    for grp in ["pro_junior", "top_20"]:
        grp_path = os.path.join(demo_dir, grp)
        if os.path.exists(grp_path):
            for root, _, files in os.walk(grp_path):
                for f in files:
                    if f.endswith(".dem"):
                        demo_to_group[os.path.splitext(f)[0]] = grp

    group_players = {"pro_junior": set(), "top_20": set()}
    if os.path.exists(pt_dir):
        for player in os.listdir(pt_dir):
            p_path = os.path.join(pt_dir, player)
            if not os.path.isdir(p_path):
                continue
            matches = [m for m in os.listdir(p_path) if m.startswith("match_")]
            for m in matches:
                for dname, grp in demo_to_group.items():
                    d_norm = dname.replace("-", "_")
                    if d_norm in m or (d_norm.split("_")[0] in m and d_norm.split("_")[2] in m):
                        group_players[grp].add(player)

    result = {k: sorted(list(v)) for k, v in group_players.items()}
    print(f"[INFO] Extracted player groups: {len(result['pro_junior'])} pro_junior, {len(result['top_20'])} top_20 players.")
    return result


def compute_topk_acc(probs, labels, k=3):
    """
    Compute Top-K accuracy.
    probs: (N, C) probability matrix
    labels: (N,) ground truth label indices
    """
    if probs is None or len(probs) == 0:
        return 0.0
    num_classes = probs.shape[1]
    actual_k = min(k, num_classes)
    topk_preds = np.argsort(probs, axis=1)[:, -actual_k:]
    hits = [labels[i] in topk_preds[i] for i in range(len(labels))]
    return float(np.mean(hits))


def compute_group_acc(preds, labels, class_names, player_to_group, group_arrays):
    """
    Compute group accuracy: probability that predicted player belongs to the same group as ground truth player.
    Corresponds to: predict_player in arr[type[true_player]]
    """
    if len(labels) == 0:
        return 0.0
    hits = []
    for p_idx, l_idx in zip(preds, labels):
        true_player = class_names[l_idx]
        pred_player = class_names[p_idx]
        true_group = player_to_group.get(true_player)
        if true_group and true_group in group_arrays:
            is_hit = pred_player in group_arrays[true_group]
            hits.append(1.0 if is_hit else 0.0)
        else:
            # Fallback if true player group is untagged: exact player match
            hits.append(1.0 if true_player == pred_player else 0.0)
    return float(np.mean(hits)) if hits else 0.0


def hard_vote_id(*probs_args):
    """
    Hard voting with confidence tie-breaking for multi-class identification.
    probs_args: list/tuple of (N, C) arrays, or multiple (N, C) arrays passed as arguments.
    """
    if len(probs_args) == 1 and isinstance(probs_args[0], (list, tuple)):
        probs_list = probs_args[0]
    else:
        probs_list = list(probs_args)

    n_mods = len(probs_list)
    if n_mods == 1:
        return np.argmax(probs_list[0], axis=1)

    preds = np.stack([np.argmax(p, axis=1) for p in probs_list], axis=1)  # (N, M)
    confs = np.stack([np.max(p, axis=1) for p in probs_list], axis=1)     # (N, M)

    n_samples = preds.shape[0]
    final_preds = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        row_preds = preds[i]
        row_confs = confs[i]
        vals, counts = np.unique(row_preds, return_counts=True)
        max_count = counts.max()
        candidates = vals[counts == max_count]
        if len(candidates) == 1:
            final_preds[i] = candidates[0]
        else:
            # Tie: select candidate with highest confidence score among tied votes
            candidate_indices = [idx for idx, p in enumerate(row_preds) if p in candidates]
            best_idx = candidate_indices[int(np.argmax(row_confs[candidate_indices]))]
            final_preds[i] = row_preds[best_idx]

    return final_preds


def hard_vote_auth(*probs_args):
    """
    Hard voting with confidence tie-breaking for binary authentication.
    probs_args: list/tuple of (N,) arrays, or multiple (N,) arrays passed as arguments.
    """
    if len(probs_args) == 1 and isinstance(probs_args[0], (list, tuple)):
        probs_list = probs_args[0]
    else:
        probs_list = list(probs_args)

    n_mods = len(probs_list)
    if n_mods == 1:
        return (probs_list[0] >= 0.5).astype(int)

    binary_preds = np.array([(p >= 0.5).astype(int) for p in probs_list])  # (M, N)
    confs = np.array([np.abs(p - 0.5) for p in probs_list])                # (M, N)

    votes = binary_preds.sum(axis=0)                                       # (N,)
    half_mods = n_mods / 2.0

    final_preds = np.zeros(len(votes), dtype=int)
    pos_mask = votes > half_mods
    neg_mask = votes < half_mods
    final_preds[pos_mask] = 1
    final_preds[neg_mask] = 0

    tie_mask = (votes == half_mods)
    if np.any(tie_mask):
        best_mod_idx = np.argmax(confs[:, tie_mask], axis=0)
        final_preds[tie_mask] = binary_preds[best_mod_idx, tie_mask]

    return final_preds


def search_soft_vote_weights_id(val_probs, val_labels):
    """
    Validation grid search for optimal modality weights in multi-class identification.
    """
    probs_list = list(val_probs)
    n_mods = len(probs_list)
    if n_mods == 1:
        return (1.0,)

    best_acc = -1.0
    best_weights = tuple([round(1.0 / n_mods, 4)] * n_mods)
    steps = int(round(1.0 / GRID_STEP)) + 1

    if n_mods == 2:
        for i in range(steps):
            w1 = round(i * GRID_STEP, 4)
            w2 = round(1.0 - w1, 4)
            p_soft = w1 * probs_list[0] + w2 * probs_list[1]
            preds = np.argmax(p_soft, axis=1)
            acc = float(np.mean(preds == val_labels))

            if acc > best_acc:
                best_acc = acc
                best_weights = (w1, w2)
    elif n_mods == 3:
        for i in range(steps):
            w1 = round(i * GRID_STEP, 4)
            for j in range(steps - i):
                w2 = round(j * GRID_STEP, 4)
                w3 = round(1.0 - w1 - w2, 4)
                if w3 < 0:
                    continue

                p_soft = w1 * probs_list[0] + w2 * probs_list[1] + w3 * probs_list[2]
                preds = np.argmax(p_soft, axis=1)
                acc = float(np.mean(preds == val_labels))

                if acc > best_acc:
                    best_acc = acc
                    best_weights = (w1, w2, w3)
    else:
        best_weights = tuple([round(1.0 / n_mods, 4)] * n_mods)

    return best_weights


def search_soft_vote_weights_auth(val_probs, val_labels):
    """
    Validation grid search for optimal modality weights in binary authentication.
    """
    probs_list = list(val_probs)
    n_mods = len(probs_list)
    if n_mods == 1:
        return (1.0,)

    best_auc = -1.0
    best_weights = tuple([round(1.0 / n_mods, 4)] * n_mods)
    steps = int(round(1.0 / GRID_STEP)) + 1

    if n_mods == 2:
        for i in range(steps):
            w1 = round(i * GRID_STEP, 4)
            w2 = round(1.0 - w1, 4)
            p_soft = w1 * probs_list[0] + w2 * probs_list[1]
            try:
                auc = roc_auc_score(val_labels, p_soft) if len(np.unique(val_labels)) > 1 else 0.0
            except Exception:
                auc = 0.0

            if auc > best_auc:
                best_auc = auc
                best_weights = (w1, w2)
    elif n_mods == 3:
        for i in range(steps):
            w1 = round(i * GRID_STEP, 4)
            for j in range(steps - i):
                w2 = round(j * GRID_STEP, 4)
                w3 = round(1.0 - w1 - w2, 4)
                if w3 < 0:
                    continue

                p_soft = w1 * probs_list[0] + w2 * probs_list[1] + w3 * probs_list[2]
                try:
                    auc = roc_auc_score(val_labels, p_soft) if len(np.unique(val_labels)) > 1 else 0.0
                except Exception:
                    auc = 0.0

                if auc > best_auc:
                    best_auc = auc
                    best_weights = (w1, w2, w3)
    else:
        best_weights = tuple([round(1.0 / n_mods, 4)] * n_mods)

    return best_weights


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f" Running ENSEMBLE PIPELINE for TASK: {TASK.upper()}")
    print(f" Modalities: {MODALITIES}")
    print(f" Device: {device}")
    print(f"==================================================")

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = cfg["path"]["result"]

    if TASK == "identification":
        run_folder = f"ensemble_id_{ARCH}_{time_str}"
    else:
        run_folder = f"ensemble_auth_{TARGET}_{ARCH}_{time_str}"

    plot_dir = os.path.join(exp_dir, run_folder)
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Load manifests for all modalities
    manifests = {}
    for mod in MODALITIES:
        if TASK == "identification":
            manifest_file = os.path.join(OUTPUT_DIR, f"identification_{mod}_folds.json")
        else:
            manifest_file = os.path.join(OUTPUT_DIR, f"{TARGET}_{mod}_folds.json")

        if not os.path.exists(manifest_file):
            print(f"[ERROR] Manifest file not found: {manifest_file}. Please run dataset freeze step first.")
            return

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifests[mod] = json.load(f)

    # 2. Resolve fold keys
    fold_keys = sorted(
        [k for k in manifests[MODALITIES[0]].keys() if k.startswith("fold_")],
        key=lambda x: int(x.split("_")[1])
    )

    results_by_mod = {mod: [] for mod in MODALITIES}
    results_hard = []
    results_soft = []
    results_meta = []
    soft_weights_history = []
    meta_weights_history = []

    if TASK == "identification":
        class_mapping = manifests[MODALITIES[0]]["class_mapping"]
        class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]
        num_classes = len(class_names)

        # Load player groups (pro_junior vs top_20)
        player_groups = load_player_groups()
        player_to_group = {}
        for grp, p_list in player_groups.items():
            for p in p_list:
                player_to_group[p] = grp

        # Save player_groups snapshot to plot_dir for experiment auditability
        with open(os.path.join(plot_dir, "player_groups.json"), "w", encoding="utf-8") as f:
            json.dump(player_groups, f, indent=4, ensure_ascii=False)

        for fold_key in fold_keys:
            print(f"\n==================== {fold_key.upper()} ====================")

            val_probs_dict = {}
            test_probs_dict = {}
            test_gt = None
            val_gt = None

            # Train each modality model sequentially
            for mod in MODALITIES:
                print(f"\n>>> Training Modality Model: {mod.upper()}")
                fold_data = manifests[mod][fold_key]
                res, val_probs, val_labels, test_probs, test_labels = train_fold_id(
                    fold_key, fold_data, device, os.path.join(plot_dir, mod), class_names, return_preds=True
                )
                mod_preds = np.argmax(test_probs, axis=1)
                mod_top3_acc = compute_topk_acc(test_probs, test_labels, k=3)
                mod_group_acc = compute_group_acc(mod_preds, test_labels, class_names, player_to_group, player_groups)
                res["test"]["top3_acc"] = mod_top3_acc
                res["test"]["group_acc"] = mod_group_acc
                results_by_mod[mod].append(res)
                print(f"[{mod.upper()}] {fold_key} -> Test Acc: {res['test']['acc']:.4f}, Top-3: {mod_top3_acc:.4f}, Group Acc: {mod_group_acc:.4f}, Macro-F1: {res['test']['f1']:.4f}")

                val_probs_dict[mod] = val_probs
                test_probs_dict[mod] = test_probs
                test_gt = test_labels
                val_gt = val_labels

            # --- HARD VOTE ---
            hard_preds = hard_vote_id([test_probs_dict[mod] for mod in MODALITIES])
            hard_acc = float(np.mean(hard_preds == test_gt))
            hard_f1 = float(f1_score(test_gt, hard_preds, average='macro'))
            hard_top3_acc = None  # 方案 A: Hard Vote 不具有连续概率输出，Top-3 记为 None
            hard_group_acc = compute_group_acc(hard_preds, test_gt, class_names, player_to_group, player_groups)
            results_hard.append({"acc": hard_acc, "f1": hard_f1, "top3_acc": hard_top3_acc, "group_acc": hard_group_acc})
            print(f"\n[HARD VOTE] {fold_key} -> Test Acc: {hard_acc:.4f}, Top-3: N/A, Group Acc: {hard_group_acc:.4f}, Macro-F1: {hard_f1:.4f}")

            # --- SOFT VOTE ---
            val_probs_list = [val_probs_dict[mod] for mod in MODALITIES]
            soft_weights = search_soft_vote_weights_id(val_probs_list, val_gt)
            fold_weights_dict = {"fold": fold_key}
            for mod, w in zip(MODALITIES, soft_weights):
                fold_weights_dict[f"w_{mod}"] = w
            soft_weights_history.append(fold_weights_dict)

            test_soft_probs = sum(
                w * test_probs_dict[mod]
                for w, mod in zip(soft_weights, MODALITIES)
            )
            soft_preds = np.argmax(test_soft_probs, axis=1)
            soft_acc = float(np.mean(soft_preds == test_gt))
            soft_f1 = float(f1_score(test_gt, soft_preds, average='macro'))
            soft_top3_acc = compute_topk_acc(test_soft_probs, test_gt, k=3)
            soft_group_acc = compute_group_acc(soft_preds, test_gt, class_names, player_to_group, player_groups)
            results_soft.append({"acc": soft_acc, "f1": soft_f1, "top3_acc": soft_top3_acc, "group_acc": soft_group_acc})
            weights_str = ", ".join([f"{mod.upper()}={w:.2f}" for mod, w in zip(MODALITIES, soft_weights)])
            print(f"[SOFT VOTE] {fold_key} (Weights: {weights_str}) -> Test Acc: {soft_acc:.4f}, Top-3: {soft_top3_acc:.4f}, Group Acc: {soft_group_acc:.4f}, Macro-F1: {soft_f1:.4f}")

            # --- META-CLASSIFIER (Stacking Logistic Regression) ---
            # 动态拼接所有启用模态的全类别 Softmax 概率矩阵为 (len(MODALITIES) * C) 维元特征
            X_val_meta = np.concatenate([
                val_probs_dict[mod] for mod in MODALITIES
            ], axis=1)

            X_test_meta = np.concatenate([
                test_probs_dict[mod] for mod in MODALITIES
            ], axis=1)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                meta_clf = LogisticRegressionCV(
                    cv=5,
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42
                )
                meta_clf.fit(X_val_meta, val_gt)

            meta_preds = meta_clf.predict(X_test_meta)
            meta_probs = meta_clf.predict_proba(X_test_meta)
            meta_acc = float(np.mean(meta_preds == test_gt))
            meta_f1 = float(f1_score(test_gt, meta_preds, average='macro'))
            meta_top3_acc = compute_topk_acc(meta_probs, test_gt, k=3)
            meta_group_acc = compute_group_acc(meta_preds, test_gt, class_names, player_to_group, player_groups)
            results_meta.append({"acc": meta_acc, "f1": meta_f1, "top3_acc": meta_top3_acc, "group_acc": meta_group_acc})

            # 解析系数矩阵与模态对角权重贡献
            coef_matrix = meta_clf.coef_  # (C, len(MODALITIES) * C)
            mod_diags = {}
            for idx, mod in enumerate(MODALITIES):
                start_col = idx * num_classes
                mod_diags[mod] = [float(coef_matrix[c, start_col + c]) for c in range(num_classes)]

            best_C_val = meta_clf.C_.tolist() if hasattr(meta_clf.C_, "tolist") else [float(c) for c in meta_clf.C_]
            meta_weights_history.append({
                "fold": fold_key,
                "best_C": best_C_val,
                "avg_diag_weights": {
                    mod: float(np.mean(mod_diags[mod])) for mod in MODALITIES
                },
                "player_diag_weights": {
                    class_names[c]: {
                        mod: mod_diags[mod][c] for mod in MODALITIES
                    } for c in range(num_classes)
                },
                "coef": coef_matrix.tolist(),
                "intercept": meta_clf.intercept_.tolist()
            })
            print(f"[META-CLASSIFIER] {fold_key} (Best C mean={float(np.mean(best_C_val)):.4f}) -> Test Acc: {meta_acc:.4f}, Top-3: {meta_top3_acc:.4f}, Group Acc: {meta_group_acc:.4f}, Macro-F1: {meta_f1:.4f}")

        # Summary Metrics
        summary = {}
        for mod in MODALITIES:
            accs = [r["test"]["acc"] for r in results_by_mod[mod]]
            f1s = [r["test"]["f1"] for r in results_by_mod[mod]]
            top3s = [r["test"]["top3_acc"] for r in results_by_mod[mod]]
            groups = [r["test"]["group_acc"] for r in results_by_mod[mod]]
            summary[f"{mod}_only"] = {
                "avg_acc": float(np.mean(accs)),
                "avg_f1": float(np.mean(f1s)),
                "avg_top3_acc": float(np.mean(top3s)),
                "avg_group_acc": float(np.mean(groups))
            }

        hard_accs = [r["acc"] for r in results_hard]
        hard_f1s = [r["f1"] for r in results_hard]
        hard_groups = [r["group_acc"] for r in results_hard]

        soft_accs = [r["acc"] for r in results_soft]
        soft_f1s = [r["f1"] for r in results_soft]
        soft_top3s = [r["top3_acc"] for r in results_soft]
        soft_groups = [r["group_acc"] for r in results_soft]

        meta_accs = [r["acc"] for r in results_meta]
        meta_f1s = [r["f1"] for r in results_meta]
        meta_top3s = [r["top3_acc"] for r in results_meta]
        meta_groups = [r["group_acc"] for r in results_meta]

        summary["hard_vote"] = {
            "avg_acc": float(np.mean(hard_accs)),
            "avg_f1": float(np.mean(hard_f1s)),
            "avg_top3_acc": None,
            "avg_group_acc": float(np.mean(hard_groups))
        }
        summary["soft_vote"] = {
            "avg_acc": float(np.mean(soft_accs)),
            "avg_f1": float(np.mean(soft_f1s)),
            "avg_top3_acc": float(np.mean(soft_top3s)),
            "avg_group_acc": float(np.mean(soft_groups)),
            "weights": soft_weights_history
        }
        summary["meta_classifier"] = {
            "avg_acc": float(np.mean(meta_accs)),
            "avg_f1": float(np.mean(meta_f1s)),
            "avg_top3_acc": float(np.mean(meta_top3s)),
            "avg_group_acc": float(np.mean(meta_groups)),
            "meta_details": meta_weights_history
        }
        summary["detailed_folds"] = {
            fold_key: {
                **{f"{mod}_acc": results_by_mod[mod][i]["test"]["acc"] for mod in MODALITIES},
                **{f"{mod}_f1": results_by_mod[mod][i]["test"]["f1"] for mod in MODALITIES},
                **{f"{mod}_top3_acc": results_by_mod[mod][i]["test"]["top3_acc"] for mod in MODALITIES},
                **{f"{mod}_group_acc": results_by_mod[mod][i]["test"]["group_acc"] for mod in MODALITIES},
                "hard_vote_acc": results_hard[i]["acc"],
                "hard_vote_f1": results_hard[i]["f1"],
                "hard_vote_top3_acc": None,
                "hard_vote_group_acc": results_hard[i]["group_acc"],
                "soft_vote_acc": results_soft[i]["acc"],
                "soft_vote_f1": results_soft[i]["f1"],
                "soft_vote_top3_acc": results_soft[i]["top3_acc"],
                "soft_vote_group_acc": results_soft[i]["group_acc"],
                "meta_classifier_acc": results_meta[i]["acc"],
                "meta_classifier_f1": results_meta[i]["f1"],
                "meta_classifier_top3_acc": results_meta[i]["top3_acc"],
                "meta_classifier_group_acc": results_meta[i]["group_acc"],
                "soft_weights": soft_weights_history[i],
                "meta_summary": {
                    "best_C": meta_weights_history[i]["best_C"],
                    "avg_diag_weights": meta_weights_history[i]["avg_diag_weights"]
                }
            }
            for i, fold_key in enumerate(fold_keys)
        }
        summary["player_groups"] = {
            "pro_junior": player_groups.get("pro_junior", []),
            "top_20": player_groups.get("top_20", [])
        }

        # Plot Comparison
        plt.figure(figsize=(16, 6))
        models = [f"{mod}_only" for mod in MODALITIES] + ["hard_vote", "soft_vote", "meta_classifier"]
        avg_accs = [summary[m]["avg_acc"] for m in models]
        avg_top3 = [summary[m]["avg_top3_acc"] if summary[m]["avg_top3_acc"] is not None else 0.0 for m in models]
        avg_group = [summary[m]["avg_group_acc"] for m in models]
        avg_f1s = [summary[m]["avg_f1"] for m in models]

        x = np.arange(len(models))
        width = 0.2
        plt.bar(x - 1.5 * width, avg_accs, width, label='Avg Accuracy (Top-1)', color='forestgreen')
        plt.bar(x - 0.5 * width, avg_top3, width, label='Avg Top-3 Acc', color='goldenrod')
        plt.bar(x + 0.5 * width, avg_group, width, label='Avg Group Acc', color='darkorange')
        plt.bar(x + 1.5 * width, avg_f1s, width, label='Avg Macro-F1', color='royalblue')
        plt.ylabel('Score')
        plt.title('Identification Multi-Modal Ensemble Comparison (Top-1, Top-3, Group Acc, Macro-F1)')
        plt.xticks(x, [m.replace("_", " ").title() for m in models], rotation=15)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "summary_ensemble_comparison.png"), dpi=300)
        plt.close()

    else:
        # Binary Authentication Ensemble
        for fold_key in fold_keys:
            print(f"\n==================== {fold_key.upper()} ====================")

            val_probs_dict = {}
            test_probs_dict = {}
            test_gt = None
            val_gt = None

            for mod in MODALITIES:
                print(f"\n>>> Training Modality Model: {mod.upper()}")
                fold_data = manifests[mod][fold_key]
                res, val_pr, val_labels, test_pr, test_labels = train_fold(
                    fold_key, fold_data, device, os.path.join(plot_dir, mod), return_preds=True
                )
                results_by_mod[mod].append(res)

                val_probs_dict[mod] = val_pr
                test_probs_dict[mod] = test_pr
                test_gt = test_labels
                val_gt = val_labels

            # --- HARD VOTE ---
            hard_pred_binary = hard_vote_auth([test_probs_dict[mod] for mod in MODALITIES])
            hard_prob_blend = np.mean([test_probs_dict[mod] for mod in MODALITIES], axis=0)
            hard_acc = float(np.mean(hard_pred_binary == test_gt))
            hard_auc = float(roc_auc_score(test_gt, hard_prob_blend)) if len(np.unique(test_gt)) > 1 else 0.0
            hard_eer = compute_eer(test_gt, hard_prob_blend)
            results_hard.append({"acc": hard_acc, "auc": hard_auc, "eer": hard_eer})
            print(f"\n[HARD VOTE] {fold_key} -> Test AUC: {hard_auc:.4f}, EER: {hard_eer:.4f}")

            # --- SOFT VOTE ---
            val_probs_list = [val_probs_dict[mod] for mod in MODALITIES]
            soft_weights = search_soft_vote_weights_auth(val_probs_list, val_gt)
            fold_weights_dict = {"fold": fold_key}
            for mod, w in zip(MODALITIES, soft_weights):
                fold_weights_dict[f"w_{mod}"] = w
            soft_weights_history.append(fold_weights_dict)

            test_soft_probs = sum(
                w * test_probs_dict[mod]
                for w, mod in zip(soft_weights, MODALITIES)
            )
            soft_pred_binary = (test_soft_probs >= 0.5)
            soft_acc = float(np.mean(soft_pred_binary == test_gt))
            soft_auc = float(roc_auc_score(test_gt, test_soft_probs)) if len(np.unique(test_gt)) > 1 else 0.0
            soft_eer = compute_eer(test_gt, test_soft_probs)
            results_soft.append({"acc": soft_acc, "auc": soft_auc, "eer": soft_eer})
            weights_str = ", ".join([f"{mod.upper()}={w:.2f}" for mod, w in zip(MODALITIES, soft_weights)])
            print(f"[SOFT VOTE] {fold_key} (Weights: {weights_str}) -> Test AUC: {soft_auc:.4f}, EER: {soft_eer:.4f}")

        # Summary Metrics
        summary = {}
        for mod in MODALITIES:
            aucs = [r["test"]["auc"] for r in results_by_mod[mod]]
            eers = [r["test"]["eer"] for r in results_by_mod[mod]]
            summary[f"{mod}_only"] = {"avg_auc": float(np.mean(aucs)), "avg_eer": float(np.mean(eers))}

        hard_aucs = [r["auc"] for r in results_hard]
        hard_eers = [r["eer"] for r in results_hard]
        soft_aucs = [r["auc"] for r in results_soft]
        soft_eers = [r["eer"] for r in results_soft]

        summary["hard_vote"] = {"avg_auc": float(np.mean(hard_aucs)), "avg_eer": float(np.mean(hard_eers))}
        summary["soft_vote"] = {"avg_auc": float(np.mean(soft_aucs)), "avg_eer": float(np.mean(soft_eers)), "weights": soft_weights_history}
        summary["detailed_folds"] = {
            fold_key: {
                **{f"{mod}_auc": results_by_mod[mod][i]["test"]["auc"] for mod in MODALITIES},
                **{f"{mod}_eer": results_by_mod[mod][i]["test"]["eer"] for mod in MODALITIES},
                "hard_vote_auc": results_hard[i]["auc"],
                "hard_vote_eer": results_hard[i]["eer"],
                "soft_vote_auc": results_soft[i]["auc"],
                "soft_vote_eer": results_soft[i]["eer"],
                "soft_weights": soft_weights_history[i]
            }
            for i, fold_key in enumerate(fold_keys)
        }

        # Plot Comparison
        plt.figure(figsize=(12, 6))
        models = [f"{mod}_only" for mod in MODALITIES] + ["hard_vote", "soft_vote"]
        avg_aucs = [summary[m]["avg_auc"] for m in models]
        avg_eers = [summary[m]["avg_eer"] for m in models]

        x = np.arange(len(models))
        width = 0.35
        plt.bar(x - width/2, avg_aucs, width, label='Avg AUC', color='forestgreen')
        plt.bar(x + width/2, avg_eers, width, label='Avg EER', color='crimson')
        plt.ylabel('Score')
        plt.title('Authentication Ensemble vs Single Modality Performance')
        plt.xticks(x, models, rotation=15)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "summary_ensemble_comparison.png"), dpi=300)
        plt.close()

    # Save summary json
    perf_path = os.path.join(plot_dir, "performance_ensemble.json")
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\n==================================================")
    print(f" ENSEMBLE PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" Results saved to: {plot_dir}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
