import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
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


def hard_vote_id(p_kb, p_ms, p_cb):
    """
    Hard voting with confidence tie-breaking for multi-class identification.
    p_k: (N, C) probability arrays.
    """
    pred_kb = np.argmax(p_kb, axis=1)
    pred_ms = np.argmax(p_ms, axis=1)
    pred_cb = np.argmax(p_cb, axis=1)

    conf_kb = np.max(p_kb, axis=1)
    conf_ms = np.max(p_ms, axis=1)
    conf_cb = np.max(p_cb, axis=1)

    n_samples = len(pred_kb)
    final_preds = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        k, m, c = pred_kb[i], pred_ms[i], pred_cb[i]
        if k == m or k == c:
            final_preds[i] = k
        elif m == c:
            final_preds[i] = m
        else:
            # 1:1:1 Tie -> Select prediction with highest confidence score
            confs = [conf_kb[i], conf_ms[i], conf_cb[i]]
            preds = [k, m, c]
            final_preds[i] = preds[int(np.argmax(confs))]

    return final_preds


def search_soft_vote_weights_id(val_probs, val_labels):
    """
    Validation grid search for optimal modality weights in multi-class identification.
    """
    p_kb, p_ms, p_cb = val_probs
    best_acc = -1.0
    best_weights = (1/3, 1/3, 1/3)

    steps = int(round(1.0 / GRID_STEP)) + 1
    for i in range(steps):
        w1 = round(i * GRID_STEP, 4)
        for j in range(steps - i):
            w2 = round(j * GRID_STEP, 4)
            w3 = round(1.0 - w1 - w2, 4)
            if w3 < 0:
                continue

            p_soft = w1 * p_kb + w2 * p_ms + w3 * p_cb
            preds = np.argmax(p_soft, axis=1)
            acc = float(np.mean(preds == val_labels))

            if acc > best_acc:
                best_acc = acc
                best_weights = (w1, w2, w3)

    return best_weights


def search_soft_vote_weights_auth(val_probs, val_labels):
    """
    Validation grid search for optimal modality weights in binary authentication.
    """
    p_kb, p_ms, p_cb = val_probs
    best_auc = -1.0
    best_weights = (1/3, 1/3, 1/3)

    steps = int(round(1.0 / GRID_STEP)) + 1
    for i in range(steps):
        w1 = round(i * GRID_STEP, 4)
        for j in range(steps - i):
            w2 = round(j * GRID_STEP, 4)
            w3 = round(1.0 - w1 - w2, 4)
            if w3 < 0:
                continue

            p_soft = w1 * p_kb + w2 * p_ms + w3 * p_cb
            try:
                auc = roc_auc_score(val_labels, p_soft) if len(np.unique(val_labels)) > 1 else 0.0
            except Exception:
                auc = 0.0

            if auc > best_auc:
                best_auc = auc
                best_weights = (w1, w2, w3)

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
    soft_weights_history = []

    if TASK == "identification":
        class_mapping = manifests[MODALITIES[0]]["class_mapping"]
        class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]

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
                results_by_mod[mod].append(res)

                val_probs_dict[mod] = val_probs
                test_probs_dict[mod] = test_probs
                test_gt = test_labels
                val_gt = val_labels

            # --- HARD VOTE ---
            hard_preds = hard_vote_id(
                test_probs_dict["keyboard"],
                test_probs_dict["mouse"],
                test_probs_dict["combined"]
            )
            hard_acc = float(np.mean(hard_preds == test_gt))
            hard_f1 = float(f1_score(test_gt, hard_preds, average='macro'))
            results_hard.append({"acc": hard_acc, "f1": hard_f1})
            print(f"\n[HARD VOTE] {fold_key} -> Test Acc: {hard_acc:.4f}, Macro-F1: {hard_f1:.4f}")

            # --- SOFT VOTE ---
            val_tuple = (val_probs_dict["keyboard"], val_probs_dict["mouse"], val_probs_dict["combined"])
            w_kb, w_ms, w_cb = search_soft_vote_weights_id(val_tuple, val_gt)
            soft_weights_history.append({"fold": fold_key, "w_kb": w_kb, "w_ms": w_ms, "w_cb": w_cb})

            test_soft_probs = (
                w_kb * test_probs_dict["keyboard"] +
                w_ms * test_probs_dict["mouse"] +
                w_cb * test_probs_dict["combined"]
            )
            soft_preds = np.argmax(test_soft_probs, axis=1)
            soft_acc = float(np.mean(soft_preds == test_gt))
            soft_f1 = float(f1_score(test_gt, soft_preds, average='macro'))
            results_soft.append({"acc": soft_acc, "f1": soft_f1})
            print(f"[SOFT VOTE] {fold_key} (Weights: KB={w_kb:.2f}, MS={w_ms:.2f}, CB={w_cb:.2f}) -> Test Acc: {soft_acc:.4f}, Macro-F1: {soft_f1:.4f}")

        # Summary Metrics
        summary = {}
        for mod in MODALITIES:
            accs = [r["test"]["acc"] for r in results_by_mod[mod]]
            f1s = [r["test"]["f1"] for r in results_by_mod[mod]]
            summary[f"{mod}_only"] = {"avg_acc": float(np.mean(accs)), "avg_f1": float(np.mean(f1s))}

        hard_accs = [r["acc"] for r in results_hard]
        hard_f1s = [r["f1"] for r in results_hard]
        soft_accs = [r["acc"] for r in results_soft]
        soft_f1s = [r["f1"] for r in results_soft]

        summary["hard_vote"] = {"avg_acc": float(np.mean(hard_accs)), "avg_f1": float(np.mean(hard_f1s))}
        summary["soft_vote"] = {"avg_acc": float(np.mean(soft_accs)), "avg_f1": float(np.mean(soft_f1s)), "weights": soft_weights_history}
        summary["detailed_folds"] = {
            fold_key: {
                **{f"{mod}_acc": results_by_mod[mod][i]["test"]["acc"] for mod in MODALITIES},
                **{f"{mod}_f1": results_by_mod[mod][i]["test"]["f1"] for mod in MODALITIES},
                "hard_vote_acc": results_hard[i]["acc"],
                "hard_vote_f1": results_hard[i]["f1"],
                "soft_vote_acc": results_soft[i]["acc"],
                "soft_vote_f1": results_soft[i]["f1"],
                "soft_weights": soft_weights_history[i]
            }
            for i, fold_key in enumerate(fold_keys)
        }

        # Plot Comparison
        plt.figure(figsize=(12, 6))
        models = [f"{mod}_only" for mod in MODALITIES] + ["hard_vote", "soft_vote"]
        avg_accs = [summary[m]["avg_acc"] for m in models]
        avg_f1s = [summary[m]["avg_f1"] for m in models]

        x = np.arange(len(models))
        width = 0.35
        plt.bar(x - width/2, avg_accs, width, label='Avg Accuracy', color='forestgreen')
        plt.bar(x + width/2, avg_f1s, width, label='Avg Macro-F1', color='royalblue')
        plt.ylabel('Score')
        plt.title('Identification Ensemble vs Single Modality Performance')
        plt.xticks(x, models, rotation=15)
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
            hard_pred_binary = (
                (test_probs_dict["keyboard"] >= 0.5).astype(int) +
                (test_probs_dict["mouse"] >= 0.5).astype(int) +
                (test_probs_dict["combined"] >= 0.5).astype(int)
            ) >= 2
            hard_prob_blend = (test_probs_dict["keyboard"] + test_probs_dict["mouse"] + test_probs_dict["combined"]) / 3.0
            hard_acc = float(np.mean(hard_pred_binary == test_gt))
            hard_auc = float(roc_auc_score(test_gt, hard_prob_blend)) if len(np.unique(test_gt)) > 1 else 0.0
            hard_eer = compute_eer(test_gt, hard_prob_blend)
            results_hard.append({"acc": hard_acc, "auc": hard_auc, "eer": hard_eer})
            print(f"\n[HARD VOTE] {fold_key} -> Test AUC: {hard_auc:.4f}, EER: {hard_eer:.4f}")

            # --- SOFT VOTE ---
            val_tuple = (val_probs_dict["keyboard"], val_probs_dict["mouse"], val_probs_dict["combined"])
            w_kb, w_ms, w_cb = search_soft_vote_weights_auth(val_tuple, val_gt)
            soft_weights_history.append({"fold": fold_key, "w_kb": w_kb, "w_ms": w_ms, "w_cb": w_cb})

            test_soft_probs = (
                w_kb * test_probs_dict["keyboard"] +
                w_ms * test_probs_dict["mouse"] +
                w_cb * test_probs_dict["combined"]
            )
            soft_pred_binary = (test_soft_probs >= 0.5)
            soft_acc = float(np.mean(soft_pred_binary == test_gt))
            soft_auc = float(roc_auc_score(test_gt, test_soft_probs)) if len(np.unique(test_gt)) > 1 else 0.0
            soft_eer = compute_eer(test_gt, test_soft_probs)
            results_soft.append({"acc": soft_acc, "auc": soft_auc, "eer": soft_eer})
            print(f"[SOFT VOTE] {fold_key} (Weights: KB={w_kb:.2f}, MS={w_ms:.2f}, CB={w_cb:.2f}) -> Test AUC: {soft_auc:.4f}, EER: {soft_eer:.4f}")

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
