"""
evaluate.py  –  Multi-protocol, multi-checkpoint evaluation
=============================================================
Evaluates all checkpoints in a folder for Protocol 2.1 and/or 2.2,
then reports per-checkpoint metrics AND mean ± std across all runs.

Usage
-----
Single checkpoint (original behaviour):
    python evaluate.py --config config.yaml --checkpoint checkpoints_p2.2/best_model.pth

Whole checkpoint folder, one protocol:
    python evaluate.py --config config.yaml --ckpt_dir checkpoints_J_p2.2 --protocol 2.2

Both protocols at once (recommended):
    python evaluate.py --config config.yaml ^
        --ckpt_dir_p21 "C:/…/checkpoints_J_p2.1" ^
        --ckpt_dir_p22 "C:/…/checkpoints_J_p2.2"

The config must contain:
    p21_test_txt, p21_dev_txt   for Protocol 2.1
    p22_test_txt, p22_dev_txt   for Protocol 2.2
    (falls back to test_txt / dev_txt if protocol-specific keys absent)
"""

import os
import glob
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from model   import UniAttackDetection
from dataset import UniAttackDataset, get_transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from metrics import evaluate, evaluate_with_eer_threshold


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True)

    # single-checkpoint (original)
    p.add_argument("--checkpoint",   default=None)
    p.add_argument("--split",        default="test", choices=["dev","test"])

    # folder – single protocol
    p.add_argument("--ckpt_dir",     default=None)
    p.add_argument("--protocol",     default=None, choices=["2.1","2.2"])

    # folder – both protocols
    p.add_argument("--ckpt_dir_p21", default=None,
                   help="Checkpoint folder for Protocol 2.1")
    p.add_argument("--ckpt_dir_p22", default=None,
                   help="Checkpoint folder for Protocol 2.2")

    p.add_argument("--split_multi",  default="test", choices=["dev","test"])
    p.add_argument("--ckpt_glob",    default="*.pth",
                   help="Glob pattern inside checkpoint folder (default: *.pth)")
    p.add_argument("--save_csv",     default=None)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────
def build_model(cfg, device):
    return UniAttackDetection(
        clip_model_name       = cfg.get("clip_model", "ViT-B/16"),
        num_student_tokens    = cfg.get("num_student_tokens", 16),
        num_teacher_templates = cfg.get("num_teacher_templates", 6),
        lam                   = cfg.get("lambda_ufm", 1.0),
        device                = device,
    ).to(device)


def load_checkpoint(model, ckpt_path, device):
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    epoch    = ckpt.get("epoch", "?")
    best_auc = ckpt.get("best_auc",  None)
    best_acer= ckpt.get("best_acer", None)
    dev_thr  = float(ckpt.get("dev_acer_threshold",
                     ckpt.get("threshold", 0.5)))
    tag = f"epoch={epoch}"
    if best_auc  is not None: tag += f"  AUC={float(best_auc):.4f}%"
    if best_acer is not None: tag += f"  ACER={float(best_acer):.4f}%"
    return model, dev_thr, tag


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
def run_inference(model, txt_path, data_root, cfg, device):
    ds = UniAttackDataset(txt_path, data_root, get_transforms(False))
    loader = DataLoader(ds,
                        batch_size  = cfg.get("batch_size", 32),
                        shuffle     = False,
                        num_workers = cfg.get("num_workers", 4))
    all_labels, all_scores = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Inference", leave=False):
            logits, _ = model(imgs.to(device))
            probs = torch.softmax(logits, -1)[:, 1].cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_scores)


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATE ONE CHECKPOINT
# ──────────────────────────────────────────────────────────────────────────────
def eval_one(model, ckpt_path, txt_path, data_root, cfg, device):
    model, dev_thr, tag = load_checkpoint(model, ckpt_path, device)
    y_true, y_score     = run_inference(model, txt_path, data_root, cfg, device)
    m_dev = evaluate(y_true, y_score, threshold=dev_thr)
    m_eer = evaluate_with_eer_threshold(y_true, y_score)
    return {
        "ckpt":    os.path.basename(ckpt_path),
        "tag":     tag,
        "dev_thr": dev_thr,
        "eer_thr": m_eer["threshold"],
        "dev":     m_dev,
        "eer":     m_eer,
        "y_true":  y_true,    # kept for confusion matrix
        "y_score": y_score,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────────────────────────────────────
METRIC_KEYS = ["ACER", "ACC", "AUC", "EER", "APCER", "BPCER"]


def compute_stats(results, variant="dev"):
    collected = defaultdict(list)
    for r in results:
        for k in METRIC_KEYS:
            collected[k].append(r[variant][k])
    stats = {}
    for k, vals in collected.items():
        arr = np.array(vals, dtype=float)
        stats[k] = (float(arr.mean()), float(arr.std(ddof=0)), arr.tolist())
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def sep(c="─", w=72): print(c * w)


def print_confusion_matrix(y_true, y_score, threshold, label=""):
    """Print a labelled 2x2 confusion matrix for a given threshold."""
    preds = (y_score >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)
    # guard against shapes smaller than 2x2 (degenerate splits)
    tn = cm[0, 0] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    tp = cm[1, 1] if cm.shape == (2, 2) else 0
    total = tn + fp + fn + tp
    if label:
        print(f"  Confusion Matrix  [{label}]  threshold={threshold:.4f}")
    else:
        print(f"  Confusion Matrix  threshold={threshold:.4f}")
    print(f"                       Pred: Live   Pred: Spoof")
    print(f"    True: Live   {tn:>12}   {fp:>11}   (BPCER = {fp}/{tn+fp} = {fp/max(tn+fp,1)*100:.2f}%)")
    print(f"    True: Spoof  {fn:>12}   {tp:>11}   (APCER = {fn}/{fn+tp} = {fn/max(fn+tp,1)*100:.2f}%)")
    print(f"    Total: {total}   Correct: {tn+tp} ({(tn+tp)/max(total,1)*100:.2f}%)")


def print_one_result(r, variant="dev"):
    m   = r[variant]
    thr = f"dev_thr={r['dev_thr']:.4f}" if variant == "dev" \
          else f"eer_thr={r['eer_thr']:.4f}"
    print(f"    [{thr}]  "
          f"ACER={m['ACER']:.2f}%  ACC={m['ACC']:.2f}%  "
          f"AUC={m['AUC']:.2f}%  EER={m['EER']:.2f}%  "
          f"APCER={m['APCER']:.2f}%  BPCER={m['BPCER']:.2f}%")


def print_stats_table(stats, label=""):
    if label:
        print(f"\n  {label}")
    sep("·", 72)
    print(f"  {'Metric':<8}  {'Mean':>9}  {'Std':>9}  {'Min':>9}  {'Max':>9}  per-run values")
    sep("·", 72)
    for k in METRIC_KEYS:
        mean, std, vals = stats[k]
        vstr = "  ".join(f"{v:.2f}" for v in vals)
        print(f"  {k:<8}  {mean:>8.4f}%  {std:>8.4f}%  "
              f"{min(vals):>8.4f}%  {max(vals):>8.4f}%  [{vstr}]")
    sep("·", 72)
    # Bold one-liner summary
    print("  " + "  |  ".join(
        f"{k} = {stats[k][0]:.2f} ± {stats[k][1]:.2f}%" for k in METRIC_KEYS
    ))


def print_protocol_results(proto_label, results):
    sep("═")
    print(f"  PROTOCOL {proto_label}  –  {len(results)} checkpoint(s)")
    sep("═")

    for i, r in enumerate(results, 1):
        print(f"\n  Run {i}/{len(results)}  {r['ckpt']}  ({r['tag']})")
        print("    Dev-ACER threshold  :", end=""); print_one_result(r, "dev")
        print("    EER threshold       :", end=""); print_one_result(r, "eer")
        print()
        print_confusion_matrix(r["y_true"], r["y_score"],
                               r["dev_thr"], "Dev-ACER threshold")
        print()
        print_confusion_matrix(r["y_true"], r["y_score"],
                               r["eer_thr"], "EER threshold")

    if len(results) > 1:
        for variant, label in [("dev", "Dev-ACER threshold"),
                                ("eer", "EER threshold (cross-domain)")]:
            stats = compute_stats(results, variant)
            print_stats_table(stats,
                f"mean ± std  [{label}]  N={len(results)} runs")
        # Aggregate confusion matrix: pool all y_true / y_score across runs
        print()
        sep()
        print("  AGGREGATE CONFUSION MATRICES  (all runs pooled)")
        sep()
        y_true_all  = np.concatenate([r["y_true"]  for r in results])
        y_score_all = np.concatenate([r["y_score"] for r in results])
        # Use mean thresholds across runs
        mean_dev_thr = float(np.mean([r["dev_thr"] for r in results]))
        mean_eer_thr = float(np.mean([r["eer_thr"] for r in results]))
        print()
        print_confusion_matrix(y_true_all, y_score_all,
                               mean_dev_thr, f"Dev-ACER threshold (mean={mean_dev_thr:.4f})")
        print()
        print_confusion_matrix(y_true_all, y_score_all,
                               mean_eer_thr, f"EER threshold (mean={mean_eer_thr:.4f})")
    else:
        print("\n  (Only 1 checkpoint — no mean±std.)")


# ──────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ──────────────────────────────────────────────────────────────────────────────
def save_csv(all_results_by_protocol, csv_path):
    import csv
    rows = []
    for proto, results in all_results_by_protocol.items():
        for r in results:
            for variant in ("dev", "eer"):
                row = {"protocol": proto,
                       "checkpoint": r["ckpt"],
                       "threshold_type": variant,
                       "threshold": r[f"{variant}_thr"]}
                row.update({k: r[variant][k] for k in METRIC_KEYS})
                rows.append(row)
    if not rows: return
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[INFO] CSV saved → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def get_txt(cfg, protocol, split):
    """Return test/dev txt path for the given protocol from config."""
    if protocol == "2.1":
        key_test, key_dev = "p21_test_txt", "p21_dev_txt"
    else:
        key_test, key_dev = "p22_test_txt", "p22_dev_txt"
    if split == "test":
        return cfg.get(key_test, cfg.get("test_txt"))
    return cfg.get(key_dev,  cfg.get("dev_txt"))


def collect_checkpoints(folder, pattern):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(folder, "**", pattern),
                                 recursive=True))
    return paths


def run_protocol(proto, folder, split, model, cfg, device, ckpt_glob):
    txt = get_txt(cfg, proto, split)
    if txt is None:
        print(f"[ERROR] No txt path for Protocol {proto} in config — skipping.")
        return []

    ckpts = collect_checkpoints(folder, ckpt_glob)
    if not ckpts:
        print(f"[WARN] No checkpoints in {folder} — skipping P{proto}.")
        return []

    data_root = cfg["data_root"]
    print(f"\n[INFO] Protocol {proto}  |  {len(ckpts)} checkpoint(s)")
    print(f"       Folder : {folder}")
    print(f"       Split  : {split}  →  {txt}")

    results = []
    for ckpt_path in ckpts:
        print(f"\n  Checkpoint: {os.path.basename(ckpt_path)}")
        r = eval_one(model, ckpt_path, txt, data_root, cfg, device)
        results.append(r)
        print_one_result(r, "dev")
        print_one_result(r, "eer")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    cfg    = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model       = build_model(cfg, device)
    all_results = {}   # protocol → list[result_dict]

    # ── MODE A: single checkpoint ─────────────────────────────────────────────
    if args.checkpoint:
        proto = "2.1" if ("2.1" in args.checkpoint or "p21" in args.checkpoint.lower()) else "2.2"
        txt   = get_txt(cfg, proto, args.split)
        if txt is None:
            print(f"[ERROR] No txt for P{proto} in config."); return
        print(f"\n[INFO] Single checkpoint | Protocol {proto} | split={args.split}")
        r = eval_one(model, args.checkpoint, txt, cfg["data_root"], cfg, device)
        all_results[proto] = [r]

    # ── MODE B: single folder + explicit protocol ─────────────────────────────
    elif args.ckpt_dir and args.protocol:
        results = run_protocol(args.protocol, args.ckpt_dir,
                               args.split_multi, model, cfg, device, args.ckpt_glob)
        all_results[args.protocol] = results

    # ── MODE C: both protocol folders (default / main use-case) ──────────────
    else:
        # Build directory map (CLI args take priority, then hardcoded defaults)
        base = r"C:\Users\bhask\Documents\Project\UniAttack"
        dirs = {
            "2.1": args.ckpt_dir_p21 or os.path.join(base, "checkpoints_J_p2.1"),
            "2.2": args.ckpt_dir_p22 or os.path.join(base, "checkpoints_J_p2.2"),
        }

        for proto, folder in dirs.items():
            if not os.path.isdir(folder):
                print(f"[WARN] Folder not found, skipping P{proto}: {folder}")
                continue
            results = run_protocol(proto, folder, args.split_multi,
                                   model, cfg, device, args.ckpt_glob)
            all_results[proto] = results

    # ── PRINT CONSOLIDATED RESULTS ────────────────────────────────────────────
    print("\n\n" + "═"*72)
    print("  CONSOLIDATED RESULTS")
    print("═"*72)

    for proto in sorted(all_results.keys()):
        results = all_results[proto]
        if not results:
            print(f"\n  Protocol {proto}: no results.")
            continue
        print_protocol_results(proto, results)

    # ── Cross-protocol mean ± std ─────────────────────────────────────────────
    non_empty = {p: r for p, r in all_results.items() if r}
    if len(non_empty) == 2:
        all_flat = [r for results in non_empty.values() for r in results]
        print()
        sep("═")
        print(f"  CROSS-PROTOCOL AGGREGATE  (P2.1 + P2.2,  N={len(all_flat)} total checkpoints)")
        sep("═")
        for variant, label in [("dev", "Dev-ACER threshold"),
                                ("eer", "EER threshold (cross-domain)")]:
            stats = compute_stats(all_flat, variant)
            print_stats_table(stats, f"[{label}]")

    # ── CSV ───────────────────────────────────────────────────────────────────
    if args.save_csv:
        save_csv(all_results, args.save_csv)

    print()


if __name__ == "__main__":
    main()
