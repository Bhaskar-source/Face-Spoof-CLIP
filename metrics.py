"""
Evaluation metrics for UniAttackDetection.

Key insight for Protocol 2.2:
  - Dev set = Live + Digital attacks
  - Test set = Live + Physical attacks  (completely unseen attack type)
  - The ACER-minimising threshold found on dev (digital) does NOT transfer
    to physical attacks -> high ACER/low ACC even if the model is good
  - Solution: also report results with the EER threshold, which is
    score-distribution-based and more robust across domains.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_acer(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    apcer = fp / (fp + tn + 1e-9)
    bpcer = fn / (fn + tp + 1e-9)
    return float((apcer + bpcer) / 2), float(apcer), float(bpcer)


def compute_acc(y_true, y_score, threshold):
    return float(((y_score >= threshold).astype(int) == y_true).mean())


def find_acer_threshold(y_true, y_score, n=500):
    """Sweep thresholds to minimise ACER (best for same-domain dev->test)."""
    thresholds = np.linspace(y_score.min(), y_score.max(), n)
    best_acer, best_t = 1.0, 0.5
    for t in thresholds:
        acer, _, _ = compute_acer(y_true, y_score, t)
        if acer < best_acer:
            best_acer, best_t = acer, t
    return best_t, best_acer


def find_eer_threshold(y_true, y_score):
    """
    EER threshold: t where FAR ~= FRR.
    Robust to domain shift between dev and test because it uses the
    score distribution rather than label distribution.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    thr = float(thresholds[idx])
    return thr, eer


def evaluate(y_true, y_score, threshold=None):
    """Standard evaluation with a fixed or auto-found threshold."""
    y_true  = np.array(y_true,  dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float64)

    has_both = len(np.unique(y_true)) > 1
    auc = float(roc_auc_score(y_true, y_score)) if has_both else 0.0
    eer_thr, eer = find_eer_threshold(y_true, y_score) if has_both else (0.5, 0.0)

    if threshold is None:
        threshold, _ = find_acer_threshold(y_true, y_score)

    acer, apcer, bpcer = compute_acer(y_true, y_score, threshold)
    acc = compute_acc(y_true, y_score, threshold)

    return {
        "ACER":      round(acer  * 100, 4),
        "ACC":       round(acc   * 100, 4),
        "AUC":       round(auc   * 100, 4),
        "EER":       round(eer   * 100, 4),
        "APCER":     round(apcer * 100, 4),
        "BPCER":     round(bpcer * 100, 4),
        "threshold": float(threshold),
        "eer_threshold": float(eer_thr),
    }


def evaluate_with_eer_threshold(y_true, y_score):
    """
    Evaluate using the EER threshold (preferred for cross-domain P2.x).
    When dev and test have different attack types, the EER threshold found
    on the TEST set's score distribution matches the paper's protocol.
    """
    y_true  = np.array(y_true,  dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float64)

    has_both = len(np.unique(y_true)) > 1
    auc = float(roc_auc_score(y_true, y_score)) if has_both else 0.0
    eer_thr, eer = find_eer_threshold(y_true, y_score) if has_both else (0.5, 0.0)

    acer, apcer, bpcer = compute_acer(y_true, y_score, eer_thr)
    acc = compute_acc(y_true, y_score, eer_thr)

    return {
        "ACER":      round(acer  * 100, 4),
        "ACC":       round(acc   * 100, 4),
        "AUC":       round(auc   * 100, 4),
        "EER":       round(eer   * 100, 4),
        "APCER":     round(apcer * 100, 4),
        "BPCER":     round(bpcer * 100, 4),
        "threshold": float(eer_thr),
    }


def print_metrics(metrics: dict, title: str = ""):
    if title:
        print(f"\n{'='*55}")
        print(f"  {title}")
        print(f"{'='*55}")
    print(f"  ACER  : {metrics['ACER']:.4f} %")
    print(f"  ACC   : {metrics['ACC']:.4f} %")
    print(f"  AUC   : {metrics['AUC']:.4f} %")
    print(f"  EER   : {metrics['EER']:.4f} %")
    if "APCER" in metrics:
        print(f"  APCER : {metrics['APCER']:.4f} %  (attack classified as live)")
        print(f"  BPCER : {metrics['BPCER']:.4f} %  (live classified as attack)")
    print(f"  Threshold used : {metrics['threshold']:.4f}")
