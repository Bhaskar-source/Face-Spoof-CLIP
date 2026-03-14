"""
Training script for UniAttackDetection - Protocol 2.2

Key improvements over v1:
  1. LR warmup (10% of total steps) for stable prompt learning
  2. AUC-based model selection (AUC is threshold-independent, better for cross-domain)
  3. Dual threshold reporting at test time:
       (a) dev-set ACER threshold  -- as in paper's protocol
       (b) EER threshold           -- robust cross-domain alternative
  4. More epochs + longer cosine decay
  5. Gradient accumulation option for large effective batch sizes
"""

import os
import argparse
import yaml
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from model   import UniAttackDetection
from dataset import build_dataloaders
from metrics import evaluate, evaluate_with_eer_threshold, print_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Warmup + Cosine LR scheduler ──────────────────────────────────────────────
def get_lr(step, total_steps, lr_max, warmup_steps, lr_min=1e-6):
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ── One epoch ─────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device,
              train=True, step=0, total_steps=1,
              lr_max=1e-4, warmup_steps=0, accum_steps=1):
    model.train(train)
    ce_loss_fn = nn.CrossEntropyLoss()

    all_labels, all_scores = [], []
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (imgs, labels) in enumerate(
            tqdm(loader, desc="Train" if train else "Eval", leave=False)
        ):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            # Update LR each step during training
            if train:
                lr = get_lr(step, total_steps, lr_max, warmup_steps)
                set_lr(optimizer, lr)
                step += 1

            logits, ufm_loss, features = model(imgs)                    # loss change 
            oc_loss = model.oc_softmax_loss(features, labels)
            cls_loss = ce_loss_fn(logits, labels)
            loss = cls_loss + model.lam * ufm_loss + 0.5 * oc_loss

            if train:
                loss.backward()
                if (batch_idx + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / len(loader), np.array(all_labels), np.array(all_scores), step


# ── Main ──────────────────────────────────────────────────────────────────────
def  main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    epochs      = cfg.get("epochs", 50)
    lr_max      = cfg.get("lr", 1e-4)
    wd          = cfg.get("wd", 1e-4)
    accum_steps = cfg.get("accum_steps", 1)
    output_dir  = cfg.get("output_dir", "checkpoints_C_p2.2")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, dev_loader, test_loader = build_dataloaders(
        train_txt   = cfg["train_txt"],
        dev_txt     = cfg["dev_txt"],
        test_txt    = cfg["test_txt"],
        data_root   = cfg["data_root"],
        batch_size  = cfg.get("batch_size", 32),
        num_workers = cfg.get("num_workers", 4),
    )

    total_steps   = epochs * len(train_loader)
    warmup_steps  = int(0.1 * total_steps)   # 10% warmup
    print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UniAttackDetection(
        clip_model_name       = cfg.get("clip_model", "ViT-B/16"),
        num_student_tokens    = cfg.get("num_student_tokens", 16),
        num_teacher_templates = cfg.get("num_teacher_templates", 6),
        lam                   = cfg.get("lambda_ufm", 1.0),
        device                = device,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(trainable, lr=lr_max, weight_decay=wd)
    optimizer.zero_grad()

    os.makedirs(output_dir, exist_ok=True)

    best_auc      = 0.0
    best_ckpt     = os.path.join(output_dir, "best_model.pth")
    step          = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, _, _, step = run_epoch(
            model, train_loader, optimizer, device,
            train=True, step=step, total_steps=total_steps,
            lr_max=lr_max, warmup_steps=warmup_steps, accum_steps=accum_steps
        )

        _, dev_labels, dev_scores, _ = run_epoch(
            model, dev_loader, optimizer, device, train=False
        )
        dev_m = evaluate(dev_labels, dev_scores)   # finds best ACER threshold on dev

        current_lr = get_lr(step, total_steps, lr_max, warmup_steps)
        elapsed    = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | Loss {train_loss:.4f} | "
            f"Dev: ACER {dev_m['ACER']:.2f}% ACC {dev_m['ACC']:.2f}% "
            f"AUC {dev_m['AUC']:.2f}% EER {dev_m['EER']:.2f}% | "
            f"LR {current_lr:.2e} | {elapsed:.1f}s"
        )

        # ── Save best checkpoint based on AUC (threshold-free, cross-domain robust)
        if dev_m["AUC"] > best_auc:
            best_auc = dev_m["AUC"]
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step":            step,
                "best_auc":        best_auc,
                "dev_acer_threshold": dev_m["threshold"],
                "dev_eer_threshold":  dev_m["eer_threshold"],
                "dev_metrics":     dev_m,
            }, best_ckpt)
            print(f"  ✓ Best model saved (AUC={best_auc:.4f}%)")

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL TEST EVALUATION  –  Protocol 2.2")
    print("="*65)
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, test_labels, test_scores, _ = run_epoch(
        model, test_loader, optimizer, device, train=False
    )

    # ── Method A: dev-set ACER threshold (paper protocol) ─────────────────────
    dev_thr = ckpt["dev_acer_threshold"]
    m_dev   = evaluate(test_labels, test_scores, threshold=dev_thr)
    print_metrics(m_dev,
        f"Using dev-set ACER threshold ({dev_thr:.4f})")

    # ── Method B: EER threshold on test (robust cross-domain) ─────────────────
    m_eer = evaluate_with_eer_threshold(test_labels, test_scores)
    print_metrics(m_eer,
        "Using EER threshold (cross-domain robust)")

    print(f"\n  Best dev AUC was: {ckpt['best_auc']:.4f}% (epoch {ckpt['epoch']})")

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "test_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump({
            "protocol": "P2.2",
            "dev_threshold_metrics":  m_dev,
            "eer_threshold_metrics":  m_eer,
            "best_epoch":             int(ckpt["epoch"]),
            "best_dev_auc":           float(ckpt["best_auc"]),
        }, f)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
