"""
Phase 6 Training Engine for G-FNO Models.

Script: scripts/phase6_train.py
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team
"""

from typing import Dict, Tuple, Optional, Any, List
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.observability import SensorConfiguration
from src.graph.structural_graph import StructuralGraph
from src.ml.gfno import DualStreamGFNO
from src.ml.dataset import MultimodalDamageDataset
from src.ml.losses import HierarchicalDamageLoss
from src.forward_fno_model import split_simulation_dataset


def compute_epoch_metrics(
    all_preds: np.ndarray,
    all_trues: np.ndarray,
    all_probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes localization and severity metrics:
      - Support Precision, Recall, F1 (threshold = 0.05 on predicted damage or prob > 0.5)
      - Top-1 element localization accuracy
      - Top-2 element localization accuracy
      - Severity MAE (all elements)
      - Damaged-only severity MAE
      - False positive ghost damage magnitude
    """
    N, num_ele = all_trues.shape
    z_true = (all_trues > 0.01).astype(int)

    if all_probs is not None:
        z_pred = (all_probs > 0.5).astype(int)
    else:
        z_pred = (all_preds > 0.05).astype(int)

    # Classification Metrics (Support)
    tp = np.sum((z_pred == 1) & (z_true == 1))
    fp = np.sum((z_pred == 1) & (z_true == 0))
    fn = np.sum((z_pred == 0) & (z_true == 1))
    tn = np.sum((z_pred == 0) & (z_true == 0))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Top-1 and Top-2 Localization on Damaged Structures
    top1_correct = 0
    top2_correct = 0
    num_damaged_structs = 0

    for i in range(N):
        if np.max(all_trues[i]) > 0.01:
            num_damaged_structs += 1
            true_top = int(np.argmax(all_trues[i]))
            pred_order = np.argsort(all_preds[i])[::-1]
            if pred_order[0] == true_top:
                top1_correct += 1
            if true_top in pred_order[:2]:
                top2_correct += 1

    top1_acc = float(top1_correct / num_damaged_structs) if num_damaged_structs > 0 else 0.0
    top2_acc = float(top2_correct / num_damaged_structs) if num_damaged_structs > 0 else 0.0

    # Severity Metrics
    mae_all = float(np.mean(np.abs(all_preds - all_trues)))
    damaged_mask = (all_trues > 0.01)
    mae_dam = float(np.mean(np.abs(all_preds[damaged_mask] - all_trues[damaged_mask]))) if np.sum(damaged_mask) > 0 else 0.0

    # Ghost damage on healthy elements
    healthy_mask = (all_trues <= 0.01)
    ghost_mag = float(np.mean(all_preds[healthy_mask])) if np.sum(healthy_mask) > 0 else 0.0

    return {
        "support_precision": prec,
        "support_recall": rec,
        "support_f1": f1,
        "top1_localization_acc": top1_acc,
        "top2_localization_acc": top2_acc,
        "severity_mae_all": mae_all,
        "severity_mae_damaged": mae_dam,
        "ghost_damage_mag": ghost_mag,
    }


def train_phase6_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int = 35,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Executes training loop with validation tracking and best-model checkpointing.
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    history = []

    for ep in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            Y = batch["Y"].to(device)
            Y_glob = batch["Y_global"].to(device)
            gm = batch["ground_accel"].to(device)
            d_true = batch["damage"].to(device)

            optimizer.zero_grad()
            out = model(Y, ground_accel=gm, Y_global=Y_glob)
            loss, _ = loss_fn(out, d_true, H_nodes=out.get("H_nodes", None))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss_accum += float(loss.item())

        train_loss_avg = train_loss_accum / len(train_loader)

        # --- VALIDATE ---
        model.eval()
        val_loss_accum = 0.0
        val_preds, val_trues, val_probs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                Y = batch["Y"].to(device)
                Y_glob = batch["Y_global"].to(device)
                gm = batch["ground_accel"].to(device)
                d_true = batch["damage"].to(device)

                out = model(Y, ground_accel=gm, Y_global=Y_glob)
                loss, _ = loss_fn(out, d_true, H_nodes=out.get("H_nodes", None))
                val_loss_accum += float(loss.item())

                val_preds.append(out["damage_pred"].cpu().numpy())
                val_trues.append(d_true.cpu().numpy())
                val_probs.append(out["prob_support"].cpu().numpy())

        val_loss_avg = val_loss_accum / len(val_loader)
        val_preds_mat = np.concatenate(val_preds, axis=0)
        val_trues_mat = np.concatenate(val_trues, axis=0)
        val_probs_mat = np.concatenate(val_probs, axis=0)

        ep_metrics = compute_epoch_metrics(val_preds_mat, val_trues_mat, val_probs_mat)
        ep_metrics["epoch"] = ep
        ep_metrics["train_loss"] = train_loss_avg
        ep_metrics["val_loss"] = val_loss_avg
        history.append(ep_metrics)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep

        if verbose and (ep % 10 == 0 or ep == epochs):
            print(f"  Epoch [{ep:2d}/{epochs:2d}] | Train: {train_loss_avg:.4f} | Val: {val_loss_avg:.4f} | "
                  f"F1: {ep_metrics['support_f1']:.3f} | Top1: {ep_metrics['top1_localization_acc']*100:.1f}% | "
                  f"Dam-MAE: {ep_metrics['severity_mae_damaged']:.4f}")

    # Load best checkpoint weights
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Final evaluation of best model on validation set
    model.eval()
    val_preds, val_trues, val_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            Y = batch["Y"].to(device)
            Y_glob = batch["Y_global"].to(device)
            gm = batch["ground_accel"].to(device)
            d_true = batch["damage"].to(device)

            out = model(Y, ground_accel=gm, Y_global=Y_glob)
            val_preds.append(out["damage_pred"].cpu().numpy())
            val_trues.append(d_true.cpu().numpy())
            val_probs.append(out["prob_support"].cpu().numpy())

    final_metrics = compute_epoch_metrics(
        np.concatenate(val_preds, axis=0),
        np.concatenate(val_trues, axis=0),
        np.concatenate(val_probs, axis=0),
    )
    final_metrics["best_epoch"] = best_epoch
    final_metrics["best_val_loss"] = best_val_loss

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_metrics": final_metrics,
        "history": history,
    }
