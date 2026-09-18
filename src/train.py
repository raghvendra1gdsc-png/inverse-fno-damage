"""
Training Engine for Regularized Inverse Fourier Neural Operator (Inverse FNO).

Module: src.train
Author: Inverse FNO Project Team
Context: Phase 3 - Physics-Constrained Inverse Training & Benchmark Audit

Logs three separate loss terms per epoch:
  1. Data Fidelity (MSE on damage field)
  2. Sparsity / Total Variation Prior (L1 + Graph TV on structural members)
  3. Cycle-Consistency (Dynamic sensor response verification via Forward FNO)

Supports configurable loss weights and executes side-by-side held-out evaluation
against the unregularized Phase 2 baseline.
"""

from typing import Optional, Dict, Any, List, Tuple
import os
import sys
import json
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.forward_fno_model import ForwardFNO, split_simulation_dataset
from src.inverse_fno_model import InverseFNO, InverseDataset, compute_inverse_metrics
from src.losses import CompositeInverseLoss


def train_regularized_inverse_fno(
    data_dir: str = "data/opensees_runs",
    forward_model_path: str = "models/forward_fno_best.pt",
    norm_stats_path: str = "models/normalization_stats.json",
    save_path: str = "models/inverse_fno_regularized.pt",
    lambda_data: float = 1.0,
    lambda_sparse: float = 0.05,
    lambda_tv: float = 0.01,
    lambda_cycle: float = 0.20,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1.5e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Trains Inverse FNO with configurable composite physics-informed loss.
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 1. Load Normalization Statistics
    with open(norm_stats_path, "r") as f:
        norm_stats = json.load(f)

    # 2. Data Splits
    train_files, val_files, test_files = split_simulation_dataset(data_dir=data_dir)
    train_ds = InverseDataset(train_files, stats=norm_stats, is_train=True)
    val_ds = InverseDataset(val_files, stats=norm_stats, is_train=False)
    test_ds = InverseDataset(test_files, stats=norm_stats, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Load Pre-Trained Forward FNO (Frozen for Cycle-Consistency)
    fwd_checkpoint = torch.load(forward_model_path, map_location=device)
    forward_fno = ForwardFNO(in_channels=11, out_channels=3, modes=32, width=64, n_layers=4)
    forward_fno.load_state_dict(fwd_checkpoint["model_state_dict"])
    forward_fno.to(device)
    forward_fno.eval()
    for param in forward_fno.parameters():
        param.requires_grad = False

    # 4. Instantiate Inverse FNO
    inverse_fno = InverseFNO(in_channels=3, num_elements=9, modes=32, width=64, n_layers=4)
    inverse_fno.to(device)

    # 5. Composite Loss Function with Individual Term Logging
    loss_fn = CompositeInverseLoss(
        forward_model=forward_fno,
        lambda_data=lambda_data,
        lambda_sparse=lambda_sparse,
        lambda_tv=lambda_tv,
        lambda_cycle=lambda_cycle,
    )

    optimizer = torch.optim.AdamW(inverse_fno.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    if verbose:
        print("=" * 88)
        print("PHASE 3: REGULARIZED INVERSE FNO TRAINING (Sparsity + TV + Forward Cycle-Consistency)")
        print("=" * 88)
        print(f"Device:               {device}")
        print(f"Model Parameters:     {inverse_fno.get_num_parameters():,}")
        print(f"Training Partitions:  Train = {len(train_ds)} | Val = {len(val_ds)} | Held-Out Test = {len(test_ds)}")
        print(f"Configurable Weights: lambda_data={lambda_data:.2f} | lambda_sparse={lambda_sparse:.3f} | lambda_tv={lambda_tv:.3f} | lambda_cycle={lambda_cycle:.3f}")
        print("=" * 88)
        print(f"{'Epoch':<7} | {'Total Loss':<10} | {'Data (MSE)':<10} | {'Sparsity':<9} | {'TV Prior':<9} | {'Cycle Loss':<10} | {'Val Total':<10} | {'Val MSE':<10}")
        print("-" * 88)

    training_history = []
    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        inverse_fno.train()
        epoch_terms = {"total": 0.0, "data": 0.0, "sparse": 0.0, "tv": 0.0, "cycle": 0.0}
        n_batches = 0

        for batch in train_loader:
            y_obs = batch["y"].to(device)
            d_true = batch["damage"].to(device)
            gm_norm = batch["gm_norm"].to(device)
            time_norm = batch["time_norm"].to(device)

            optimizer.zero_grad()
            d_pred = inverse_fno(y_obs)

            loss, loss_dict = loss_fn(
                d_pred=d_pred,
                d_true=d_true,
                y_obs=y_obs,
                gm_norm=gm_norm,
                time_norm=time_norm,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(inverse_fno.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_terms:
                epoch_terms[k] += loss_dict[k]
            n_batches += 1

        scheduler.step()
        for k in epoch_terms:
            epoch_terms[k] /= n_batches

        # Validation loop
        inverse_fno.eval()
        val_total = 0.0
        val_mse = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                y_obs = batch["y"].to(device)
                d_true = batch["damage"].to(device)
                gm_norm = batch["gm_norm"].to(device)
                time_norm = batch["time_norm"].to(device)

                d_pred = inverse_fno(y_obs)
                v_loss, v_dict = loss_fn(
                    d_pred=d_pred,
                    d_true=d_true,
                    y_obs=y_obs,
                    gm_norm=gm_norm,
                    time_norm=time_norm,
                )
                val_total += v_dict["total"]
                val_mse += v_dict["data"]
                val_batches += 1

        val_total /= val_batches
        val_mse /= val_batches

        epoch_record = {
            "epoch": epoch,
            "train": epoch_terms,
            "val_total": float(val_total),
            "val_mse": float(val_mse),
        }
        training_history.append(epoch_record)

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({
                "model_state_dict": inverse_fno.state_dict(),
                "norm_stats": norm_stats,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": {
                    "lambda_data": lambda_data,
                    "lambda_sparse": lambda_sparse,
                    "lambda_tv": lambda_tv,
                    "lambda_cycle": lambda_cycle,
                },
            }, save_path)

        # Log every 5 epochs, plus epoch 1 and last
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(
                f"{epoch:5d}   | "
                f"{epoch_terms['total']:10.5f} | "
                f"{epoch_terms['data']:10.5f} | "
                f"{epoch_terms['sparse']:9.5f} | "
                f"{epoch_terms['tv']:9.5f} | "
                f"{epoch_terms['cycle']:10.5f} | "
                f"{val_total:10.5f} | "
                f"{val_mse:10.5f}"
            )

    if verbose:
        print("=" * 88)

    # Save training log
    log_file = "results/inverse_regularized_training_log.json"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(training_history, f, indent=2)

    # Authoritative Evaluation on Held-Out Test Set
    checkpoint = torch.load(save_path, map_location=device)
    inverse_fno.load_state_dict(checkpoint["model_state_dict"])
    inverse_fno.eval()

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    all_d_preds = []
    all_d_trues = []
    is_healthy_flags = []

    with torch.no_grad():
        for batch in test_loader:
            y_obs = batch["y"].to(device)
            d_true = batch["damage"].numpy()[0]
            is_h = bool(batch["is_healthy"].item())

            d_pred = inverse_fno(y_obs).cpu().numpy()[0]

            all_d_preds.append(d_pred)
            all_d_trues.append(d_true)
            is_healthy_flags.append(is_h)

    d_preds_arr = np.array(all_d_preds)
    d_trues_arr = np.array(all_d_trues)

    reg_metrics = compute_inverse_metrics(d_preds_arr, d_trues_arr, is_healthy_flags)
    reg_metrics["config"] = {
        "lambda_data": lambda_data,
        "lambda_sparse": lambda_sparse,
        "lambda_tv": lambda_tv,
        "lambda_cycle": lambda_cycle,
        "epochs": epochs,
    }

    # Save regularized metrics
    reg_metrics_file = "results/inverse_fno_regularized_metrics.json"
    with open(reg_metrics_file, "w") as f:
        json.dump(reg_metrics, f, indent=2)

    # Side-by-side comparison against unregularized naive baseline
    naive_metrics_file = "results/inverse_fno_naive_metrics.json"
    naive_metrics = {}
    if os.path.exists(naive_metrics_file):
        with open(naive_metrics_file, "r") as f:
            naive_metrics = json.load(f)

    if verbose:
        print_comparison_table(naive_metrics, reg_metrics)

    return inverse_fno, reg_metrics, naive_metrics


def print_comparison_table(naive: Dict[str, Any], reg: Dict[str, Any]) -> None:
    """Prints a clear side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("PHASE 3 AUDIT: SIDE-BY-SIDE COMPARISON (NAIVE MSE BASELINE vs REGULARIZED INVERSE FNO)")
    print("=" * 90)
    print(f"{'Metric Description':<44} | {'Naive MSE Baseline':<20} | {'Regularized (+Cycle+Sparsity)':<20}")
    print("-" * 90)

    n_loc = naive.get("localization", {})
    r_loc = reg.get("localization", {})
    n_sev = naive.get("severity", {})
    r_sev = reg.get("severity", {})
    n_ov = naive.get("overall", {})
    r_ov = reg.get("overall", {})

    def fmt_pct(v):
        return f"{v:.1f}%" if v is not None else "N/A"

    def fmt_flt(v, d=4):
        return f"{v:.{d}f}" if v is not None else "N/A"

    print(f"{'Top-1 Localization Accuracy':<44} | {fmt_pct(n_loc.get('top1_accuracy_pct')):<20} | {fmt_pct(r_loc.get('top1_accuracy_pct')):<20}")
    print(f"{'Top-1 Localization Error Rate':<44} | {fmt_pct(n_loc.get('top1_error_rate_pct')):<20} | {fmt_pct(r_loc.get('top1_error_rate_pct')):<20}")
    print(f"{'Top-2 Localization Accuracy':<44} | {fmt_pct(n_loc.get('top2_accuracy_pct')):<20} | {fmt_pct(r_loc.get('top2_accuracy_pct')):<20}")
    print(f"{'Mean Story Localization Error':<44} | {fmt_flt(n_loc.get('mean_story_localization_error'), 2) + ' stories':<20} | {fmt_flt(r_loc.get('mean_story_localization_error'), 2) + ' stories':<20}")
    print(f"{'Severity Error on Damaged Elements':<44} | {fmt_flt(n_sev.get('mean_severity_error_on_damaged_elements')):<20} | {fmt_flt(r_sev.get('mean_severity_error_on_damaged_elements')):<20}")
    print(f"{'Ghost Damage on Intact Elements':<44} | {fmt_flt(n_sev.get('mean_false_positive_damage_on_intact_elements')):<20} | {fmt_flt(r_sev.get('mean_false_positive_damage_on_intact_elements')):<20}")
    print(f"{'False Positives on Healthy Frames':<44} | {fmt_flt(n_sev.get('mean_false_positive_on_healthy_records')):<20} | {fmt_flt(r_sev.get('mean_false_positive_on_healthy_records')):<20}")
    print(f"{'Overall MAE':<44} | {fmt_flt(n_ov.get('overall_mae')):<20} | {fmt_flt(r_ov.get('overall_mae')):<20}")
    print(f"{'Overall RMSE':<44} | {fmt_flt(n_ov.get('overall_rmse')):<20} | {fmt_flt(r_ov.get('overall_rmse')):<20}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Regularized Inverse FNO")
    parser.add_argument("--data-dir", type=str, default="data/opensees_runs")
    parser.add_argument("--forward-model", type=str, default="models/forward_fno_best.pt")
    parser.add_argument("--save-path", type=str, default="models/inverse_fno_regularized.pt")
    parser.add_argument("--lambda-data", type=float, default=1.0, help="Weight for data fidelity MSE")
    parser.add_argument("--lambda-sparse", type=float, default=0.05, help="Weight for L1 sparsity prior")
    parser.add_argument("--lambda-tv", type=float, default=0.01, help="Weight for graph TV prior")
    parser.add_argument("--lambda-cycle", type=float, default=0.20, help="Weight for forward cycle consistency")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    args = parser.parse_args()

    train_regularized_inverse_fno(
        data_dir=args.data_dir,
        forward_model_path=args.forward_model,
        save_path=args.save_path,
        lambda_data=args.lambda_data,
        lambda_sparse=args.lambda_sparse,
        lambda_tv=args.lambda_tv,
        lambda_cycle=args.lambda_cycle,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
