"""
Inverse Fourier Neural Operator (Inverse FNO) for Structural Damage Identification.

Module: src.inverse_fno_model
Author: Inverse FNO Project Team
Context: Phase 2 - Inverse Operator Baseline (Unregularized Plain MSE Loss)

Maps:
    Sparse Sensor Accelerations Y(t) in R^{3 x T} -> Spatial Damage Field d in R^9
"""

from typing import Optional, Tuple, Dict, Any, List
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.forward_fno_model import SpectralConv1d, FNOBlock1d, split_simulation_dataset


class InverseFNO(nn.Module):
    """
    Inverse Fourier Neural Operator mapping:
        Sensor Accelerations Y(t) in R^{B x 3 x T} -> Damage Field d in R^{B x 9}

    Architecture:
      1. Lifting: Conv1d(3 -> width)
      2. 4 FNO Blocks: SpectralConv1d (32 modes) + 1x1 Conv bypass + GELU
      3. Global Spatiotemporal Pooling: [mean_pool, max_pool] -> R^{B x 2*width}
      4. Regression MLP Head: Linear(2*width -> 128 -> 64 -> 9) with ReLU activation
         (ensuring non-negative damage predictions d_e >= 0).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_elements: int = 9,
        modes: int = 32,
        width: int = 64,
        n_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_elements = num_elements
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        # 1. Lifting Layer P: in_channels -> width
        self.lifting = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=1),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.1),
            nn.Conv1d(width, width, kernel_size=1),
        )

        # 2. Fourier Neural Operator Blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(
                width=width,
                modes=modes,
                activation=activation,
            )
            for _ in range(n_layers)
        ])

        # 3. Regression Head: [mean_pool, max_pool] -> num_elements
        self.regressor = nn.Sequential(
            nn.Linear(width * 2, 128),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.1),
            nn.Linear(64, num_elements),
            nn.ReLU(),  # Physical constraint: damage cannot be negative
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        y : torch.Tensor
            Sensor response tensor [Batch, 3, Time_steps].

        Returns
        -------
        d_pred : torch.Tensor
            Predicted damage field [Batch, 9].
        """
        # Lifting
        v = self.lifting(y)

        # Spectral feature extraction across time
        for block in self.fno_blocks:
            v = block(v)

        # Global temporal feature pooling: combine mean and max features
        mean_pool = torch.mean(v, dim=-1)  # [B, width]
        max_pool = torch.max(v, dim=-1)[0]  # [B, width]
        feat = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2*width]

        # Predict element damage field
        d_pred = self.regressor(feat)
        return d_pred

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class InverseDataset(Dataset):
    """
    Dataset loader for inverse problem:
      Inputs: Sensor total acceleration histories Y(t) in R^{3 x T} (normalized)
      Targets: Ground truth damage field d in R^9
    """

    def __init__(
        self,
        file_list: List[str],
        stats: Optional[Dict[str, float]] = None,
        sensor_indices: Optional[List[int]] = None,
        is_train: bool = False,
    ):
        self.file_list = sorted(file_list)
        self.records = []
        self.is_train = is_train
        self.sensor_indices = sensor_indices

        for f in self.file_list:
            data = np.load(f)
            rec = {
                "filename": os.path.basename(f),
                "sensor_accel": data["sensor_total_accel"].astype(np.float32),  # (n_sensors, T)
                "ground_accel": data["ground_accel"].astype(np.float32),        # (T,)
                "time": data["time"].astype(np.float32),                        # (T,)
                "damage_field": data["damage_field"].astype(np.float32),        # (9,)
                "is_healthy": bool(np.max(data["damage_field"]) < 1e-4),
            }
            self.records.append(rec)

        if stats is None:
            all_sensor = np.concatenate([r["sensor_accel"].flatten() for r in self.records])
            all_gm = np.concatenate([r["ground_accel"].flatten() for r in self.records])
            self.stats = {
                "sensor_mean": float(np.mean(all_sensor)),
                "sensor_std": float(np.std(all_sensor)) if np.std(all_sensor) > 1e-6 else 1.0,
                "gm_mean": float(np.mean(all_gm)),
                "gm_std": float(np.std(all_gm)) if np.std(all_gm) > 1e-6 else 1.0,
            }
        else:
            self.stats = stats

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        T = len(rec["time"])
        raw_sensor = rec["sensor_accel"]
        if self.sensor_indices is not None:
            raw_sensor = raw_sensor[self.sensor_indices]

        y_norm = (raw_sensor - self.stats["sensor_mean"]) / self.stats["sensor_std"]

        # Ground motion normalized (1, T)
        gm_mean = self.stats.get("gm_mean", 0.0)
        gm_std = self.stats.get("gm_std", 1.0)
        gm_norm = ((rec["ground_accel"] - gm_mean) / gm_std)[np.newaxis, :]

        # Normalized time coordinate (1, T)
        t_max = rec["time"][-1] if rec["time"][-1] > 0 else 1.0
        time_norm = (rec["time"] / t_max)[np.newaxis, :]

        return {
            "y": torch.from_numpy(y_norm).float(),
            "damage": torch.from_numpy(rec["damage_field"]).float(),
            "gm_norm": torch.from_numpy(gm_norm).float(),
            "time_norm": torch.from_numpy(time_norm).float(),
            "is_healthy": torch.tensor(rec["is_healthy"], dtype=torch.bool),
            "filename": rec["filename"],
        }


def compute_inverse_metrics(
    d_preds: np.ndarray,
    d_trues: np.ndarray,
    is_healthy_flags: List[bool],
) -> Dict[str, Any]:
    """
    Computes rigorous localization and severity metrics on held-out test predictions.

    Parameters
    ----------
    d_preds : np.ndarray
        Predicted damage fields [N, 9].
    d_trues : np.ndarray
        Ground truth damage fields [N, 9].
    is_healthy_flags : List[bool]
        Boolean list indicating if ground truth was healthy (undamaged).
    """
    N = len(d_preds)

    # 1. Overall Severity Metrics
    mae_all = np.mean(np.abs(d_preds - d_trues))
    rmse_all = np.sqrt(np.mean((d_preds - d_trues) ** 2))

    # 2. Damaged Partition Metrics
    dmg_indices = [i for i in range(N) if not is_healthy_flags[i]]
    healthy_indices = [i for i in range(N) if is_healthy_flags[i]]

    top1_correct = 0
    top2_correct = 0
    severity_errs_on_damaged = []
    false_positive_on_intact = []
    story_localization_errors = []

    # Map element index (0..8) to story (1, 2, 3)
    # Cols 0,1 -> Story 1; Cols 2,3 -> Story 2; Cols 4,5 -> Story 3;
    # Beams 6 -> Story 1; Beam 7 -> Story 2; Beam 8 -> Story 3
    ele_to_story = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 1, 7: 2, 8: 3}

    for i in dmg_indices:
        pred_i = d_preds[i]
        true_i = d_trues[i]

        true_peak_ele = int(np.argmax(true_i))
        pred_top1_ele = int(np.argmax(pred_i))
        pred_top2_eles = np.argsort(pred_i)[-2:]

        if pred_top1_ele == true_peak_ele:
            top1_correct += 1
        if true_peak_ele in pred_top2_eles:
            top2_correct += 1

        # Story level localization distance
        true_story = ele_to_story[true_peak_ele]
        pred_story = ele_to_story[pred_top1_ele]
        story_localization_errors.append(abs(true_story - pred_story))

        # Severity error on truly damaged elements (d_true > 0.05)
        damaged_mask = true_i > 0.05
        if np.sum(damaged_mask) > 0:
            sev_err = np.mean(np.abs(pred_i[damaged_mask] - true_i[damaged_mask]))
            severity_errs_on_damaged.append(sev_err)

        # False positive ghost damage on truly intact elements (d_true < 0.01)
        intact_mask = true_i < 0.01
        if np.sum(intact_mask) > 0:
            fp_val = np.mean(pred_i[intact_mask])
            false_positive_on_intact.append(fp_val)

    # 3. Healthy Partition False Positives
    healthy_fps = []
    for i in healthy_indices:
        healthy_fps.append(np.mean(d_preds[i]))

    n_dmg = len(dmg_indices)
    top1_accuracy = (top1_correct / n_dmg) * 100.0 if n_dmg > 0 else 0.0
    top2_accuracy = (top2_correct / n_dmg) * 100.0 if n_dmg > 0 else 0.0
    top1_error_rate = 100.0 - top1_accuracy

    return {
        "overall": {
            "num_test_samples": N,
            "overall_mae": float(mae_all),
            "overall_rmse": float(rmse_all),
            "healthy_samples": len(healthy_indices),
            "damaged_samples": len(dmg_indices),
        },
        "localization": {
            "top1_accuracy_pct": float(top1_accuracy),
            "top1_error_rate_pct": float(top1_error_rate),
            "top2_accuracy_pct": float(top2_accuracy),
            "mean_story_localization_error": float(np.mean(story_localization_errors)) if story_localization_errors else 0.0,
        },
        "severity": {
            "mean_severity_error_on_damaged_elements": float(np.mean(severity_errs_on_damaged)) if severity_errs_on_damaged else 0.0,
            "mean_false_positive_damage_on_intact_elements": float(np.mean(false_positive_on_intact)) if false_positive_on_intact else 0.0,
            "mean_false_positive_on_healthy_records": float(np.mean(healthy_fps)) if healthy_fps else 0.0,
        },
    }


def train_inverse_fno(
    data_dir: str = "data/opensees_runs",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1.5e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    save_path: str = "models/inverse_fno_naive_mse.pt",
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Trains the Inverse FNO with plain MSE loss only (unregularized naive baseline).
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_files, val_files, test_files = split_simulation_dataset(data_dir=data_dir)

    train_ds = InverseDataset(train_files, is_train=True)
    norm_stats = train_ds.stats
    val_ds = InverseDataset(val_files, stats=norm_stats)
    test_ds = InverseDataset(test_files, stats=norm_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = InverseFNO(in_channels=3, num_elements=9, modes=32, width=64, n_layers=4)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if verbose:
        print("=" * 75)
        print("NAIVE INVERSE FNO TRAINING (Phase 2 Baseline — Plain MSE Loss Only)")
        print("=" * 75)
        print(f"Device:           {device}")
        print(f"Model Parameters: {model.get_num_parameters():,}")
        print(f"Loss Function:    Plain MSE (Unregularized, No Sparsity, No Cycle-Consistency)")
        print(f"Train Samples:    {len(train_ds)} | Val: {len(val_ds)} | Held-Out Test: {len(test_ds)}")
        print("=" * 75)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            y = batch["y"].to(device)
            d_true = batch["damage"].to(device)

            optimizer.zero_grad()
            d_pred = model(y)
            loss = criterion(d_pred, d_true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            total_batches += 1

        scheduler.step()
        avg_train = train_loss / total_batches

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                y = batch["y"].to(device)
                d_true = batch["damage"].to(device)
                d_pred = model(y)
                val_loss += criterion(d_pred, d_true).item()
                val_batches += 1

        avg_val = val_loss / val_batches

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats": norm_stats,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, save_path)

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f} (Best: {best_val_loss:.6f})")

    # Authoritative evaluation on held-out test split
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    all_d_preds = []
    all_d_trues = []
    is_healthy_flags = []
    sample_details = []

    with torch.no_grad():
        for batch in test_loader:
            y = batch["y"].to(device)
            d_true = batch["damage"].numpy()[0]
            is_h = bool(batch["is_healthy"].item())
            fname = batch["filename"][0]

            d_pred = model(y).cpu().numpy()[0]

            all_d_preds.append(d_pred)
            all_d_trues.append(d_true)
            is_healthy_flags.append(is_h)

            sample_details.append({
                "filename": fname,
                "is_healthy": is_h,
                "d_true": d_true.tolist(),
                "d_pred": d_pred.tolist(),
                "mae": float(np.mean(np.abs(d_pred - d_true))),
                "peak_true_ele": int(np.argmax(d_true)),
                "peak_pred_ele": int(np.argmax(d_pred)),
            })

    d_preds_arr = np.array(all_d_preds)
    d_trues_arr = np.array(all_d_trues)

    metrics = compute_inverse_metrics(d_preds_arr, d_trues_arr, is_healthy_flags)

    # Save metrics to results
    metrics_file = "results/inverse_fno_naive_metrics.json"
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print("\n" + "=" * 75)
        print("HELD-OUT TEST SET EVALUATION AUDIT (NAIVE MSE BASELINE)")
        print("=" * 75)
        print(f"Overall MAE:                     {metrics['overall']['overall_mae']:.4f}")
        print(f"Overall RMSE:                    {metrics['overall']['overall_rmse']:.4f}")
        print(f"Top-1 Localization Accuracy:     {metrics['localization']['top1_accuracy_pct']:.1f}% (Error Rate: {metrics['localization']['top1_error_rate_pct']:.1f}%)")
        print(f"Top-2 Localization Accuracy:     {metrics['localization']['top2_accuracy_pct']:.1f}%")
        print(f"Mean Story Localization Error:   {metrics['localization']['mean_story_localization_error']:.2f} stories")
        print(f"Mean Severity Error on Damaged:  {metrics['severity']['mean_severity_error_on_damaged_elements']:.4f}")
        print(f"Ghost Damage on Intact Elements: {metrics['severity']['mean_false_positive_damage_on_intact_elements']:.4f}")
        print("=" * 75)

    return model, metrics, d_preds_arr, d_trues_arr, sample_details


if __name__ == "__main__":
    train_inverse_fno()
