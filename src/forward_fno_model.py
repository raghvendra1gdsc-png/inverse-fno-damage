"""
Forward Fourier Neural Operator (FNO) for Structural Dynamics with Damage Fields.

Module: src.forward_fno_model
Author: Inverse FNO Project Team
Context: Phase 2 - Forward Operator Surrogate for Cycle-Consistency

Maps:
    (Ground Motion a_g(t) + Damage Field d) -> Sparse Sensor Response Y(t)
"""

from typing import Optional, Tuple, Dict, Any, List
import json
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution Layer (Li et al., 2020).

    Computes Fourier space convolution via Real FFT (rfft):
        K(v)(t) = F^{-1}( R(k) * F(v)(k) )(t)
    where R(k) is a learnable complex parameter tensor for k in [0, modes1 - 1].
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """(batch, in_channel, modes), (in_channel, out_channel, modes) -> (batch, out_channel, modes)"""
        return torch.einsum("bix,iox->box", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [Batch, In_channels, Time_steps]

        Returns
        -------
        out : torch.Tensor
            [Batch, Out_channels, Time_steps]
        """
        batchsize = x.shape[0]
        n_steps = x.shape[-1]

        # 1D real FFT
        x_ft = torch.fft.rfft(x, dim=-1)

        # Allocate complex output in frequency domain
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )

        active_modes = min(self.modes1, x_ft.shape[-1])
        out_ft[:, :, :active_modes] = self.compl_mul1d(
            x_ft[:, :, :active_modes], self.weights1[:, :, :active_modes]
        )

        # Inverse real FFT back to time domain
        x_out = torch.fft.irfft(out_ft, n=n_steps, dim=-1)
        return x_out


class FNOBlock1d(nn.Module):
    """
    Standard 1D Fourier Neural Operator Layer Block:
        v_{l+1}(t) = GELU( K(v_l)(t) + W v_l(t) )
    """

    def __init__(
        self,
        width: int,
        modes: int = 32,
        activation: str = "gelu",
        use_norm: bool = False,
    ):
        super().__init__()
        self.width = width
        self.modes = modes
        self.spectral_conv = SpectralConv1d(
            in_channels=width,
            out_channels=width,
            modes1=modes,
        )
        self.skip_conv = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.1)
        self.norm = nn.BatchNorm1d(width) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spectral = self.spectral_conv(x)
        x_skip = self.skip_conv(x)
        x_out = x_spectral + x_skip
        x_out = self.norm(x_out)
        return self.act(x_out)


class ForwardFNO(nn.Module):
    """
    Forward Fourier Neural Operator mapping:
        (Ground Motion a_g(t) + Damage Field d + Temporal Grid) -> Sensor Accelerations Y(t)

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 ground motion + 9 element damage + 1 time grid = 11).
    out_channels : int
        Number of output channels (3 sparse sensors: Floor 1, Floor 2, Roof total accelerations).
    modes : int
        Number of low-frequency Fourier modes kept in spectral convolutions (default: 32).
    width : int
        Latent channel width (default: 64).
    n_layers : int
        Number of FNO blocks (default: 4).
    """

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 3,
        modes: int = 32,
        width: int = 64,
        n_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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

        # 3. Projection Layer Q: width -> 2*width -> out_channels
        self.projection = nn.Sequential(
            nn.Conv1d(width, width * 2, kernel_size=1),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.1),
            nn.Conv1d(width * 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [Batch, In_channels, Time_steps].

        Returns
        -------
        out : torch.Tensor
            Predicted response tensor [Batch, Out_channels, Time_steps].
        """
        v = self.lifting(x)
        for block in self.fno_blocks:
            v = block(v)
        out = self.projection(v)
        return out

    def get_num_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DamageFrameDataset(Dataset):
    """
    Dataset loader for OpenSees simulation runs.

    Prepares inputs:
      Channel 0: Ground acceleration a_g(t) [normalized]
      Channels 1..9: Element damage field d_1..d_9 (repeated along time)
      Channel 10: Normalized temporal grid t / T_max in [0, 1]
    Prepares targets:
      Channels 0..2: Sparse sensor total accelerations (Floor 1, Floor 2, Roof) [normalized]
    """

    def __init__(
        self,
        file_list: List[str],
        stats: Optional[Dict[str, float]] = None,
        is_train: bool = False,
    ):
        self.file_list = sorted(file_list)
        self.records = []
        self.is_train = is_train

        # Preload records in memory
        for f in self.file_list:
            data = np.load(f)
            rec = {
                "filename": os.path.basename(f),
                "ground_accel": data["ground_accel"].astype(np.float32),   # (T,)
                "damage_field": data["damage_field"].astype(np.float32),   # (9,)
                "sensor_accel": data["sensor_total_accel"].astype(np.float32), # (3, T)
                "time": data["time"].astype(np.float32),                   # (T,)
                "modal_frequencies": data["modal_frequencies"].astype(np.float32),
                "is_healthy": bool(np.max(data["damage_field"]) < 1e-4),
            }
            self.records.append(rec)

        # Compute or use normalization statistics
        if stats is None:
            all_gm = np.concatenate([r["ground_accel"] for r in self.records])
            all_sensor = np.concatenate([r["sensor_accel"].flatten() for r in self.records])
            self.stats = {
                "gm_mean": float(np.mean(all_gm)),
                "gm_std": float(np.std(all_gm)) if np.std(all_gm) > 1e-6 else 1.0,
                "sensor_mean": float(np.mean(all_sensor)),
                "sensor_std": float(np.std(all_sensor)) if np.std(all_sensor) > 1e-6 else 1.0,
            }
        else:
            self.stats = stats

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        T = len(rec["time"])

        # 1. Normalize ground acceleration: (1, T)
        gm_norm = (rec["ground_accel"] - self.stats["gm_mean"]) / self.stats["gm_std"]
        gm_channel = gm_norm[np.newaxis, :]  # (1, T)

        # 2. Damage channels: (9, T) broadcast along time
        dmg_channels = np.repeat(rec["damage_field"][:, np.newaxis], T, axis=1)  # (9, T)

        # 3. Time grid channel: (1, T)
        t_max = rec["time"][-1] if rec["time"][-1] > 0 else 1.0
        time_channel = (rec["time"] / t_max)[np.newaxis, :]  # (1, T)

        # Concatenate 11 input channels
        input_tensor = np.concatenate([gm_channel, dmg_channels, time_channel], axis=0)  # (11, T)

        # 4. Normalize sensor output: (3, T)
        target_norm = (rec["sensor_accel"] - self.stats["sensor_mean"]) / self.stats["sensor_std"]

        return {
            "x": torch.from_numpy(input_tensor).float(),
            "y": torch.from_numpy(target_norm).float(),
            "y_raw": torch.from_numpy(rec["sensor_accel"]).float(),
            "damage": torch.from_numpy(rec["damage_field"]).float(),
            "is_healthy": torch.tensor(rec["is_healthy"], dtype=torch.bool),
            "filename": rec["filename"],
        }


def relative_l2_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Relative L2 error loss: ||y_pred - y_true||_2 / ||y_true||_2."""
    diff_norm = torch.norm(y_pred - y_true, p=2, dim=-1)
    true_norm = torch.norm(y_true, p=2, dim=-1) + 1e-8
    return torch.mean(diff_norm / true_norm)


def split_simulation_dataset(
    data_dir: str = "data/opensees_runs",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Splits simulation runs into stratified Train, Val, and Held-out Test sets."""
    index_file = os.path.join(data_dir, "dataset_index.json")
    with open(index_file, "r") as f:
        catalog = json.load(f)

    rng = np.random.RandomState(seed)
    healthy_files = [os.path.join(data_dir, r["filename"]) for r in catalog if r["num_damaged_elements"] == 0]
    damaged_files = [os.path.join(data_dir, r["filename"]) for r in catalog if r["num_damaged_elements"] > 0]

    rng.shuffle(healthy_files)
    rng.shuffle(damaged_files)

    def partition(files):
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return files[:n_train], files[n_train : n_train + n_val], files[n_train + n_val :]

    tr_h, val_h, te_h = partition(healthy_files)
    tr_d, val_d, te_d = partition(damaged_files)

    train_files = tr_h + tr_d
    val_files = val_h + val_d
    test_files = te_h + te_d

    rng.shuffle(train_files)
    rng.shuffle(val_files)
    rng.shuffle(test_files)

    return train_files, val_files, test_files


def train_forward_fno(
    data_dir: str = "data/opensees_runs",
    epochs: int = 35,
    batch_size: int = 16,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    save_path: str = "models/forward_fno_best.pt",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Trains the Forward FNO model on Phase 1 simulation data.

    Returns
    -------
    Dict[str, Any]
        Comprehensive evaluation report comparing overall, healthy, and damaged accuracy.
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_files, val_files, test_files = split_simulation_dataset(data_dir=data_dir)

    train_ds = DamageFrameDataset(train_files, is_train=True)
    norm_stats = train_ds.stats
    val_ds = DamageFrameDataset(val_files, stats=norm_stats)
    test_ds = DamageFrameDataset(test_files, stats=norm_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = ForwardFNO(in_channels=11, out_channels=3, modes=32, width=64, n_layers=4)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    mse_criterion = nn.MSELoss()

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if verbose:
        print("=" * 75)
        print("FORWARD FNO TRAINING (Phase 2 Surrogate Model)")
        print("=" * 75)
        print(f"Device:               {device}")
        print(f"Total Parameters:     {model.get_num_parameters():,}")
        print(f"Dataset Partitions:   Train = {len(train_ds)} | Val = {len(val_ds)} | Held-Out Test = {len(test_ds)}")
        print(f"Held-Out Test Split:  {sum(1 for r in test_ds.records if r['is_healthy'])} Healthy, {sum(1 for r in test_ds.records if not r['is_healthy'])} Damaged")
        print("=" * 75)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_rel_l2 = 0.0
        total_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            y_pred = model(x)

            loss_rel = relative_l2_loss(y_pred, y)
            loss_mse = mse_criterion(y_pred, y)
            loss = loss_rel + 0.1 * loss_mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_rel_l2 += loss_rel.item()
            total_batches += 1

        scheduler.step()
        avg_train_loss = train_rel_l2 / total_batches

        # Validation loop
        model.eval()
        val_rel_l2 = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                y_pred = model(x)
                val_rel_l2 += relative_l2_loss(y_pred, y).item()
                val_batches += 1

        avg_val_loss = val_rel_l2 / val_batches
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats": norm_stats,
                "config": {
                    "in_channels": 11,
                    "out_channels": 3,
                    "modes": 32,
                    "width": 64,
                    "n_layers": 4,
                },
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, save_path)

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Rel L2: {avg_train_loss * 100:.2f}% | Val Rel L2: {avg_val_loss * 100:.2f}% (Best: {best_val_loss * 100:.2f}%)")

    # Load best checkpoint for authoritative testing
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Authoritative Held-out Evaluation
    metrics = evaluate_forward_fno(model, test_ds, norm_stats, device=device)

    # Save stats and metrics
    metrics_path = "results/forward_fno_metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    stats_path = "models/normalization_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    return metrics


def evaluate_forward_fno(
    model: nn.Module,
    dataset: DamageFrameDataset,
    stats: Dict[str, float],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates the Forward FNO model on physical acceleration units (m/s^2),
    breaking down results into Overall, Healthy-Only, and Damaged-Only partitions.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    overall_rel_l2 = []
    overall_corr = []
    overall_peak_err = []

    healthy_rel_l2 = []
    healthy_corr = []
    healthy_peak_err = []

    damaged_rel_l2 = []
    damaged_corr = []
    damaged_peak_err = []

    sensor_rel_l2 = {0: [], 1: [], 2: []}  # Floor 1, Floor 2, Roof

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_raw = batch["y_raw"].numpy()[0]  # (3, T) physical m/s^2
            is_healthy = bool(batch["is_healthy"].item())

            y_pred_norm = model(x).cpu().numpy()[0]  # (3, T)
            # Unnormalize to physical m/s^2
            y_pred_raw = y_pred_norm * stats["sensor_std"] + stats["sensor_mean"]

            # Compute relative L2 error across all 3 sensors
            rel_l2 = np.linalg.norm(y_pred_raw - y_raw) / np.linalg.norm(y_raw) * 100.0

            # Pearson correlation
            corrs = [np.corrcoef(y_pred_raw[s], y_raw[s])[0, 1] for s in range(3)]
            mean_corr = float(np.mean(corrs))

            # Peak acceleration error
            peak_pred = np.max(np.abs(y_pred_raw))
            peak_true = np.max(np.abs(y_raw))
            peak_err = abs(peak_pred - peak_true) / peak_true * 100.0

            # Per-sensor error
            for s in range(3):
                s_l2 = np.linalg.norm(y_pred_raw[s] - y_raw[s]) / np.linalg.norm(y_raw[s]) * 100.0
                sensor_rel_l2[s].append(s_l2)

            overall_rel_l2.append(rel_l2)
            overall_corr.append(mean_corr)
            overall_peak_err.append(peak_err)

            if is_healthy:
                healthy_rel_l2.append(rel_l2)
                healthy_corr.append(mean_corr)
                healthy_peak_err.append(peak_err)
            else:
                damaged_rel_l2.append(rel_l2)
                damaged_corr.append(mean_corr)
                damaged_peak_err.append(peak_err)

    report = {
        "overall": {
            "num_samples": len(overall_rel_l2),
            "mean_rel_l2_pct": float(np.mean(overall_rel_l2)),
            "std_rel_l2_pct": float(np.std(overall_rel_l2)),
            "median_rel_l2_pct": float(np.median(overall_rel_l2)),
            "mean_correlation": float(np.mean(overall_corr)),
            "mean_peak_error_pct": float(np.mean(overall_peak_err)),
        },
        "healthy_only": {
            "num_samples": len(healthy_rel_l2),
            "mean_rel_l2_pct": float(np.mean(healthy_rel_l2)) if healthy_rel_l2 else 0.0,
            "std_rel_l2_pct": float(np.std(healthy_rel_l2)) if healthy_rel_l2 else 0.0,
            "mean_correlation": float(np.mean(healthy_corr)) if healthy_corr else 0.0,
            "mean_peak_error_pct": float(np.mean(healthy_peak_err)) if healthy_peak_err else 0.0,
        },
        "damaged_only": {
            "num_samples": len(damaged_rel_l2),
            "mean_rel_l2_pct": float(np.mean(damaged_rel_l2)) if damaged_rel_l2 else 0.0,
            "std_rel_l2_pct": float(np.std(damaged_rel_l2)) if damaged_rel_l2 else 0.0,
            "mean_correlation": float(np.mean(damaged_corr)) if damaged_corr else 0.0,
            "mean_peak_error_pct": float(np.mean(damaged_peak_err)) if damaged_peak_err else 0.0,
        },
        "per_sensor": {
            "floor_1_node_3_rel_l2_pct": float(np.mean(sensor_rel_l2[0])),
            "floor_2_node_5_rel_l2_pct": float(np.mean(sensor_rel_l2[1])),
            "roof_node_7_rel_l2_pct": float(np.mean(sensor_rel_l2[2])),
        },
        "degradation_audit": {
            "delta_rel_l2_damaged_minus_healthy_pct": float(np.mean(damaged_rel_l2) - np.mean(healthy_rel_l2)) if (damaged_rel_l2 and healthy_rel_l2) else 0.0,
            "seismo_fno_elastic_benchmark_rel_l2_pct": 2.67,  # from seismoFNO linear_mvp_metrics.json (2.67%)
        },
    }

    return report


if __name__ == "__main__":
    train_forward_fno(epochs=35, batch_size=16, learning_rate=2e-3)
