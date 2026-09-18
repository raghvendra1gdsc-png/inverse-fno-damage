"""
Sensor Sparsity Sweep and Graceful Degradation Evaluation Engine.

Module: src.evaluate
Author: Inverse FNO Project Team
Context: Phase 4 - Sensor Sparsity Sweep (Sensor Count from 10 down to 2)

Evaluates inverse damage identification performance under progressive sensor degradation:
  - Sensor counts: 10, 8, 6, 4, 2
  - Deterministic subset selection using fixed random seed
  - Retraining from scratch for each count to evaluate true physical capacity
  - Rigorous localization and severity metrics on held-out test split
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

from src.forward_fno_model import split_simulation_dataset
from src.inverse_fno_model import InverseFNO, InverseDataset, compute_inverse_metrics


CHANNEL_DESCRIPTIONS = {
    0: "Node 3 DOF 1 (Story 1 Left Horiz)",
    1: "Node 4 DOF 1 (Story 1 Right Horiz)",
    2: "Node 5 DOF 1 (Story 2 Left Horiz)",
    3: "Node 6 DOF 1 (Story 2 Right Horiz)",
    4: "Node 7 DOF 1 (Story 3 Left Horiz)",
    5: "Node 8 DOF 1 (Story 3 Right Horiz)",
    6: "Node 3 DOF 2 (Story 1 Left Vert)",
    7: "Node 4 DOF 2 (Story 1 Right Vert)",
    8: "Node 5 DOF 2 (Story 2 Left Vert)",
    9: "Node 6 DOF 2 (Story 2 Right Vert)",
}


def select_nested_sensor_subsets(
    total_sensors: int = 10,
    counts: List[int] = [10, 8, 6, 4, 2],
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Generates nested deterministic sensor subsets by progressively pruning sensors.
    Single-seed result (seed=42).
    """
    rng = np.random.RandomState(seed)
    current_sensors = list(range(total_sensors))
    subsets = {total_sensors: list(current_sensors)}

    sorted_counts = sorted([c for c in counts if c < total_sensors], reverse=True)
    for target_count in sorted_counts:
        num_to_remove = len(current_sensors) - target_count
        drop_indices = rng.choice(current_sensors, size=num_to_remove, replace=False)
        current_sensors = [s for s in current_sensors if s not in drop_indices]
        subsets[target_count] = list(current_sensors)

    return subsets


def train_and_eval_single_count(
    k_sensors: int,
    sensor_indices: List[int],
    data_dir: str = "data/opensees_runs",
    norm_stats_path: str = "models/normalization_stats.json",
    save_path: str = "models/inverse_fno_sensor_sweep.pt",
    lambda_data: float = 1.0,
    lambda_sparse: float = 0.02,
    epochs: int = 35,
    batch_size: int = 16,
    learning_rate: float = 1.5e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Retrains and evaluates an Inverse FNO model for a specific sensor count K.
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(norm_stats_path, "r") as f:
        norm_stats = json.load(f)

    train_files, val_files, test_files = split_simulation_dataset(data_dir=data_dir)

    train_ds = InverseDataset(train_files, stats=norm_stats, sensor_indices=sensor_indices, is_train=True)
    val_ds = InverseDataset(val_files, stats=norm_stats, sensor_indices=sensor_indices, is_train=False)
    test_ds = InverseDataset(test_files, stats=norm_stats, sensor_indices=sensor_indices, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Instantiate dedicated model for this exact sensor count K
    model = InverseFNO(in_channels=k_sensors, num_elements=9, modes=32, width=64, n_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    mse_criterion = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            y = batch["y"].to(device)
            d_true = batch["damage"].to(device)

            optimizer.zero_grad()
            d_pred = model(y)
            loss_data = mse_criterion(d_pred, d_true)
            loss_sparse = torch.mean(torch.abs(d_pred))
            total_loss = lambda_data * loss_data + lambda_sparse * loss_sparse

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_mse = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                y = batch["y"].to(device)
                d_true = batch["damage"].to(device)
                val_mse += mse_criterion(model(y), d_true).item()
                val_batches += 1
        val_mse /= val_batches

        if val_mse < best_val:
            best_val = val_mse
            torch.save({"model_state_dict": model.state_dict()}, save_path)

    # Authoritative evaluation on held-out test split
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    all_d_preds = []
    all_d_trues = []
    is_healthy_flags = []
    sample_records = []

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

            sample_records.append({
                "filename": fname,
                "is_healthy": is_h,
                "d_true": d_true.tolist(),
                "d_pred": d_pred.tolist(),
                "peak_true": int(np.argmax(d_true)),
                "peak_pred": int(np.argmax(d_pred)),
            })

    metrics = compute_inverse_metrics(np.array(all_d_preds), np.array(all_d_trues), is_healthy_flags)
    metrics["sensor_count"] = k_sensors
    metrics["sensor_indices"] = sensor_indices
    metrics["channel_descriptions"] = [CHANNEL_DESCRIPTIONS[i] for i in sensor_indices]

    return metrics, sample_records


def run_sensor_sparsity_sweep(
    data_dir: str = "data/opensees_runs",
    norm_stats_path: str = "models/normalization_stats.json",
    counts: List[int] = [10, 8, 6, 4, 2],
    seed: int = 42,
    epochs: int = 35,
    output_file: str = "results/sensor_sparsity_sweep.json",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes the sensor count sweep from 10 down to 2 sensors.
    """
    subsets = select_nested_sensor_subsets(total_sensors=10, counts=counts, seed=seed)

    if verbose:
        print("=" * 88)
        print("PHASE 4: SENSOR SPARSITY SWEEP (Graceful Degradation Analysis)")
        print("=" * 88)
        print(f"Sweep Counts:   {counts}")
        print(f"Random Seed:    {seed} (NOTE: This is a single-seed result)")
        print("Methodology:    Retraining dedicated Inverse FNO per sensor count to evaluate true capacity")
        print("=" * 88)
        for k in counts:
            print(f"  K = {k:2d} Sensors: Indices {subsets[k]}")
        print("=" * 88)

    sweep_results = {
        "metadata": {
            "counts": counts,
            "seed": seed,
            "note": "Single-seed result (seed=42) using deterministic nested sensor pruning.",
            "epochs": epochs,
            "methodology": "Retraining dedicated InverseFNO per sensor count with L1 sparsity prior.",
        },
        "metrics_by_count": {},
        "samples_by_count": {},
    }

    for k in counts:
        indices = subsets[k]
        save_path = f"models/inverse_fno_sweep_k{k}.pt"
        if verbose:
            print(f"\n--- Training & Evaluating Inverse FNO for K = {k} Sensors ---")

        metrics, sample_records = train_and_eval_single_count(
            k_sensors=k,
            sensor_indices=indices,
            data_dir=data_dir,
            norm_stats_path=norm_stats_path,
            save_path=save_path,
            epochs=epochs,
            seed=seed,
        )

        sweep_results["metrics_by_count"][str(k)] = metrics
        sweep_results["samples_by_count"][str(k)] = sample_records

        if verbose:
            loc = metrics["localization"]
            sev = metrics["severity"]
            print(f"  Result K={k:2d}: Top-1 Acc = {loc['top1_accuracy_pct']:5.1f}% | "
                  f"Top-2 Acc = {loc['top2_accuracy_pct']:5.1f}% | "
                  f"Story Error = {loc['mean_story_localization_error']:.2f} | "
                  f"Severity Error = {sev['mean_severity_error_on_damaged_elements']:.4f} | "
                  f"Ghost Dmg = {sev['mean_false_positive_damage_on_intact_elements']:.4f}")

    # Save results to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(sweep_results, f, indent=2)

    if verbose:
        print_sweep_summary(sweep_results)

    return sweep_results


def print_sweep_summary(sweep_results: Dict[str, Any]) -> None:
    """Prints a clear tabular summary of the sensor count sweep."""
    print("\n" + "=" * 96)
    print("PHASE 4 SUMMARY: SENSOR SPARSITY SWEEP (10 DOWN TO 2 SENSORS)")
    print("NOTE: This is a single-seed result (seed=42).")
    print("=" * 96)
    print(f"{'Sensors (K)':<12} | {'Top-1 Acc (%)':<14} | {'Top-2 Acc (%)':<14} | {'Story Error':<14} | {'Severity Error':<16} | {'Ghost Damage':<14}")
    print("-" * 96)

    counts = sweep_results["metadata"]["counts"]
    for k in counts:
        m = sweep_results["metrics_by_count"][str(k)]
        loc = m["localization"]
        sev = m["severity"]
        print(f"{k:<12d} | "
              f"{loc['top1_accuracy_pct']:<14.1f} | "
              f"{loc['top2_accuracy_pct']:<14.1f} | "
              f"{loc['mean_story_localization_error']:<14.2f} | "
              f"{sev['mean_severity_error_on_damaged_elements']:<16.4f} | "
              f"{sev['mean_false_positive_damage_on_intact_elements']:<14.4f}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sensor Sparsity Sweep")
    parser.add_argument("--data-dir", type=str, default="data/opensees_runs")
    parser.add_argument("--norm-stats", type=str, default="models/normalization_stats.json")
    parser.add_argument("--output-file", type=str, default="results/sensor_sparsity_sweep.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()

    run_sensor_sparsity_sweep(
        data_dir=args.data_dir,
        norm_stats_path=args.norm_stats,
        output_file=args.output_file,
        seed=args.seed,
        epochs=args.epochs,
    )
