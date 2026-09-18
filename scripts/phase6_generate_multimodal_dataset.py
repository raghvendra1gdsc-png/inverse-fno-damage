#!/usr/bin/env python3
"""
Phase 6 Dataset Generator: Augmented Multimodal Structural Sensor Observations.

Script: scripts/phase6_generate_multimodal_dataset.py
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team

Generates exact physical sensor observations for S0, S1, S2, and S4 across the 200
existing OpenSees simulation runs, strictly preserving the deterministic damage fields,
ground motions, run IDs, and train/val/test splits. Normalization statistics are
computed strictly from training runs to ensure zero test set leakage.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import numpy as np

from src.damage_injection import MultiStoryFrame, FrameConfig
from src.observability import SensorConfiguration, ForwardObservationMap
from src.forward_fno_model import split_simulation_dataset


def generate_multimodal_dataset():
    print("=" * 80)
    print("📦 [PHASE 6] GENERATING MULTIMODAL SENSOR DATASET (S0, S1, S2, S4)")
    print("=" * 80)

    src_dir = "data/opensees_runs"
    out_dir = "data/phase6_multimodal_runs"
    os.makedirs(out_dir, exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(src_dir, "sim_*.npz")))
    assert len(input_files) == 200, f"Expected 200 runs, found {len(input_files)}"

    frame = MultiStoryFrame(FrameConfig())
    cfg_s0 = SensorConfiguration.create_S0()
    cfg_s1 = SensorConfiguration.create_S1()
    cfg_s2 = SensorConfiguration.create_S2()
    cfg_s4 = SensorConfiguration.create_S4()

    print(f"Loaded {len(input_files)} source simulation runs from {src_dir}.")
    print("Generating exact OpenSees observations for S0, S1, S2, and S4...")

    for idx, fpath in enumerate(input_files, start=1):
        data = np.load(fpath)
        d_vec = data["damage_field"]
        gm = data["ground_accel"]
        dt = float(data["dt"])
        run_id = int(data["run_id"])
        gm_rec = str(data["gm_record"])

        # Extract sensor observations via Phase 5 verified ForwardObservationMap
        res_s0 = ForwardObservationMap.evaluate(frame, d_vec, gm, dt, cfg_s0)
        res_s1 = ForwardObservationMap.evaluate(frame, d_vec, gm, dt, cfg_s1)
        res_s2 = ForwardObservationMap.evaluate(frame, d_vec, gm, dt, cfg_s2)
        res_s4 = ForwardObservationMap.evaluate(frame, d_vec, gm, dt, cfg_s4)

        out_path = os.path.join(out_dir, os.path.basename(fpath))
        np.savez_compressed(
            out_path,
            run_id=run_id,
            gm_record=gm_rec,
            ground_accel=gm.astype(np.float32),
            damage_field=d_vec.astype(np.float32),
            time=res_s0["time"].astype(np.float32),
            dt=dt,
            S0_Y=res_s0["Y"].astype(np.float32),
            S1_Y=res_s1["Y"].astype(np.float32),
            S2_Y=res_s2["Y"].astype(np.float32),
            S4_Y=res_s4["Y"].astype(np.float32),
        )

        if idx % 40 == 0 or idx == len(input_files):
            print(f"  Processed [{idx:3d}/{len(input_files)}] runs -> {out_path}")

    # --------------------------------------------------------------------------
    # COMPUTE NORMALIZATION STATISTICS STRICTLY FROM TRAINING SPLIT
    # --------------------------------------------------------------------------
    print("\nComputing normalization statistics strictly on the training runs...")
    train_files, val_files, test_files = split_simulation_dataset(data_dir=src_dir)
    print(f"Split sizes: Train = {len(train_files)}, Val = {len(val_files)}, Held-out Test = {len(test_files)}")

    train_out_files = [os.path.join(out_dir, os.path.basename(f)) for f in train_files]

    # Accumulate per-channel statistics for each configuration
    configs_keys = ["S0_Y", "S1_Y", "S2_Y", "S4_Y", "ground_accel"]
    stats = {}

    for k in ["S0_Y", "S1_Y", "S2_Y", "S4_Y"]:
        all_vals = []
        for tf in train_out_files:
            d = np.load(tf)
            all_vals.append(d[k])  # [C, T]
        concat_vals = np.stack(all_vals, axis=0)  # [120, C, T]

        # Channel-wise mean and std across train batch and time
        ch_mean = np.mean(concat_vals, axis=(0, 2)).tolist()
        ch_std = np.std(concat_vals, axis=(0, 2))
        ch_std = np.maximum(ch_std, 1e-6).tolist()
        stats[k] = {
            "mean": ch_mean,
            "std": ch_std,
            "num_channels": len(ch_mean),
        }

    # Ground motion scalar stats
    all_gms = np.stack([np.load(tf)["ground_accel"] for tf in train_out_files], axis=0)
    stats["ground_accel"] = {
        "mean": float(np.mean(all_gms)),
        "std": float(max(np.std(all_gms), 1e-6)),
    }

    stats_path = os.path.join(out_dir, "phase6_normalization_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Normalization stats saved to {stats_path}")
    print(f"🎉 Dataset generation complete: 200 multimodal simulation runs ready in {out_dir}")


if __name__ == "__main__":
    generate_multimodal_dataset()
