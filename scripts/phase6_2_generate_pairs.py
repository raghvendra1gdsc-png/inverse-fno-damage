"""
Phase 6.2: Generate Controlled Bilateral Pairs for Identifiability Learning.

For each selected earthquake ground motion:
  State A: 30% stiffness loss in Left Column 1 (Story 1)
  State B: 30% stiffness loss in Right Column 1 (Story 1)
Uses identical ground excitation a_g(t) for both states.
Extracts observations for S0, S1, S2, S4.
Ensures zero leakage with the canonical held-out test benchmark.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
from src.damage_injection import MultiStoryFrame, FrameConfig
from src.observability import SensorConfiguration, ForwardObservationMap, BilateralSymmetry
from src.forward_fno_model import split_simulation_dataset


def generate_bilateral_pairs(
    num_train_pairs: int = 40,
    num_val_pairs: int = 15,
    out_dir: str = "data/phase6_2_pairs",
    manifest_file: str = "results/phase6_2/pair_dataset_manifest.json",
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_file), exist_ok=True)

    # 1. Obtain train/val splits strictly from opensees runs
    train_files, val_files, test_files = split_simulation_dataset()
    print(f"Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # 2. Get canonical bilateral damage vectors
    d_A, d_B = BilateralSymmetry.get_canonical_states()
    v_AB = (d_A - d_B) / np.linalg.norm(d_A - d_B)

    frame = MultiStoryFrame(FrameConfig())
    configs = {
        "S0": SensorConfiguration.create_S0(),
        "S1": SensorConfiguration.create_S1(),
        "S2": SensorConfiguration.create_S2(),
        "S4": SensorConfiguration.create_S4(),
    }

    manifest = {
        "num_train_pairs": num_train_pairs,
        "num_val_pairs": num_val_pairs,
        "damage_state_A": [float(x) for x in d_A],
        "damage_state_B": [float(x) for x in d_B],
        "unit_direction_v_AB": [float(x) for x in v_AB],
        "train_pairs": [],
        "val_pairs": [],
    }

    # 3. Generate Training Pairs
    print(f"\nGenerating {num_train_pairs} training bilateral pairs...")
    for idx in range(num_train_pairs):
        f = train_files[idx]
        d_sim = np.load(f)
        gm = d_sim["ground_accel"]
        dt = float(d_sim["dt"])
        gm_name = str(d_sim["gm_record"])

        pair_data = {
            "pair_id": f"train_pair_{idx:03d}",
            "gm_record": gm_name,
            "dt": dt,
            "ground_accel": gm,
            "d_A": d_A,
            "d_B": d_B,
            "v_AB": v_AB,
        }

        # Run OpenSees forward simulation for State A and State B
        for cname, cobj in configs.items():
            res_A = ForwardObservationMap.evaluate(frame, d_A, gm, dt, cobj)
            res_B = ForwardObservationMap.evaluate(frame, d_B, gm, dt, cobj)
            pair_data[f"{cname}_Y_A"] = res_A["Y"]
            pair_data[f"{cname}_Y_B"] = res_B["Y"]

        pair_path = os.path.join(out_dir, f"pair_train_{idx:03d}.npz")
        np.savez_compressed(pair_path, **pair_data)
        manifest["train_pairs"].append({
            "pair_id": f"train_pair_{idx:03d}",
            "file": pair_path,
            "gm_record": gm_name,
        })
        if (idx + 1) % 10 == 0:
            print(f"  Generated {idx + 1}/{num_train_pairs} training pairs.")

    # 4. Generate Validation Pairs (strictly disjoint ground motions from train pairs)
    train_gm_set = {p["gm_record"] for p in manifest["train_pairs"]}
    val_files_disjoint = [f for f in val_files if str(np.load(f)["gm_record"]) not in train_gm_set]
    print(f"\nGenerating {num_val_pairs} validation bilateral pairs (strictly disjoint GMs)...")
    for idx in range(num_val_pairs):
        f = val_files_disjoint[idx]
        d_sim = np.load(f)
        gm = d_sim["ground_accel"]
        dt = float(d_sim["dt"])
        gm_name = str(d_sim["gm_record"])

        pair_data = {
            "pair_id": f"val_pair_{idx:03d}",
            "gm_record": gm_name,
            "dt": dt,
            "ground_accel": gm,
            "d_A": d_A,
            "d_B": d_B,
            "v_AB": v_AB,
        }

        for cname, cobj in configs.items():
            res_A = ForwardObservationMap.evaluate(frame, d_A, gm, dt, cobj)
            res_B = ForwardObservationMap.evaluate(frame, d_B, gm, dt, cobj)
            pair_data[f"{cname}_Y_A"] = res_A["Y"]
            pair_data[f"{cname}_Y_B"] = res_B["Y"]

        pair_path = os.path.join(out_dir, f"pair_val_{idx:03d}.npz")
        np.savez_compressed(pair_path, **pair_data)
        manifest["val_pairs"].append({
            "pair_id": f"val_pair_{idx:03d}",
            "file": pair_path,
            "gm_record": gm_name,
        })

    # Save manifest
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ All pairs generated and saved in {out_dir}/")
    print(f"✅ Manifest saved to {manifest_file}")
    return manifest


if __name__ == "__main__":
    generate_bilateral_pairs()
