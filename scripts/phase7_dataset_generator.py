"""
Phase 7.2 Dataset Generator for Cross-Structure Generalization.

Script: scripts/phase7_dataset_generator.py
Author: SeismoFNO Research Team
Context: Phase 7.2 Dataset Generation & Manifest Audit
"""

import os
import sys
import glob
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.damage_injection import MultiStoryFrame, parse_at2_ground_motion, sample_random_damage_field
from src.structure_variants import (
    get_structure_config,
    get_dynamic_sensor_config,
    TopologyObservationMap,
    STRUCTURE_SPECS,
)
from src.structure_features import extract_structural_graph_features


def generate_phase7_dataset(
    output_dir: str = "data/phase7_simulations",
    gm_dir: str = "data/raw_ground_motions",
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 80)
    print("🚀 PHASE 7.2 DATASET GENERATION: CROSS-STRUCTURE BENCHMARK")
    print("=" * 80)

    t0_all = time.time()
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    # 1. Audit Ground Motions and Verify Partition Disjointness
    gm_files = sorted(glob.glob(os.path.join(gm_dir, "*.AT2")))
    assert len(gm_files) == 120, f"Expected 120 .AT2 files, found {len(gm_files)}"

    # Exact globally disjoint partition
    train_gm_files = gm_files[:70]      # RSN0001..RSN0070
    val_gm_files = gm_files[70:90]      # RSN0071..RSN0090
    test_gm_files = gm_files[90:120]    # RSN0091..RSN0120

    train_names = {os.path.basename(f) for f in train_gm_files}
    val_names = {os.path.basename(f) for f in val_gm_files}
    test_names = {os.path.basename(f) for f in test_gm_files}

    assert len(train_names.intersection(val_names)) == 0, "Train ∩ Val overlap detected!"
    assert len(train_names.intersection(test_names)) == 0, "Train ∩ Test overlap detected!"
    assert len(val_names.intersection(test_names)) == 0, "Val ∩ Test overlap detected!"
    print(f"Verified 120 ground motions: 70 Train, 20 Val, 30 Test (Disjointness = 100%)")

    # Pre-parse and interpolate ground motions to 1000 steps at target_dt = 0.01 s
    target_dt = 0.01
    n_steps = 1000
    sim_times = np.arange(n_steps) * target_dt

    parsed_gms = {}
    for f in gm_files:
        fname = os.path.basename(f)
        dt_native, full_accel, meta = parse_at2_ground_motion(f)
        native_times = np.arange(len(full_accel)) * dt_native
        # Safe interpolation / zero-padding
        accel_interp = np.interp(sim_times, native_times, full_accel, left=0.0, right=0.0)
        parsed_gms[fname] = {
            "record_name": meta["record_name"],
            "pga_ms2": float(np.max(np.abs(accel_interp))),
            "accel": accel_interp,
        }

    manifest_records = []
    sim_counter = 0

    def run_and_save_sim(
        struct_id: str,
        gm_fname: str,
        damage_vector: np.ndarray,
        split_tag: str,
        damage_case_type: str,
        pair_id: int = -1,
        pair_label: str = "none",
    ) -> str:
        nonlocal sim_counter
        cfg = get_structure_config(struct_id)
        frame = MultiStoryFrame(cfg)
        num_stories = cfg.num_stories
        num_elements = frame.num_elements

        # Extract continuous physical graph features from the pristine, undamaged frame
        graph_feats = extract_structural_graph_features(frame)

        gm_data = parsed_gms[gm_fname]
        accel = gm_data["accel"]

        # Evaluate S0, S1, S4 observations
        s0_cfg = get_dynamic_sensor_config("S0", num_stories)
        s1_cfg = get_dynamic_sensor_config("S1", num_stories)
        s4_cfg = get_dynamic_sensor_config("S4", num_stories)

        res_s0 = TopologyObservationMap.evaluate(frame, damage_vector, accel, target_dt, s0_cfg)
        res_s1 = TopologyObservationMap.evaluate(frame, damage_vector, accel, target_dt, s1_cfg)
        res_s4 = TopologyObservationMap.evaluate(frame, damage_vector, accel, target_dt, s4_cfg)

        sim_filename = f"sim_{sim_counter:04d}_{struct_id}_{split_tag}_{gm_data['record_name']}.npz"
        out_path = os.path.join(output_dir, sim_filename)

        np.savez_compressed(
            out_path,
            sim_id=sim_counter,
            struct_id=struct_id,
            split_tag=split_tag,
            damage_case_type=damage_case_type,
            pair_id=pair_id,
            pair_label=pair_label,
            gm_record=gm_data["record_name"],
            dt=target_dt,
            time=sim_times,
            gm_accel=accel,
            damage_vector=damage_vector,
            S0_response=res_s0["Y"],
            S1_response=res_s1["Y"],
            S4_response=res_s4["Y"],
            node_features=graph_feats["node_features"],
            edge_features=graph_feats["edge_features"],
            edge_connectivity=graph_feats["edge_connectivity"],
            adjacency=graph_feats["adjacency"],
            global_scalars=graph_feats["global_scalars"],
            num_stories=num_stories,
            num_nodes=len(graph_feats["node_features"]),
            num_elements=num_elements,
        )

        manifest_records.append({
            "sim_id": sim_counter,
            "filename": sim_filename,
            "struct_id": struct_id,
            "split_tag": split_tag,
            "damage_case_type": damage_case_type,
            "pair_id": pair_id,
            "pair_label": pair_label,
            "gm_record": gm_data["record_name"],
            "max_damage": float(np.max(damage_vector)),
            "num_damaged_elements": int(np.sum(damage_vector > 1e-4)),
        })

        sim_counter += 1
        return out_path

    # ==========================================================================
    # BATCH 1: Protocol P1 SOURCE_A Training Data (90 simulations)
    # ==========================================================================
    print("\n[1/4] Generating Protocol P1 (SOURCE_A) Training Simulations (90 runs)...")
    # 70 standard simulations across RSN0001..RSN0070
    # 20% pristine (14), 80% sparse (56)
    pristine_indices = set(rng.choice(70, size=14, replace=False))
    for idx, gm_file in enumerate(train_gm_files):
        gm_fname = os.path.basename(gm_file)
        if idx in pristine_indices:
            d_vec = np.zeros(9)
            d_type = "pristine"
        else:
            d_vec = sample_random_damage_field(9, mode="sparse", max_damaged_elements=2, severity_range=(0.10, 0.50), rng=rng)
            d_type = "sparse_random"
        run_and_save_sim("SOURCE_A", gm_fname, d_vec, "train_P1", d_type)

    # 20 bilateral training pairs on first 10 earthquakes
    for pair_idx in range(10):
        gm_fname = os.path.basename(train_gm_files[pair_idx])
        # State A: Col 1 (Element 1) = 30%
        dA = np.zeros(9); dA[0] = 0.30
        run_and_save_sim("SOURCE_A", gm_fname, dA, "train_P1_pair", "bilateral_state_A", pair_id=pair_idx, pair_label="A")
        # State B: Col 2 (Element 2) = 30%
        dB = np.zeros(9); dB[1] = 0.30
        run_and_save_sim("SOURCE_A", gm_fname, dB, "train_P1_pair", "bilateral_state_B", pair_id=pair_idx, pair_label="B")

    print(f"  P1 SOURCE_A simulations generated: {sim_counter} runs")

    # ==========================================================================
    # BATCH 2: Protocol P2 Multi-Structure Diversity Training (140 simulations)
    # ==========================================================================
    print("\n[2/4] Generating Protocol P2 Additional B_train Simulations (140 runs)...")
    # B_train_1: 35 runs on RSN0001..RSN0035
    # B_train_2: 35 runs on RSN0036..RSN0070
    # B_train_3: 35 runs on RSN0001..RSN0035
    # B_train_4: 35 runs on RSN0036..RSN0070
    b_allocations = [
        ("B_train_1", train_gm_files[:35]),
        ("B_train_2", train_gm_files[35:70]),
        ("B_train_3", train_gm_files[:35]),
        ("B_train_4", train_gm_files[35:70]),
    ]
    for b_struct, gms in b_allocations:
        p_set = set(rng.choice(len(gms), size=7, replace=False))
        for idx, gm_file in enumerate(gms):
            gm_fname = os.path.basename(gm_file)
            if idx in p_set:
                d_vec = np.zeros(9)
                d_type = "pristine"
            else:
                d_vec = sample_random_damage_field(9, mode="sparse", max_damaged_elements=2, severity_range=(0.10, 0.50), rng=rng)
                d_type = "sparse_random"
            run_and_save_sim(b_struct, gm_fname, d_vec, "train_P2_diversity", d_type)

    print(f"  Total training simulations (P1 + P2 additions): {sim_counter} runs")

    # ==========================================================================
    # BATCH 3: Validation Dataset (60 simulations)
    # ==========================================================================
    print("\n[3/4] Generating Validation Simulations (60 runs across RSN0071..RSN0090)...")
    # 20 standard SOURCE_A simulations
    p_set_val_A = set(rng.choice(20, size=4, replace=False))
    for idx, gm_file in enumerate(val_gm_files):
        gm_fname = os.path.basename(gm_file)
        if idx in p_set_val_A:
            d_vec = np.zeros(9); d_type = "pristine"
        else:
            d_vec = sample_random_damage_field(9, mode="sparse", max_damaged_elements=2, severity_range=(0.10, 0.50), rng=rng)
            d_type = "sparse_random"
        run_and_save_sim("SOURCE_A", gm_fname, d_vec, "val_standard", d_type)

    # 20 standard B_train_1 simulations
    p_set_val_B = set(rng.choice(20, size=4, replace=False))
    for idx, gm_file in enumerate(val_gm_files):
        gm_fname = os.path.basename(gm_file)
        if idx in p_set_val_B:
            d_vec = np.zeros(9); d_type = "pristine"
        else:
            d_vec = sample_random_damage_field(9, mode="sparse", max_damaged_elements=2, severity_range=(0.10, 0.50), rng=rng)
            d_type = "sparse_random"
        run_and_save_sim("B_train_1", gm_fname, d_vec, "val_standard", d_type)

    # 10 bilateral validation pairs on SOURCE_A across first 10 val earthquakes
    for pair_idx in range(10):
        gm_fname = os.path.basename(val_gm_files[pair_idx])
        dA = np.zeros(9); dA[0] = 0.30
        run_and_save_sim("SOURCE_A", gm_fname, dA, "val_pair", "bilateral_state_A", pair_id=pair_idx, pair_label="A")
        dB = np.zeros(9); dB[1] = 0.30
        run_and_save_sim("SOURCE_A", gm_fname, dB, "val_pair", "bilateral_state_B", pair_id=pair_idx, pair_label="B")

    print(f"  Total simulations after validation: {sim_counter} runs")

    # ==========================================================================
    # BATCH 4: Held-Out Test Dataset Across 4 Levels (240 simulation cases = 120 pairs)
    # Evaluates all 30 test earthquakes (RSN0091..RSN0120) across all 4 levels
    # ==========================================================================
    print("\n[4/4] Generating Held-Out Test Datasets (240 runs across all 30 test earthquakes RSN0091..RSN0120)...")
    # Level 1: SOURCE_A on all 30 test earthquakes (30 pairs = 60 runs)
    for pair_idx in range(30):
        gm_fname = os.path.basename(test_gm_files[pair_idx])
        dA = np.zeros(9); dA[0] = 0.30
        run_and_save_sim("SOURCE_A", gm_fname, dA, "test_L1_ID", "bilateral_state_A", pair_id=pair_idx, pair_label="A")
        dB = np.zeros(9); dB[1] = 0.30
        run_and_save_sim("SOURCE_A", gm_fname, dB, "test_L1_ID", "bilateral_state_B", pair_id=pair_idx, pair_label="B")

    # Level 2A: Parametric Interpolation (B_int_1 on 15 GMs, B_int_2 on 15 GMs = 30 pairs = 60 runs)
    for pair_idx in range(15):
        gm_fname = os.path.basename(test_gm_files[pair_idx])
        dA = np.zeros(9); dA[0] = 0.30
        run_and_save_sim("B_int_1", gm_fname, dA, "test_L2A_interp", "bilateral_state_A", pair_id=pair_idx, pair_label="A")
        dB = np.zeros(9); dB[1] = 0.30
        run_and_save_sim("B_int_1", gm_fname, dB, "test_L2A_interp", "bilateral_state_B", pair_id=pair_idx, pair_label="B")

    for pair_idx in range(15, 30):
        gm_fname = os.path.basename(test_gm_files[pair_idx])
        dA = np.zeros(9); dA[0] = 0.30
        run_and_save_sim("B_int_2", gm_fname, dA, "test_L2A_interp", "bilateral_state_A", pair_id=pair_idx, pair_label="A")
        dB = np.zeros(9); dB[1] = 0.30
        run_and_save_sim("B_int_2", gm_fname, dB, "test_L2A_interp", "bilateral_state_B", pair_id=pair_idx, pair_label="B")

    # Level 2B: Parametric Extrapolation (B_ext_soft on 15 GMs, B_ext_stiff on 15 GMs = 30 pairs = 60 runs)
    for pair_idx in range(15):
        gm_fname = os.path.basename(test_gm_files[pair_idx])
        dA = np.zeros(9); dA[0] = 0.30
        run_and_save_sim("B_ext_soft", gm_fname, dA, "test_L2B_extrap", "bilateral_state_A", pair_id=pair_idx, pair_label="A")
        dB = np.zeros(9); dB[1] = 0.30
        run_and_save_sim("B_ext_soft", gm_fname, dB, "test_L2B_extrap", "bilateral_state_B", pair_id=pair_idx, pair_label="B")

    for pair_idx in range(15, 30):
        gm_fname = os.path.basename(test_gm_files[pair_idx])
        dA = np.zeros(9); dA[0] = 0.30
        run_and_save_sim("B_ext_stiff", gm_fname, dA, "test_L2B_extrap", "bilateral_state_A", pair_id=pair_idx, pair_label="A")
        dB = np.zeros(9); dB[1] = 0.30
        run_and_save_sim("B_ext_stiff", gm_fname, dB, "test_L2B_extrap", "bilateral_state_B", pair_id=pair_idx, pair_label="B")

    # Level 3: Structural Topological OOD (C_4story on all 30 test GMs = 30 pairs = 60 runs)
    for pair_idx in range(30):
        gm_fname = os.path.basename(test_gm_files[pair_idx])
        dA = np.zeros(12); dA[0] = 0.30  # Element 1 (Left Col 1) = 30%
        run_and_save_sim("C_4story", gm_fname, dA, "test_L3_topology", "bilateral_state_A", pair_id=pair_idx, pair_label="A")
        dB = np.zeros(12); dB[1] = 0.30  # Element 2 (Right Col 1) = 30%
        run_and_save_sim("C_4story", gm_fname, dB, "test_L3_topology", "bilateral_state_B", pair_id=pair_idx, pair_label="B")

    total_duration = time.time() - t0_all
    print("\n" + "=" * 80)
    print(f"✅ PHASE 7.2 DATASET GENERATION COMPLETE: {sim_counter} TOTAL RUNS GENERATED in {total_duration:.2f}s")
    print("=" * 80)

    # Save complete dataset manifest
    manifest_path = "results/phase7/statistics/dataset_integrity_audit.json"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_simulations_generated": sim_counter,
                "target_dt_seconds": target_dt,
                "duration_seconds": 10.0,
                "num_timesteps": n_steps,
                "partitions": {
                    "train_P1_basis": 90,
                    "train_P2_diversity": 140,
                    "validation_total": 60,
                    "test_total": 120,
                },
                "disjointness_check": "Train ∩ Val = Train ∩ Test = Val ∩ Test = ∅ (PASS)",
                "records": manifest_records,
            },
            f,
            indent=2,
        )

    return {"sim_count": sim_counter, "duration": total_duration, "manifest_path": manifest_path}


if __name__ == "__main__":
    generate_phase7_dataset()
