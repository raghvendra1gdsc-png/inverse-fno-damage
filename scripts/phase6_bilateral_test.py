"""
Canonical Bilateral Damage Benchmark and Fisher Correlation Test.

Script: scripts/phase6_bilateral_test.py
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team

Directly tests trained neural models on Canonical Damage State A (Left Col 1 = 30%)
and State B (Right Col 1 = 30%) across the 10 earthquake ground motions. Evaluates
whether asymmetric sensing (S1, S2, S4) enables the symmetry-aware neural operator
to resolve the bilateral ambiguity, and directly links empirical attribution accuracy
to Phase 5 directional Fisher sensitivity sqrt(I_AB).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import glob
import numpy as np
import torch

from src.damage_injection import MultiStoryFrame, FrameConfig
from src.observability import SensorConfiguration, ForwardObservationMap, BilateralSymmetry
from src.graph.structural_graph import StructuralGraph
from src.ml.gfno import DualStreamGFNO


def run_bilateral_benchmark():
    print("=" * 80)
    print("⚖️  [PHASE 6] CANONICAL BILATERAL DAMAGE BENCHMARK (STATE A VS. STATE B)")
    print("=" * 80)

    device = "cpu"
    graph = StructuralGraph()
    frame = MultiStoryFrame(FrameConfig())

    d_A, d_B = BilateralSymmetry.get_canonical_states()  # State A: Ele 1=0.30; State B: Ele 2=0.30

    from src.damage_injection import parse_at2_ground_motion
    sim_dt = 0.01
    sim_dur = 10.0
    n_steps = int(sim_dur / sim_dt)
    t_target = np.arange(n_steps) * sim_dt

    gm_files = sorted(glob.glob("data/raw_ground_motions/*.AT2"))[:10]
    assert len(gm_files) == 10, f"Expected 10 ground motions, got {len(gm_files)}"

    records = []
    for f in gm_files:
        dt_n, acc_n, meta = parse_at2_ground_motion(f)
        t_native = np.arange(len(acc_n)) * dt_n
        gm_interp = np.interp(t_target, t_native, acc_n)
        records.append({
            "name": meta["record_name"],
            "gm": gm_interp,
        })
    print(f"Testing on {len(records)} earthquake ground motions.")

    configs = [
        ("S0", SensorConfiguration.create_S0(), 5.64),
        ("S1", SensorConfiguration.create_S1(), 1829.61),
        ("S2", SensorConfiguration.create_S2(), 1344.18),
        ("S4", SensorConfiguration.create_S4(), 2270.51),
    ]

    # Load training normalization statistics
    stats_file = "data/phase6_multimodal_runs/phase6_normalization_stats.json"
    with open(stats_file, "r") as f:
        norm_stats = json.load(f)

    models_to_test = [
        ("Baseline_InverseFNO", False, False, False, False),
        ("DualStream_GFNO",     True,  True,  False, False),
        ("Symmetry_GFNO",       True,  True,  True,  False),
        ("Proposed_GFNO",       True,  True,  True,  True),
    ]

    results = {}

    for cfg_code, cfg_obj, fisher_sqrt in configs:
        print(f"\nEvaluating Sensor Configuration [{cfg_code}] (Fisher Sensitivity sqrt(I_AB) = {fisher_sqrt:.1f}):")
        results[cfg_code] = {
            "fisher_sensitivity_sqrt": fisher_sqrt,
            "models": {},
        }

        cfg_key = f"{cfg_code}_Y"
        ch_mean = np.array(norm_stats[cfg_key]["mean"], dtype=np.float32)[:, np.newaxis]
        ch_std = np.array(norm_stats[cfg_key]["std"], dtype=np.float32)[:, np.newaxis]
        s0_mean = np.array(norm_stats["S0_Y"]["mean"], dtype=np.float32)[:, np.newaxis]
        s0_std = np.array(norm_stats["S0_Y"]["std"], dtype=np.float32)[:, np.newaxis]
        gm_mean = float(norm_stats["ground_accel"]["mean"])
        gm_std = float(norm_stats["ground_accel"]["std"])

        for m_name, dual, gr, sym, hier in models_to_test:
            ckpt_path = f"models/phase6/{cfg_code}_{m_name}_best.pt"
            if not os.path.exists(ckpt_path):
                continue

            model = DualStreamGFNO(
                sensor_config=cfg_obj,
                structural_graph=graph,
                use_dual_stream=dual,
                use_graph=gr,
                use_symmetry=sym,
                use_hierarchical=hier,
                width_temporal=32,
                modes_temporal=12,
                n_temporal_layers=2,
                width_graph=32,
                n_graph_layers=2,
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            correct_A = 0
            correct_B = 0
            confusion = np.zeros((2, 2), dtype=int)  # [[AA, AB], [BA, BB]]
            d_preds_A = []
            d_preds_B = []

            for rec in records:
                gm = rec["gm"]
                dt = 0.01

                # Generate OpenSees response for State A
                res_A = ForwardObservationMap.evaluate(frame, d_A, gm, dt, cfg_obj)
                res_A_s0 = ForwardObservationMap.evaluate(frame, d_A, gm, dt, SensorConfiguration.create_S0())

                # Generate OpenSees response for State B
                res_B = ForwardObservationMap.evaluate(frame, d_B, gm, dt, cfg_obj)
                res_B_s0 = ForwardObservationMap.evaluate(frame, d_B, gm, dt, SensorConfiguration.create_S0())

                # Normalize and decimate by 4x
                Y_A = ((res_A["Y"] - ch_mean) / ch_std)[:, ::4]
                Y_B = ((res_B["Y"] - ch_mean) / ch_std)[:, ::4]
                S0_A = ((res_A_s0["Y"] - s0_mean) / s0_std)[:, ::4]
                S0_B = ((res_B_s0["Y"] - s0_mean) / s0_std)[:, ::4]
                gm_norm = ((gm - gm_mean) / gm_std)[::4]

                t_Y_A = torch.from_numpy(Y_A).float().unsqueeze(0)
                t_Y_B = torch.from_numpy(Y_B).float().unsqueeze(0)
                t_S0_A = torch.from_numpy(S0_A).float().unsqueeze(0)
                t_S0_B = torch.from_numpy(S0_B).float().unsqueeze(0)
                t_gm = torch.from_numpy(gm_norm).float().unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    out_A = model(t_Y_A, ground_accel=t_gm, Y_global=t_S0_A)
                    out_B = model(t_Y_B, ground_accel=t_gm, Y_global=t_S0_B)

                pred_A = out_A["damage_pred"].squeeze(0).numpy()
                pred_B = out_B["damage_pred"].squeeze(0).numpy()
                d_preds_A.append(pred_A)
                d_preds_B.append(pred_B)

                # Strict Attribution:
                # State A (True Col 1 damaged): requires pred_A[0] > pred_A[1] by margin
                # State B (True Col 2 damaged): requires pred_B[1] > pred_B[0] by margin
                tol = 1e-4
                diff_A = pred_A[0] - pred_A[1]
                diff_B = pred_B[1] - pred_B[0]

                if diff_A > tol:
                    attr_A = 0  # correctly Col 1
                    correct_A += 1
                elif diff_A < -tol:
                    attr_A = 1  # incorrectly Col 2
                else:
                    attr_A = 1  # ambiguous/tied, count as unclassified/failure

                if diff_B > tol:
                    attr_B = 1  # correctly Col 2
                    correct_B += 1
                elif diff_B < -tol:
                    attr_B = 0  # incorrectly Col 1
                else:
                    attr_B = 0  # ambiguous/tied

                confusion[0, attr_A] += 1
                confusion[1, attr_B] += 1

            total_acc = float((correct_A + correct_B) / (2 * len(gm_files)))
            avg_pred_A = np.mean(d_preds_A, axis=0).tolist()
            avg_pred_B = np.mean(d_preds_B, axis=0).tolist()

            results[cfg_code]["models"][m_name] = {
                "accuracy": total_acc,
                "confusion_matrix": confusion.tolist(),
                "correct_A": correct_A,
                "correct_B": correct_B,
                "avg_pred_A": avg_pred_A,
                "avg_pred_B": avg_pred_B,
            }

            print(f"  Model [{m_name:20s}] -> A/B Binary Accuracy: {total_acc*100:5.1f}% | Confusion: {confusion.tolist()}")

    # Save results
    out_file = "results/phase6/bilateral_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Bilateral benchmark results saved to {out_file}")


if __name__ == "__main__":
    run_bilateral_benchmark()
