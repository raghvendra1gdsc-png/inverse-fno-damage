"""
Systematic Ablation Study for Phase 3 Regularization Terms.

Evaluates on the exact same held-out test split:
  1. Naive MSE Baseline (Phase 2)
  2. L1 Sparsity Prior Only (lambda_sparse > 0, lambda_tv = 0, lambda_cycle = 0)
  3. Total Variation Prior Only (lambda_sparse = 0, lambda_tv > 0, lambda_cycle = 0)
  4. Forward Cycle-Consistency Only (lambda_cycle > 0, lambda_sparse = 0, lambda_tv = 0)
  5. Combined (L1 Sparsity + Cycle-Consistency)
"""

import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import train_regularized_inverse_fno


def run_ablations():
    experiments = [
        {
            "name": "L1_Sparsity_Only",
            "desc": "L1 Sparsity Prior Only (lambda_sparse=0.05)",
            "save_path": "models/inverse_fno_l1_only.pt",
            "lambda_data": 1.0,
            "lambda_sparse": 0.05,
            "lambda_tv": 0.0,
            "lambda_cycle": 0.0,
        },
        {
            "name": "TV_Prior_Only",
            "desc": "Graph Total Variation Only (lambda_tv=0.05)",
            "save_path": "models/inverse_fno_tv_only.pt",
            "lambda_data": 1.0,
            "lambda_sparse": 0.0,
            "lambda_tv": 0.05,
            "lambda_cycle": 0.0,
        },
        {
            "name": "Cycle_Consistency_Only",
            "desc": "Forward Cycle-Consistency Only (lambda_cycle=0.02)",
            "save_path": "models/inverse_fno_cycle_only.pt",
            "lambda_data": 1.0,
            "lambda_sparse": 0.0,
            "lambda_tv": 0.0,
            "lambda_cycle": 0.02,
        },
        {
            "name": "Combined_L1_and_Cycle",
            "desc": "Balanced L1 Sparsity + Cycle (sparse=0.02, cycle=0.01)",
            "save_path": "models/inverse_fno_combined.pt",
            "lambda_data": 1.0,
            "lambda_sparse": 0.02,
            "lambda_tv": 0.0,
            "lambda_cycle": 0.01,
        },
    ]

    all_results = {}
    
    # Load Naive MSE Baseline
    with open("results/inverse_fno_naive_metrics.json", "r") as f:
        all_results["Naive_MSE_Baseline"] = json.load(f)

    for exp in experiments:
        print("\n" + "=" * 80)
        print(f"RUNNING ABLATION: {exp['name']} ({exp['desc']})")
        print("=" * 80)

        _, metrics, _ = train_regularized_inverse_fno(
            save_path=exp["save_path"],
            lambda_data=exp["lambda_data"],
            lambda_sparse=exp["lambda_sparse"],
            lambda_tv=exp["lambda_tv"],
            lambda_cycle=exp["lambda_cycle"],
            epochs=35,
            batch_size=16,
            learning_rate=1.5e-3,
            verbose=False,
        )
        all_results[exp["name"]] = metrics
        print(f"Top-1 Acc: {metrics['localization']['top1_accuracy_pct']:.1f}% | "
              f"Top-2 Acc: {metrics['localization']['top2_accuracy_pct']:.1f}% | "
              f"Ghost Dmg: {metrics['severity']['mean_false_positive_damage_on_intact_elements']:.4f} | "
              f"Severity Err: {metrics['severity']['mean_severity_error_on_damaged_elements']:.4f}")

    # Save summary table
    out_file = "results/regularization_ablation_summary.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll ablation results saved to: {out_file}")


if __name__ == "__main__":
    run_ablations()
