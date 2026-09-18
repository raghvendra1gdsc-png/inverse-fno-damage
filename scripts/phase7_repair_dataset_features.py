"""
Phase 7 Dataset Feature Repair Script.

Module: scripts.phase7_repair_dataset_features
Context: Replaces contaminated post-damage edge features with pristine nominal features
         across all 530 Phase 7 simulation files.
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.damage_injection import MultiStoryFrame
from src.structure_variants import get_structure_config
from src.structure_features import extract_structural_graph_features


def repair_all_simulation_features(sim_dir: str = "data/phase7_simulations"):
    files = sorted(glob.glob(os.path.join(sim_dir, "*.npz")))
    assert len(files) == 530, f"Expected 530 files in {sim_dir}, found {len(files)}"

    print(f"🔧 Starting feature repair across {len(files)} simulation files in {sim_dir}...")

    # Pre-cache clean graph features for all 10 structural configurations
    struct_ids = ["SOURCE_A", "B_train_1", "B_train_2", "B_train_3", "B_train_4",
                  "B_int_1", "B_int_2", "B_ext_soft", "B_ext_stiff", "C_4story"]
    clean_cache = {}
    for sid in struct_ids:
        cfg = get_structure_config(sid)
        frame = MultiStoryFrame(cfg)
        feats = extract_structural_graph_features(frame)
        clean_cache[sid] = feats
        # Verify clean cache features have no damage dependence
        assert np.allclose(feats["edge_features"][:, 3], 1.0), f"Clean cache for {sid} has non-unit E_norm!"

    print(f"✅ Pre-cached clean structural features for all {len(struct_ids)} structural configurations.")

    modified_count = 0
    leaked_before_count = 0

    for idx, f in enumerate(files):
        data = dict(np.load(f))
        sid = str(data["struct_id"])
        dmg = data["damage_vector"]
        old_ef = data["edge_features"]

        # Check if old file was leaked
        if np.max(dmg) > 0.01 and np.allclose(old_ef[:, 3], 1.0 - dmg):
            leaked_before_count += 1

        clean_feats = clean_cache[sid]

        # Overwrite structural features with pristine versions
        data["edge_features"] = clean_feats["edge_features"]
        data["node_features"] = clean_feats["node_features"]
        data["edge_connectivity"] = clean_feats["edge_connectivity"]
        data["adjacency"] = clean_feats["adjacency"]
        data["global_scalars"] = clean_feats["global_scalars"]

        # Save back atomically
        np.savez_compressed(f, **data)
        modified_count += 1

        if (idx + 1) % 100 == 0 or idx == len(files) - 1:
            print(f"  Processed {idx + 1}/{len(files)} files...")

    print(f"\n🎉 REPAIR COMPLETE:")
    print(f"  Total files updated: {modified_count}")
    print(f"  Files with confirmed damage leakage repaired: {leaked_before_count}")

    # Final Verification Pass across all 530 files
    print("\n🔍 Running verification pass across all 530 repaired files...")
    leaked_after_count = 0
    for f in files:
        data = np.load(f)
        ef = data["edge_features"]
        d = data["damage_vector"]
        if np.max(d) > 0.01 and np.allclose(ef[:, 3], 1.0 - d):
            leaked_after_count += 1
        assert np.allclose(ef[:, 3], 1.0), f"File {f} does not have nominal E_norm = 1.0!"

    assert leaked_after_count == 0, f"Critical error: {leaked_after_count} files still contain leaked damage!"
    print(f"✅ Zero feature leakage confirmed across all 530 files in {sim_dir}!")


if __name__ == "__main__":
    repair_all_simulation_features()
