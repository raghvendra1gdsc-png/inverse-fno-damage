"""
PyTorch Dataset and DataLoader for Phase 7 Cross-Structure Generalization.

Module: src.ml.phase7_dataset
Author: SeismoFNO Research Team
Context: Phase 7.2 Dataset Formatting & PyTorch Pipeline
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple

from src.structure_variants import get_dynamic_symmetry_permutations


_NPZ_CACHE: Dict[str, Dict[str, Any]] = {}


class Phase7SimulationDataset(Dataset):
    """
    Loads Phase 7 simulation .npz files for standard (single-structure) training and validation.
    """

    def __init__(
        self,
        files: List[str],
        sensor_modality: str = "S1",
    ):
        self.files = sorted(files)
        self.sensor_modality = sensor_modality.upper()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        if path not in _NPZ_CACHE:
            _NPZ_CACHE[path] = dict(np.load(path))
        data = _NPZ_CACHE[path]

        num_stories = int(data["num_stories"])
        num_nodes = int(data["num_nodes"])
        num_elements = int(data["num_elements"])
        gm_accel = data["gm_accel"]  # (1000,)
        T = len(gm_accel)

        # 1. Global Signals [5, T] (Floor horizontal accels + ground accel)
        S0 = data["S0_response"]  # [num_stories, 1000]
        global_signals = np.zeros((5, T), dtype=np.float32)
        global_signals[:num_stories, :] = S0
        global_signals[-1, :] = gm_accel

        # 2. Node Signals [num_nodes, 4, T]
        # Channels: [a_x, a_y, strain, a_g]
        node_signals = np.zeros((num_nodes, 4, T), dtype=np.float32)
        # Ground nodes (0, 1)
        node_signals[0, 0, :] = gm_accel
        node_signals[0, 3, :] = gm_accel
        node_signals[1, 0, :] = gm_accel
        node_signals[1, 3, :] = gm_accel

        if self.sensor_modality == "S0":
            # Only floor horizontal accelerations
            for s in range(num_stories):
                fl_acc = S0[s]
                node_signals[2 * (s + 1), 0, :] = fl_acc
                node_signals[2 * (s + 1) + 1, 0, :] = fl_acc
                node_signals[2 * (s + 1), 3, :] = gm_accel
                node_signals[2 * (s + 1) + 1, 3, :] = gm_accel

        elif self.sensor_modality == "S1":
            # S1 has horizontal (num_stories) + vertical (2 * num_stories)
            S1 = data["S1_response"]  # [3 * num_stories, 1000]
            for s in range(num_stories):
                fl_horiz = S1[s]
                vert_left = S1[num_stories + 2 * s]
                vert_right = S1[num_stories + 2 * s + 1]

                nl = 2 * (s + 1)
                nr = 2 * (s + 1) + 1
                node_signals[nl, 0, :] = fl_horiz
                node_signals[nl, 1, :] = vert_left
                node_signals[nl, 3, :] = gm_accel

                node_signals[nr, 0, :] = fl_horiz
                node_signals[nr, 1, :] = vert_right
                node_signals[nr, 3, :] = gm_accel

        elif self.sensor_modality == "S4":
            # Multimodal: S4 has 6 * num_stories channels
            S4 = data["S4_response"]  # [6 * num_stories, 1000]
            for s in range(num_stories):
                fl_horiz = S4[s]
                vl = S4[num_stories + 2 * s]
                vr = S4[num_stories + 2 * s + 1]
                eps_l = S4[3 * num_stories + s]
                eps_r = S4[4 * num_stories + s]

                nl = 2 * (s + 1)
                nr = 2 * (s + 1) + 1
                node_signals[nl, 0, :] = fl_horiz
                node_signals[nl, 1, :] = vl
                node_signals[nl, 2, :] = eps_l
                node_signals[nl, 3, :] = gm_accel

                node_signals[nr, 0, :] = fl_horiz
                node_signals[nr, 1, :] = vr
                node_signals[nr, 2, :] = eps_r
                node_signals[nr, 3, :] = gm_accel

        # Downsample along time by 4 for high efficiency (T=250, dt=0.04s, Nyquist=12.5Hz)
        global_signals_sub = global_signals[:, ::4]
        node_signals_sub = node_signals[:, :, ::4]

        # Permutations
        n_perm, e_perm = get_dynamic_symmetry_permutations(num_stories)

        return {
            "node_signals": torch.from_numpy(node_signals_sub).float(),
            "global_signals": torch.from_numpy(global_signals_sub).float(),
            "node_features": torch.from_numpy(data["node_features"]).float(),
            "edge_features": torch.from_numpy(data["edge_features"]).float(),
            "edge_connectivity": torch.from_numpy(data["edge_connectivity"]).long(),
            "adjacency": torch.from_numpy(data["adjacency"]).float(),
            "global_scalars": torch.from_numpy(data["global_scalars"]).float(),
            "damage": torch.from_numpy(data["damage_vector"]).float(),
            "node_perm": torch.tensor(n_perm, dtype=torch.long),
            "edge_perm": torch.tensor(e_perm, dtype=torch.long),
            "struct_id": str(data["struct_id"]),
            "gm_record": str(data["gm_record"]),
            "split_tag": str(data["split_tag"]),
            "damage_case_type": str(data["damage_case_type"]),
            "pair_id": int(data["pair_id"]),
            "pair_label": str(data["pair_label"]),
        }


class Phase7BilateralPairDataset(Dataset):
    """
    Pairs matched bilateral damage simulations (State A and State B) under identical ground motion.
    """

    def __init__(
        self,
        files: List[str],
        sensor_modality: str = "S1",
    ):
        self.sensor_modality = sensor_modality.upper()
        # Group files into pairs: (struct_id, gm_record, pair_id)
        grouped: Dict[Tuple[str, str, int], Dict[str, str]] = {}
        for f in files:
            d = np.load(f)
            p_id = int(d["pair_id"])
            if p_id >= 0:
                key = (str(d["struct_id"]), str(d["gm_record"]), p_id)
                if key not in grouped:
                    grouped[key] = {}
                grouped[key][str(d["pair_label"])] = f

        self.pairs = []
        for key, p_files in grouped.items():
            if "A" in p_files and "B" in p_files:
                self.pairs.append((p_files["A"], p_files["B"]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path_A, path_B = self.pairs[idx]
        dset = Phase7SimulationDataset([path_A, path_B], sensor_modality=self.sensor_modality)
        item_A = dset[0]
        item_B = dset[1]

        num_elements = len(item_A["damage"])
        # Canonical bilateral unit vector
        v_AB = np.zeros(num_elements, dtype=np.float32)
        v_AB[0] = 1.0 / np.sqrt(2.0)
        v_AB[1] = -1.0 / np.sqrt(2.0)

        return {
            "A": item_A,
            "B": item_B,
            "v_AB": torch.from_numpy(v_AB).float(),
            "struct_id": item_A["struct_id"],
            "gm_record": item_A["gm_record"],
            "pair_id": item_A["pair_id"],
        }
