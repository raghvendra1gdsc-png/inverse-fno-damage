"""
PyTorch Dataset Loader for Multimodal Sensor Configurations (S0, S1, S2, S4).

Module: src.ml.dataset
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from src.observability import SensorConfiguration


class MultimodalDamageDataset(Dataset):
    """
    Dataset loader for augmented multimodal sensor runs in data/phase6_multimodal_runs/.
    Supports configurations S0, S1, S2, and S4. Normalizes observations strictly
    using training set statistics to avoid test set leakage.
    """

    def __init__(
        self,
        file_list: List[str],
        sensor_config: SensorConfiguration,
        stats_path: str = "data/phase6_multimodal_runs/phase6_normalization_stats.json",
        is_train: bool = False,
    ):
        self.file_list = sorted(file_list)
        self.sensor_config = sensor_config
        self.cfg_key = f"{sensor_config.name}_Y"
        self.is_train = is_train

        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        # Preload records in memory for fast training
        self.records = []
        ch_mean = np.array(self.stats[self.cfg_key]["mean"], dtype=np.float32)[:, np.newaxis]
        ch_std = np.array(self.stats[self.cfg_key]["std"], dtype=np.float32)[:, np.newaxis]

        s0_mean = np.array(self.stats["S0_Y"]["mean"], dtype=np.float32)[:, np.newaxis]
        s0_std = np.array(self.stats["S0_Y"]["std"], dtype=np.float32)[:, np.newaxis]

        gm_mean = float(self.stats["ground_accel"]["mean"])
        gm_std = float(self.stats["ground_accel"]["std"])

        for fpath in self.file_list:
            data = np.load(fpath)
            raw_y = data[self.cfg_key]        # [C, T]
            raw_s0 = data["S0_Y"]             # [3, T]
            raw_gm = data["ground_accel"]     # [T]
            d_vec = data["damage_field"]      # [9]
            t_vec = data["time"]              # [T]

            # Normalize signals
            y_norm = (raw_y - ch_mean) / ch_std
            s0_norm = (raw_s0 - s0_mean) / s0_std
            gm_norm = (raw_gm - gm_mean) / gm_std

            self.records.append({
                "filename": os.path.basename(fpath),
                "Y": torch.from_numpy(y_norm).float(),
                "Y_global": torch.from_numpy(s0_norm).float(),
                "ground_accel": torch.from_numpy(gm_norm[np.newaxis, :]).float(),  # [1, T]
                "damage": torch.from_numpy(d_vec).float(),                         # [9]
                "support": torch.from_numpy((d_vec > 0.01).astype(np.float32)),     # [9]
                "time": torch.from_numpy(t_vec).float(),
                "gm_record": str(data["gm_record"]),
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]
