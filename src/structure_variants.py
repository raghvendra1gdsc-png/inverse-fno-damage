"""
Structure Variants and Topology Configurations for Cross-Structure Generalization.

Module: src.structure_variants
Context: Phase 7.1 Cross-Structure Generalization & Transferability
Author: SeismoFNO Research Team
"""

from typing import Dict, Any, Tuple, List
import numpy as np
from src.damage_injection import FrameConfig, MultiStoryFrame


# Reference Baseline Constants (SOURCE_A)
REF_FLOOR_MASS: float = 10000.0  # kg
REF_I_COL: float = 1.0e-4        # m^4
REF_I_BEAM: float = 3.0e-4       # m^4
REF_A_COL: float = 0.010         # m^2
REF_A_BEAM: float = 0.015        # m^2
REF_STORY_HEIGHT: float = 3.0    # m
REF_BAY_WIDTH: float = 6.0       # m
REF_E: float = 2.0e11            # Pa


# Frozen Structure Specifications Dictionary
STRUCTURE_SPECS: Dict[str, Dict[str, Any]] = {
    "SOURCE_A": {
        "num_stories": 3,
        "num_bays": 1,
        "mu_m": 1.00,
        "mu_k": 1.00,
        "expected_f1": 2.1856,
        "expected_f2": 6.8500,
        "expected_f3": 11.4039,
        "role": "Source Baseline (Train / Val / Level 1 Test)",
        "num_nodes": 8,
        "num_elements": 9,
    },
    "B_train_1": {
        "num_stories": 3,
        "num_bays": 1,
        "mu_m": 0.90,
        "mu_k": 0.90,
        "expected_f1": 2.1856,
        "expected_f2": 6.8500,
        "expected_f3": 11.4039,
        "role": "Multi-Structure Training Corner 1",
        "num_nodes": 8,
        "num_elements": 9,
    },
    "B_train_2": {
        "num_stories": 3,
        "num_bays": 1,
        "mu_m": 1.10,
        "mu_k": 1.10,
        "expected_f1": 2.1856,
        "expected_f2": 6.8500,
        "expected_f3": 11.4039,
        "role": "Multi-Structure Training Corner 2",
        "num_nodes": 8,
        "num_elements": 9,
    },
    "B_train_3": {
        "num_stories": 3,
        "num_bays": 1,
        "mu_m": 0.90,
        "mu_k": 1.10,
        "expected_f1": 2.4163,
        "expected_f2": 7.5729,
        "expected_f3": 12.6075,
        "role": "Multi-Structure Training Corner 3",
        "num_nodes": 8,
        "num_elements": 9,
    },
    "B_train_4": {
        "num_stories": 3,
        "num_bays": 1,
        "mu_m": 1.10,
        "mu_k": 0.90,
        "expected_f1": 1.9770,
        "expected_f2": 6.1960,
        "expected_f3": 10.3152,
        "role": "Multi-Structure Training Corner 4",
        "num_nodes": 8,
        "num_elements": 9,
    },
    "B_int_1": {
        "num_stories": 3,
        "num_bays": 1,
        "mu_m": 0.95,
        "mu_k": 1.05,
        "expected_f1": 2.2978,
        "expected_f2": 7.2015,
        "expected_f3": 11.9891,
        "role": "Level 2A Parametric Interpolation Test (Held-Out)",
        "num_nodes": 8,
        "num_elements": 9,
    },
    "B_int_2": {
        "num_stories": 3,
        "num_bays": 1,
        "mu_m": 1.05,
        "mu_k": 0.95,
        "expected_f1": 2.0789,
        "expected_f2": 6.5156,
        "expected_f3": 10.8472,
        "role": "Level 2A Parametric Interpolation Test (Held-Out)",
        "num_nodes": 8,
        "num_elements": 9,
    },
    "B_ext_soft": {
        "num_stories": 3,
        "num_bays": 1,
        "mu_m": 1.50,
        "mu_k": 0.80,
        "expected_f1": 1.5961,
        "expected_f2": 5.0025,
        "expected_f3": 8.3282,
        "role": "Level 2B Parametric Extrapolation Test (Heavy/Flexible)",
        "num_nodes": 8,
        "num_elements": 9,
    },
    "B_ext_stiff": {
        "num_stories": 3,
        "num_bays": 1,
        "mu_m": 0.65,
        "mu_k": 1.30,
        "expected_f1": 3.0909,
        "expected_f2": 9.6873,
        "expected_f3": 16.1275,
        "role": "Level 2B Parametric Extrapolation Test (Light/Stiff)",
        "num_nodes": 8,
        "num_elements": 9,
    },
    "C_4story": {
        "num_stories": 4,
        "num_bays": 1,
        "mu_m": 1.00,
        "mu_k": 1.00,
        "expected_f1": 1.6492,
        "expected_f2": 5.1392,
        "expected_f3": 8.9111,
        "role": "Level 3 Structural Topological OOD Test (10 Nodes, 12 Members)",
        "num_nodes": 10,
        "num_elements": 12,
    },
}


def get_structure_config(struct_id: str) -> FrameConfig:
    """
    Constructs a FrameConfig object corresponding to the specified structural family.
    """
    if struct_id not in STRUCTURE_SPECS:
        raise KeyError(f"Unknown structure_id '{struct_id}'. Allowed: {list(STRUCTURE_SPECS.keys())}")

    spec = STRUCTURE_SPECS[struct_id]
    mu_m = spec["mu_m"]
    mu_k = spec["mu_k"]
    num_stories = spec["num_stories"]
    num_bays = spec["num_bays"]

    return FrameConfig(
        num_stories=num_stories,
        num_bays=num_bays,
        story_height=REF_STORY_HEIGHT,
        bay_width=REF_BAY_WIDTH,
        E=REF_E,
        A_col=REF_A_COL * mu_k,
        I_col=REF_I_COL * mu_k,
        A_beam=REF_A_BEAM * mu_k,
        I_beam=REF_I_BEAM * mu_k,
        floor_mass=REF_FLOOR_MASS * mu_m,
        coord_transf="Linear",
        lump_mass_vertical=False,
    )


def get_dynamic_symmetry_permutations(num_stories: int) -> Tuple[List[int], List[int]]:
    """
    Dynamically generates 0-indexed reflection permutation arrays for any (N_stories, 1)-bay frame.

    Returns
    -------
    node_perm : List[int]
        Permutation array mapping node index to its mirror counterpart.
    edge_perm : List[int]
        Permutation array mapping edge index to its mirror counterpart.
    """
    # 1. Node Permutation
    # Total nodes = (num_stories + 1) * 2
    # At story s in [0..num_stories]: left node is 2*s, right node is 2*s + 1
    node_perm = []
    for s in range(num_stories + 1):
        left_idx = 2 * s
        right_idx = 2 * s + 1
        node_perm.extend([right_idx, left_idx])

    # 2. Edge Permutation
    # Columns: num_stories * 2 columns.
    # At story s in [0..num_stories - 1]: left col is 2*s, right col is 2*s + 1
    edge_perm = []
    num_cols = num_stories * 2
    for s in range(num_stories):
        left_col = 2 * s
        right_col = 2 * s + 1
        edge_perm.extend([right_col, left_col])

    # Beams: num_stories beams, indexed from num_cols to num_cols + num_stories - 1
    # Each beam spans the single bay and is invariant under reflection
    for b in range(num_stories):
        beam_idx = num_cols + b
        edge_perm.append(beam_idx)

    return node_perm, edge_perm


def extract_structure_frequencies(struct_id: str) -> List[float]:
    """
    Builds the OpenSees finite-element model and extracts fundamental natural frequencies in Hz.
    """
    cfg = get_structure_config(struct_id)
    frame = MultiStoryFrame(cfg)
    frame.build_model()
    eigs = frame.extract_eigenvalues(num_modes=cfg.num_stories)
    freqs = [float(f) for f in eigs["frequencies"]]
    return freqs


def get_dynamic_sensor_config(config_id: str, num_stories: int) -> Dict[str, Any]:
    """
    Constructs channel specifications for any (N_stories, 1)-bay frame.
    Supports 'S0', 'S1', 'S4'.
    """
    from src.observability import SensorChannelSpec, SensorConfiguration

    config_id = config_id.upper()
    channels: List[SensorChannelSpec] = []

    # 1. Horizontal Floor Accelerometers (S0)
    for s in range(1, num_stories + 1):
        left_node = 2 * s + 1
        name = f"H_F{s}" if s < num_stories else "H_RF"
        channels.append(
            SensorChannelSpec(
                channel_id=name,
                channel_type="accel_horiz",
                location=f"Floor {s} (Node {left_node})",
                units="m/s^2",
                node_tag=left_node,
                dof=1,
                story=s,
            )
        )

    if config_id == "S0":
        return SensorConfiguration("S0", f"{num_stories} Horizontal Floor Accelerometers", channels)

    # 2. Vertical Accelerometers (S1)
    vert_channels: List[SensorChannelSpec] = []
    for s in range(1, num_stories + 1):
        n_left = 2 * s + 1
        n_right = 2 * s + 2
        vert_channels.append(
            SensorChannelSpec(
                channel_id=f"V_N{n_left}",
                channel_type="accel_vert",
                location=f"Floor {s} Left (Node {n_left})",
                units="m/s^2",
                node_tag=n_left,
                dof=2,
                story=s,
            )
        )
        vert_channels.append(
            SensorChannelSpec(
                channel_id=f"V_N{n_right}",
                channel_type="accel_vert",
                location=f"Floor {s} Right (Node {n_right})",
                units="m/s^2",
                node_tag=n_right,
                dof=2,
                story=s,
            )
        )

    if config_id == "S1":
        return SensorConfiguration(
            "S1",
            f"{num_stories} Horizontal + {2 * num_stories} Vertical Accelerometers",
            channels + vert_channels,
        )

    # 3. Column Axial Strains (S2) & Rocking (S3) for S4
    strain_channels: List[SensorChannelSpec] = []
    for s in range(1, num_stories + 1):
        c_left = 2 * s - 1
        c_right = 2 * s
        strain_channels.append(
            SensorChannelSpec(
                channel_id=f"Eps_C{c_left}",
                channel_type="strain_col",
                location=f"Col {c_left} Left Story {s}",
                units="strain",
                ele_id=c_left,
                story=s,
            )
        )
        strain_channels.append(
            SensorChannelSpec(
                channel_id=f"Eps_C{c_right}",
                channel_type="strain_col",
                location=f"Col {c_right} Right Story {s}",
                units="strain",
                ele_id=c_right,
                story=s,
            )
        )

    rocking_channels: List[SensorChannelSpec] = []
    for s in range(1, num_stories + 1):
        name = f"Rock_F{s}" if s < num_stories else "Rock_RF"
        rocking_channels.append(
            SensorChannelSpec(
                channel_id=name,
                channel_type="rocking",
                location=f"Story {s} Rocking",
                units="rad/s^2",
                story=s,
            )
        )

    if config_id == "S4":
        return SensorConfiguration(
            "S4",
            f"Comprehensive Multi-Modal ({num_stories}H + {2*num_stories}V + {2*num_stories}Strain + {num_stories}Rock)",
            channels + vert_channels + strain_channels + rocking_channels,
        )

    raise ValueError(f"Unknown config_id '{config_id}'. Supported: S0, S1, S4.")


class TopologyObservationMap:
    """
    Topology-agnostic observation evaluator for (N_stories, 1)-bay frames.
    Maps damage vector and ground acceleration to discrete multichannel observation Y(t).
    """

    @classmethod
    def evaluate(
        cls,
        frame: MultiStoryFrame,
        damage_vector: np.ndarray,
        ground_accel_ms2: np.ndarray,
        dt: float,
        sensor_config: Any,
        damping_ratio: float = 0.03,
    ) -> Dict[str, Any]:
        num_stories = frame.config.num_stories
        L_bay = frame.config.bay_width
        h_story = frame.config.story_height

        # Dynamic Base Channels: all 4 DOFs across all elevated joint pairs
        base_channels = []
        for s in range(1, num_stories + 1):
            n_left = 2 * s + 1
            n_right = 2 * s + 2
            base_channels.extend([
                (n_left, 1), (n_right, 1),
                (n_left, 2), (n_right, 2),
            ])

        frame.build_model(damage_vector)
        raw_res = frame.run_dynamic_analysis(
            ground_accel_ms2=ground_accel_ms2,
            dt=dt,
            sensor_channels=base_channels,
            damping_ratio=damping_ratio,
        )

        time = raw_res["time"]
        n_steps = len(time)
        ch_map = {ch: idx for idx, ch in enumerate(base_channels)}

        tot_acc = raw_res["sensor_total_accel"]
        disps = raw_res["sensor_disp"]

        C = sensor_config.num_channels
        Y = np.zeros((C, n_steps), dtype=np.float64)
        channel_names = []

        for c_idx, ch_spec in enumerate(sensor_config.channels):
            channel_names.append(ch_spec.channel_id)

            if ch_spec.channel_type == "accel_horiz":
                row = ch_map[(ch_spec.node_tag, 1)]
                Y[c_idx, :] = tot_acc[row, :]

            elif ch_spec.channel_type == "accel_vert":
                row = ch_map[(ch_spec.node_tag, 2)]
                Y[c_idx, :] = tot_acc[row, :]

            elif ch_spec.channel_type == "strain_col":
                ele = ch_spec.ele_id
                # ele is odd -> left col at story (ele + 1) // 2
                # ele is even -> right col at story ele // 2
                is_left = (ele % 2 == 1)
                story = (ele + 1) // 2 if is_left else ele // 2

                if is_left:
                    n_top = 2 * story + 1
                    u_top = disps[ch_map[(n_top, 2)], :]
                    u_bot = 0.0 if story == 1 else disps[ch_map[(2 * (story - 1) + 1, 2)], :]
                else:
                    n_top = 2 * story + 2
                    u_top = disps[ch_map[(n_top, 2)], :]
                    u_bot = 0.0 if story == 1 else disps[ch_map[(2 * (story - 1) + 2, 2)], :]

                Y[c_idx, :] = (u_top - u_bot) / h_story

            elif ch_spec.channel_type == "rocking":
                story = ch_spec.story
                n_left = 2 * story + 1
                n_right = 2 * story + 2
                a_left = tot_acc[ch_map[(n_left, 2)], :]
                a_right = tot_acc[ch_map[(n_right, 2)], :]
                Y[c_idx, :] = (a_right - a_left) / L_bay

        return {
            "Y": Y,
            "y_vec": Y.flatten(),
            "time": time,
            "channel_names": channel_names,
        }


def compute_topology_damage_jacobian(
    frame: MultiStoryFrame,
    damage_ref: np.ndarray,
    ground_accel_ms2: np.ndarray,
    dt: float,
    sensor_config: Any,
    h: float = 1e-3,
    damping_ratio: float = 0.03,
) -> np.ndarray:
    """
    Computes finite-difference damage Jacobian for any frame topology.
    J_d has shape (N_obs, num_elements).
    """
    d_ref = np.asarray(damage_ref, dtype=np.float64)
    num_elements = frame.num_elements
    assert len(d_ref) == num_elements

    res_base = TopologyObservationMap.evaluate(
        frame=frame,
        damage_vector=d_ref,
        ground_accel_ms2=ground_accel_ms2,
        dt=dt,
        sensor_config=sensor_config,
        damping_ratio=damping_ratio,
    )
    N_obs = len(res_base["y_vec"])
    J_d = np.zeros((N_obs, num_elements), dtype=np.float64)

    for e in range(num_elements):
        d_plus = d_ref.copy()
        d_plus[e] = np.clip(d_plus[e] + h, 0.0, 0.95)
        res_plus = TopologyObservationMap.evaluate(
            frame=frame,
            damage_vector=d_plus,
            ground_accel_ms2=ground_accel_ms2,
            dt=dt,
            sensor_config=sensor_config,
            damping_ratio=damping_ratio,
        )

        d_minus = d_ref.copy()
        d_minus[e] = np.clip(d_minus[e] - h, 0.0, 0.95)
        res_minus = TopologyObservationMap.evaluate(
            frame=frame,
            damage_vector=d_minus,
            ground_accel_ms2=ground_accel_ms2,
            dt=dt,
            sensor_config=sensor_config,
            damping_ratio=damping_ratio,
        )

        actual_h = float(d_plus[e] - d_minus[e])
        J_d[:, e] = (res_plus["y_vec"] - res_minus["y_vec"]) / actual_h

    return J_d


def compute_topology_noise_normalized_observability(
    J_d: np.ndarray,
    Y_ref: np.ndarray,
    noise_ratio: float = 0.02,
    v_null: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Computes noise-whitened Jacobian J_R = R^(-1/2) J_d and SVD spectrum.
    R is diagonal with channel-wise variance sigma_c^2 = (noise_ratio * RMS(Y_c))^2.
    """
    C, T = Y_ref.shape
    rms_per_channel = np.sqrt(np.mean(Y_ref**2, axis=1))
    rms_per_channel = np.maximum(rms_per_channel, 1e-8)
    sigma_per_channel = noise_ratio * rms_per_channel
    noise_std_vec = np.repeat(sigma_per_channel, T)

    J_R = J_d / noise_std_vec[:, np.newaxis]
    I_F = J_R.T @ J_R

    # SVD of J_R
    U, s, Vt = np.linalg.svd(J_R, full_matrices=False)
    V = Vt.T

    cond_num = float(s[0] / s[-1]) if s[-1] > 1e-15 else float("inf")

    res = {
        "singular_values": s,
        "sigma_max": float(s[0]),
        "sigma_min": float(s[-1]),
        "condition_number": cond_num,
        "FIM": I_F,
        "J_R": J_R,
        "V": V,
    }

    if v_null is not None:
        v_norm = v_null / np.linalg.norm(v_null)
        directional_fisher = float(np.sqrt(np.maximum(v_norm.T @ I_F @ v_norm, 0.0)))
        res["directional_fisher"] = directional_fisher
        # Projections onto singular vectors
        c = V.T @ v_norm
        res["v_null_projections"] = c
        res["v_null_smallest_mode_c"] = float(abs(c[-1]))

    return res


