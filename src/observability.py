"""
Mathematical Observability, Symmetry Analysis, and Sensor-Augmentation Engine.

Module: src.observability
Context: Phase 5 - Observability, Symmetry, and Sensor-Augmentation Study
Author: Inverse FNO Project Team
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from src.damage_injection import MultiStoryFrame, FrameConfig


# ==============================================================================
# 1. Bilateral Symmetry Permutation Operator
# ==============================================================================

class BilateralSymmetry:
    """
    Formal representation of the bilateral left/right structural symmetry operator.

    For the 3-story, 1-bay moment frame (9 elements):
      Columns (6 elements):
        - Ele 1 (Story 1 Left)  <-->  Ele 2 (Story 1 Right)
        - Ele 3 (Story 2 Left)  <-->  Ele 4 (Story 2 Right)
        - Ele 5 (Story 3 Left)  <-->  Ele 6 (Story 3 Right)
      Beams (3 elements, span between left and right column lines):
        - Ele 7 (Floor 1 Beam)  <-->  Ele 7 (Self-symmetric)
        - Ele 8 (Floor 2 Beam)  <-->  Ele 8 (Self-symmetric)
        - Ele 9 (Roof Beam)     <-->  Ele 9 (Self-symmetric)
    """

    # 0-indexed permutation mapping: index i -> permuted index P[i]
    PERMUTATION_MAP: List[int] = [1, 0, 3, 2, 5, 4, 6, 7, 8]

    @classmethod
    def get_permutation_matrix(cls) -> np.ndarray:
        """Returns the 9x9 orthogonal permutation matrix P."""
        P = np.zeros((9, 9), dtype=np.float64)
        for i, j in enumerate(cls.PERMUTATION_MAP):
            P[j, i] = 1.0
        return P

    @classmethod
    def apply_permutation(cls, d: np.ndarray) -> np.ndarray:
        """Applies permutation P to a 9-element damage vector: P(d)."""
        d_arr = np.asarray(d, dtype=np.float64)
        assert len(d_arr) == 9, f"Expected 9 elements, got {len(d_arr)}"
        return d_arr[cls.PERMUTATION_MAP]

    @classmethod
    def get_canonical_states(cls) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns canonical State A (30% Left Col 1) and State B (30% Right Col 1).
        State B is strictly equal to P(State A).
        """
        d_A = np.zeros(9, dtype=np.float64)
        d_A[0] = 0.30  # Element 1: Left Column 1

        d_B = cls.apply_permutation(d_A)
        # Element 2: Right Column 1 (index 1)
        assert d_B[1] == 0.30 and d_B[0] == 0.0
        return d_A, d_B

    @classmethod
    def get_theoretical_null_vector(cls) -> np.ndarray:
        """
        Returns the normalized theoretical symmetry null direction:
        v_null = (d_A - d_B) / ||d_A - d_B||_2.
        """
        d_A, d_B = cls.get_canonical_states()
        diff = d_A - d_B
        return diff / np.linalg.norm(diff)


# ==============================================================================
# 2. Sensor Configuration Abstractions (S0 through S4)
# ==============================================================================

@dataclass
class SensorChannelSpec:
    """Specification of a physical sensor measurement channel."""
    channel_id: str
    channel_type: str  # "accel_horiz", "accel_vert", "strain_col", "rocking"
    location: str
    units: str
    node_tag: Optional[int] = None
    dof: Optional[int] = None
    ele_id: Optional[int] = None
    story: Optional[int] = None


@dataclass
class SensorConfiguration:
    """Defines a spatial sensing layout and measurement extraction logic."""
    name: str
    description: str
    channels: List[SensorChannelSpec]

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @classmethod
    def create_S0(cls) -> "SensorConfiguration":
        """S0: Baseline Commercial Layout — 3 Horizontal Floor Accelerometers."""
        channels = [
            SensorChannelSpec("H_F1", "accel_horiz", "Floor 1 (Node 3)", "m/s^2", node_tag=3, dof=1, story=1),
            SensorChannelSpec("H_F2", "accel_horiz", "Floor 2 (Node 5)", "m/s^2", node_tag=5, dof=1, story=2),
            SensorChannelSpec("H_RF", "accel_horiz", "Roof (Node 7)", "m/s^2", node_tag=7, dof=1, story=3),
        ]
        return cls("S0", "3 Horizontal Floor Accelerometers", channels)

    @classmethod
    def create_S1(cls) -> "SensorConfiguration":
        """S1: Horizontal + Vertical Accelerometers at all column joints (9 channels)."""
        s0 = cls.create_S0().channels
        vert_channels = [
            SensorChannelSpec("V_N3", "accel_vert", "Floor 1 Left (Node 3)", "m/s^2", node_tag=3, dof=2, story=1),
            SensorChannelSpec("V_N4", "accel_vert", "Floor 1 Right (Node 4)", "m/s^2", node_tag=4, dof=2, story=1),
            SensorChannelSpec("V_N5", "accel_vert", "Floor 2 Left (Node 5)", "m/s^2", node_tag=5, dof=2, story=2),
            SensorChannelSpec("V_N6", "accel_vert", "Floor 2 Right (Node 6)", "m/s^2", node_tag=6, dof=2, story=2),
            SensorChannelSpec("V_N7", "accel_vert", "Roof Left (Node 7)", "m/s^2", node_tag=7, dof=2, story=3),
            SensorChannelSpec("V_N8", "accel_vert", "Roof Right (Node 8)", "m/s^2", node_tag=8, dof=2, story=3),
        ]
        return cls("S1", "3 Horizontal + 6 Vertical Accelerometers", s0 + vert_channels)

    @classmethod
    def create_S2(cls) -> "SensorConfiguration":
        """S2: Horizontal Accelerometers + Column Axial Strains (9 channels)."""
        s0 = cls.create_S0().channels
        strain_channels = [
            SensorChannelSpec("Eps_C1", "strain_col", "Col 1 Left Story 1", "strain", ele_id=1, story=1),
            SensorChannelSpec("Eps_C2", "strain_col", "Col 2 Right Story 1", "strain", ele_id=2, story=1),
            SensorChannelSpec("Eps_C3", "strain_col", "Col 3 Left Story 2", "strain", ele_id=3, story=2),
            SensorChannelSpec("Eps_C4", "strain_col", "Col 4 Right Story 2", "strain", ele_id=4, story=2),
            SensorChannelSpec("Eps_C5", "strain_col", "Col 5 Left Story 3", "strain", ele_id=5, story=3),
            SensorChannelSpec("Eps_C6", "strain_col", "Col 6 Right Story 3", "strain", ele_id=6, story=3),
        ]
        return cls("S2", "3 Horizontal Accels + 6 Column Axial Strains", s0 + strain_channels)

    @classmethod
    def create_S3(cls) -> "SensorConfiguration":
        """S3: Horizontal Accelerometers + Antisymmetric Rocking Accelerations (6 channels)."""
        s0 = cls.create_S0().channels
        rocking_channels = [
            SensorChannelSpec("Rock_F1", "rocking", "Story 1 Rocking", "rad/s^2", story=1),
            SensorChannelSpec("Rock_F2", "rocking", "Story 2 Rocking", "rad/s^2", story=2),
            SensorChannelSpec("Rock_RF", "rocking", "Story 3 Rocking", "rad/s^2", story=3),
        ]
        return cls("S3", "3 Horizontal Accels + 3 Story Rocking Observables", s0 + rocking_channels)

    @classmethod
    def create_S4(cls) -> "SensorConfiguration":
        """S4: Comprehensive Multi-Modal Sensor Configuration (18 channels)."""
        s0 = cls.create_S0().channels
        s1_vert = [ch for ch in cls.create_S1().channels if ch.channel_type == "accel_vert"]
        s2_strain = [ch for ch in cls.create_S2().channels if ch.channel_type == "strain_col"]
        s3_rock = [ch for ch in cls.create_S3().channels if ch.channel_type == "rocking"]
        return cls("S4", "Comprehensive Multi-Modal (Horizontal + Vertical + Strain + Rocking)", s0 + s1_vert + s2_strain + s3_rock)


# ==============================================================================
# 3. Forward Observation Map
# ==============================================================================

class ForwardObservationMap:
    """
    Evaluates F(d; a_g, S): maps damage vector d and earthquake ground motion a_g
    to discrete multichannel observation histories Y(t) in R^(C x T).
    """

    # Nodes required to derive all kinematic quantities (DOFs 1 & 2 for nodes 3..8)
    BASE_CHANNELS: List[Tuple[int, int]] = [
        (3, 1), (4, 1),  # Story 1 Left & Right Horizontal
        (5, 1), (6, 1),  # Story 2 Left & Right Horizontal
        (7, 1), (8, 1),  # Story 3 Left & Right Horizontal
        (3, 2), (4, 2),  # Story 1 Left & Right Vertical
        (5, 2), (6, 2),  # Story 2 Left & Right Vertical
        (7, 2), (8, 2),  # Story 3 Left & Right Vertical
    ]

    @classmethod
    def evaluate(
        cls,
        frame: MultiStoryFrame,
        damage_vector: np.ndarray,
        ground_accel_ms2: np.ndarray,
        dt: float,
        sensor_config: SensorConfiguration,
        damping_ratio: float = 0.03,
    ) -> Dict[str, Any]:
        """
        Runs OpenSees dynamic simulation and extracts channels for sensor_config.

        Returns
        -------
        Dict with:
          - 'Y': (C, T) response matrix
          - 'y_vec': flattened 1D array (C * T,)
          - 'time': (T,) array
          - 'channel_names': List of channel IDs
        """
        # Set frame damage and run dynamic simulation with full joint DOF coverage
        frame.build_model(damage_vector)
        raw_res = frame.run_dynamic_analysis(
            ground_accel_ms2=ground_accel_ms2,
            dt=dt,
            sensor_channels=cls.BASE_CHANNELS,
            damping_ratio=damping_ratio,
        )

        time = raw_res["time"]
        n_steps = len(time)
        ch_map = {ch: idx for idx, ch in enumerate(cls.BASE_CHANNELS)}

        tot_acc = raw_res["sensor_total_accel"]  # (12, T)
        disps = raw_res["sensor_disp"]           # (12, T)
        L_bay = frame.config.bay_width           # 6.0 m
        h_story = frame.config.story_height      # 3.0 m

        C = sensor_config.num_channels
        Y = np.zeros((C, n_steps), dtype=np.float64)
        channel_names = []

        for c_idx, ch_spec in enumerate(sensor_config.channels):
            channel_names.append(ch_spec.channel_id)

            if ch_spec.channel_type == "accel_horiz":
                # Horizontal floor acceleration at node_tag, DOF 1
                row = ch_map[(ch_spec.node_tag, 1)]
                Y[c_idx, :] = tot_acc[row, :]

            elif ch_spec.channel_type == "accel_vert":
                # Vertical acceleration at node_tag, DOF 2
                row = ch_map[(ch_spec.node_tag, 2)]
                Y[c_idx, :] = tot_acc[row, :]

            elif ch_spec.channel_type == "strain_col":
                # Column axial strain = (u_y,top - u_y,bot) / h
                ele = ch_spec.ele_id
                if ele == 1:  # Story 1 Left (Node 1 fixed to Node 3)
                    u_top = disps[ch_map[(3, 2)], :]
                    u_bot = 0.0
                elif ele == 2:  # Story 1 Right (Node 2 fixed to Node 4)
                    u_top = disps[ch_map[(4, 2)], :]
                    u_bot = 0.0
                elif ele == 3:  # Story 2 Left (Node 3 to Node 5)
                    u_top = disps[ch_map[(5, 2)], :]
                    u_bot = disps[ch_map[(3, 2)], :]
                elif ele == 4:  # Story 2 Right (Node 4 to Node 6)
                    u_top = disps[ch_map[(6, 2)], :]
                    u_bot = disps[ch_map[(4, 2)], :]
                elif ele == 5:  # Story 3 Left (Node 5 to Node 7)
                    u_top = disps[ch_map[(7, 2)], :]
                    u_bot = disps[ch_map[(5, 2)], :]
                elif ele == 6:  # Story 3 Right (Node 6 to Node 8)
                    u_top = disps[ch_map[(8, 2)], :]
                    u_bot = disps[ch_map[(6, 2)], :]
                else:
                    raise ValueError(f"Unknown column element ID: {ele}")

                Y[c_idx, :] = (u_top - u_bot) / h_story

            elif ch_spec.channel_type == "rocking":
                # Story rocking acceleration = (a_y,right - a_y,left) / L_bay
                s = ch_spec.story
                if s == 1:
                    a_left = tot_acc[ch_map[(3, 2)], :]
                    a_right = tot_acc[ch_map[(4, 2)], :]
                elif s == 2:
                    a_left = tot_acc[ch_map[(5, 2)], :]
                    a_right = tot_acc[ch_map[(6, 2)], :]
                elif s == 3:
                    a_left = tot_acc[ch_map[(7, 2)], :]
                    a_right = tot_acc[ch_map[(8, 2)], :]
                else:
                    raise ValueError(f"Unknown story: {s}")

                Y[c_idx, :] = (a_right - a_left) / L_bay

            else:
                raise ValueError(f"Unsupported channel type: {ch_spec.channel_type}")

        return {
            "Y": Y,
            "y_vec": Y.flatten(),
            "time": time,
            "dt": dt,
            "channel_names": channel_names,
            "sensor_config": sensor_config,
        }


# ==============================================================================
# 4. Damage-to-Response Jacobian via Central Finite Differences
# ==============================================================================

def compute_damage_jacobian(
    frame: MultiStoryFrame,
    damage_ref: np.ndarray,
    ground_accel_ms2: np.ndarray,
    dt: float,
    sensor_config: SensorConfiguration,
    h: float = 1e-3,
    damping_ratio: float = 0.03,
) -> np.ndarray:
    """
    Computes the Damage-to-Response Jacobian matrix J_d in R^(N_obs x 9):
      J_d[:, e] = [F(d + h*e_e) - F(d - h*e_e)] / (2h)

    Parameters
    ----------
    frame : MultiStoryFrame
    damage_ref : np.ndarray of shape (9,)
    ground_accel_ms2 : np.ndarray
    dt : float
    sensor_config : SensorConfiguration
    h : float, default 1e-3
    damping_ratio : float, default 0.03

    Returns
    -------
    J_d : np.ndarray of shape (N_obs, 9)
    """
    d_ref = np.asarray(damage_ref, dtype=np.float64)
    assert len(d_ref) == 9

    # Baseline evaluation to determine N_obs
    res_base = ForwardObservationMap.evaluate(
        frame=frame,
        damage_vector=d_ref,
        ground_accel_ms2=ground_accel_ms2,
        dt=dt,
        sensor_config=sensor_config,
        damping_ratio=damping_ratio,
    )
    N_obs = len(res_base["y_vec"])
    J_d = np.zeros((N_obs, 9), dtype=np.float64)

    for e in range(9):
        # Forward perturbation
        d_plus = d_ref.copy()
        d_plus[e] = np.clip(d_plus[e] + h, 0.0, 0.95)

        res_plus = ForwardObservationMap.evaluate(
            frame=frame,
            damage_vector=d_plus,
            ground_accel_ms2=ground_accel_ms2,
            dt=dt,
            sensor_config=sensor_config,
            damping_ratio=damping_ratio,
        )

        # Backward perturbation
        d_minus = d_ref.copy()
        d_minus[e] = np.clip(d_minus[e] - h, 0.0, 0.95)

        res_minus = ForwardObservationMap.evaluate(
            frame=frame,
            damage_vector=d_minus,
            ground_accel_ms2=ground_accel_ms2,
            dt=dt,
            sensor_config=sensor_config,
            damping_ratio=damping_ratio,
        )

        # Actual step taken (accounting for clipping if near boundaries)
        actual_2h = d_plus[e] - d_minus[e]
        if actual_2h > 1e-12:
            J_d[:, e] = (res_plus["y_vec"] - res_minus["y_vec"]) / actual_2h
        else:
            J_d[:, e] = 0.0

    return J_d


# ==============================================================================
# 5. Observability & SVD Analysis
# ==============================================================================

def analyze_svd(J_d: np.ndarray, v_null: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Computes SVD of Jacobian: J_d = U Sigma V^T.

    Returns
    -------
    Dict with:
      - 'singular_values': array of 9 singular values (descending)
      - 'condition_number': sigma_1 / sigma_9
      - 'effective_rank': count of singular values > 1e-4 * sigma_1
      - 'V': 9x9 matrix of right singular vectors (columns)
      - 'v_min': 9-element right singular vector for smallest singular value
      - 'cosine_sim_null': |v_min^T v_null| if v_null provided
    """
    U, s, Vt = np.linalg.svd(J_d, full_matrices=False)
    V = Vt.T  # columns are right singular vectors v_1, ..., v_9

    sigma_max = float(s[0]) if len(s) > 0 else 0.0
    sigma_min = float(s[-1]) if len(s) > 0 else 0.0
    cond = float(sigma_max / sigma_min) if sigma_min > 1e-15 else float("inf")
    rank = int(np.sum(s > 1e-4 * sigma_max)) if sigma_max > 1e-15 else 0

    v_min = V[:, -1]
    cos_sim = None
    if v_null is not None:
        v_null_norm = v_null / np.linalg.norm(v_null)
        cos_sim = float(np.abs(np.dot(v_min, v_null_norm)))

    return {
        "singular_values": s,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "condition_number": cond,
        "effective_rank": rank,
        "V": V,
        "v_min": v_min,
        "cosine_sim_null": cos_sim,
    }


# ==============================================================================
# 6. Noise-Normalized Observability & Fisher Information
# ==============================================================================

def compute_noise_normalized_observability(
    J_d: np.ndarray,
    Y_ref: np.ndarray,
    noise_ratio: float = 0.02,
    lambda_reg: float = 1e-4,
    v_null: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes noise-normalized Jacobian J_R = R^(-1/2) J_d and Fisher Information Matrix:
      I_F = J_R^T J_R = J_d^T R^(-1) J_d.

    Parameters
    ----------
    J_d : (N_obs, 9) Jacobian
    Y_ref : (C, T) reference response
    noise_ratio : float, e.g. 0.02 for 2% noise
    lambda_reg : float, regularization for log det(I + lambda*I_9)
    v_null : Optional normalized null vector for alignment check

    Returns
    -------
    Dict of noise-aware metrics.
    """
    C, T = Y_ref.shape
    # Channel-wise RMS noise
    rms_per_channel = np.sqrt(np.mean(Y_ref**2, axis=1))  # (C,)
    # Minimum floor to prevent divide-by-zero on quiescent channels
    rms_per_channel = np.maximum(rms_per_channel, 1e-8)
    sigma_per_channel = noise_ratio * rms_per_channel

    # Noise std array across all observations
    noise_std_vec = np.repeat(sigma_per_channel, T)  # (C * T,)

    # Noise-normalized Jacobian: J_R = J_d / sigma
    J_R = J_d / noise_std_vec[:, np.newaxis]

    # Fisher Information Matrix (9x9)
    I_F = J_R.T @ J_R

    # Eigenvalues of FIM (descending)
    eigvals = np.linalg.eigvalsh(I_F)[::-1]
    lambda_max = float(eigvals[0])
    lambda_min = float(eigvals[-1])

    # Log determinant with fixed regularization lambda_reg
    log_det = float(np.sum(np.log(np.maximum(eigvals + lambda_reg, 1e-15))))

    # SVD of J_R
    svd_R = analyze_svd(J_R, v_null=v_null)

    return {
        "noise_ratio": noise_ratio,
        "FIM": I_F,
        "fim_eigenvalues": eigvals,
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "log_det": log_det,
        "svd_noise_normalized": svd_R,
        "sigma_min_noise_normalized": svd_R["sigma_min"],
        "condition_number_noise_normalized": svd_R["condition_number"],
    }


# ==============================================================================
# 7. State Separation Metrics
# ==============================================================================

def compute_state_separation(
    res_A: Dict[str, Any],
    res_B: Dict[str, Any],
    noise_ratio: float = 0.02,
) -> Dict[str, Any]:
    """
    Computes rigorous separation metrics between two damage states:
      - Absolute and relative L2 differences
      - Pearson correlation
      - Max absolute difference and RMS difference
      - Separation-to-Noise ratio rho_AB = ||y_A - y_B||_2 / ||sigma_noise||_2
    """
    Y_A = res_A["Y"]
    Y_B = res_B["Y"]
    y_A = res_A["y_vec"]
    y_B = res_B["y_vec"]
    dt = res_A["dt"]
    ch_names = res_A["channel_names"]

    diff_vec = y_A - y_B
    norm_A = float(np.linalg.norm(y_A))
    norm_diff = float(np.linalg.norm(diff_vec))
    rel_l2 = float(norm_diff / norm_A) if norm_A > 1e-12 else float("inf")

    # Pearson correlation across flattened response
    if np.std(y_A) > 1e-12 and np.std(y_B) > 1e-12:
        corr = float(np.corrcoef(y_A, y_B)[0, 1])
    else:
        corr = 1.0

    max_abs_diff = float(np.max(np.abs(diff_vec)))
    rms_diff = float(np.sqrt(np.mean(diff_vec**2)))

    # Per-channel metrics
    C, T = Y_A.shape
    per_channel = []
    total_noise_variance = 0.0

    for c in range(C):
        yA_c = Y_A[c, :]
        yB_c = Y_B[c, :]
        d_c = yA_c - yB_c
        normA_c = float(np.linalg.norm(yA_c))
        normD_c = float(np.linalg.norm(d_c))
        rel_c = float(normD_c / normA_c) if normA_c > 1e-12 else 0.0
        corr_c = float(np.corrcoef(yA_c, yB_c)[0, 1]) if np.std(yA_c) > 1e-12 and np.std(yB_c) > 1e-12 else 1.0
        rms_sig_c = float(np.sqrt(np.mean(yA_c**2)))
        sigma_noise_c = noise_ratio * max(rms_sig_c, 1e-8)
        total_noise_variance += T * (sigma_noise_c**2)

        per_channel.append({
            "channel": ch_names[c],
            "norm_A": normA_c,
            "norm_diff": normD_c,
            "relative_l2_pct": rel_c * 100.0,
            "pearson_r": corr_c,
            "max_abs_diff": float(np.max(np.abs(d_c))),
            "rms_diff": float(np.sqrt(np.mean(d_c**2))),
            "noise_rms": sigma_noise_c,
            "separation_ratio_ch": float(normD_c / (np.sqrt(T) * sigma_noise_c)),
        })

    total_noise_norm = np.sqrt(total_noise_variance)
    separation_ratio = float(norm_diff / total_noise_norm) if total_noise_norm > 1e-15 else 0.0

    return {
        "abs_l2_diff": norm_diff,
        "relative_l2_diff": rel_l2,
        "relative_l2_pct": rel_l2 * 100.0,
        "pearson_r": corr,
        "max_abs_diff": max_abs_diff,
        "rms_diff": rms_diff,
        "separation_ratio": separation_ratio,
        "noise_ratio": noise_ratio,
        "total_noise_norm": total_noise_norm,
        "per_channel": per_channel,
    }
