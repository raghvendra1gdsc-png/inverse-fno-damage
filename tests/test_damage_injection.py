"""
Unit tests for OpenSeesPy finite-element model and damage injection engine.
"""

import pytest
import numpy as np
from src.damage_injection import (
    MultiStoryFrame,
    FrameConfig,
    sample_random_damage_field,
)


def test_baseline_frame_frequencies():
    """Validates baseline 3-story frame natural frequencies against Guyan condensation."""
    frame = MultiStoryFrame()
    frame.build_model()
    modal = frame.extract_eigenvalues(num_modes=3)
    f1, f2, f3 = modal["frequencies"]

    # Target frequencies from exact condensed Guyan stiffness matrix
    assert abs(f1 - 2.1856) < 0.05, f"Fundamental frequency {f1} deviates from 2.1856 Hz"
    assert abs(f2 - 6.8500) < 0.15, f"Second frequency {f2} deviates from 6.8500 Hz"
    assert abs(f3 - 11.4039) < 0.25, f"Third frequency {f3} deviates from 11.4039 Hz"


def test_damage_sampling_bounds():
    """Ensures damage factors remain strictly within physical limits [0, 0.5]."""
    rng = np.random.RandomState(42)

    # Undamaged mode
    d_clean = sample_random_damage_field(mode="undamaged", rng=rng)
    assert np.all(d_clean == 0.0), "Undamaged damage field must be all zeros"

    # Sparse damage mode
    d_sparse = sample_random_damage_field(mode="sparse", max_damaged_elements=2, rng=rng)
    assert len(d_sparse) == 9, "Damage vector must have 9 elements"
    assert np.all(d_sparse >= 0.0), "Damage factor cannot be negative"
    assert np.all(d_sparse <= 0.50), "Damage factor cannot exceed 50% stiffness reduction"
    assert np.sum(d_sparse > 0.0) <= 2, "Sparse mode must have at most 2 damaged elements"


def test_transient_dynamic_analysis():
    """Tests execution of transient dynamic analysis and tensor shapes."""
    frame = MultiStoryFrame()
    frame.build_model()

    # Synthetic earthquake input
    dt = 0.01
    t = np.arange(200) * dt
    gm = np.sin(2 * np.pi * 2.0 * t) * 1.5  # 2 Hz sine wave, 1.5 m/s^2 peak

    channels = [(3, 1), (5, 1), (7, 1)]
    res = frame.run_dynamic_analysis(
        ground_accel_ms2=gm,
        dt=dt,
        sensor_channels=channels,
        damping_ratio=0.03,
    )

    assert res["time"].shape == (200,)
    assert res["sensor_total_accel"].shape == (3, 200)
    assert not np.isnan(res["sensor_total_accel"]).any(), "Accelerations contain NaN"
    assert len(res["modal_frequencies"]) == 3
