"""
Unit tests for evaluation metrics and sensor subset pruning.
"""

import pytest
import numpy as np
from src.inverse_fno_model import compute_inverse_metrics
from src.evaluate import select_nested_sensor_subsets


def test_compute_inverse_metrics():
    """Tests localization and severity error calculations."""
    d_trues = np.array([
        [0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Peak at E0
        [0.00, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Peak at E2
    ])
    d_preds = np.array([
        [0.25, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Peak at E0 (Correct)
        [0.00, 0.30, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Peak at E1 (Mislocalized)
    ])
    flags = [False, False]

    metrics = compute_inverse_metrics(d_preds, d_trues, flags)

    # 1 out of 2 correct = 50%
    assert metrics["localization"]["top1_accuracy_pct"] == 50.0
    assert metrics["localization"]["top1_error_rate_pct"] == 50.0
    assert metrics["overall"]["num_test_samples"] == 2


def test_nested_sensor_subsets():
    """Tests deterministic nested sensor subset selection."""
    subsets = select_nested_sensor_subsets(total_sensors=10, counts=[10, 8, 6, 4, 2], seed=42)

    assert len(subsets[10]) == 10
    assert len(subsets[8]) == 8
    assert len(subsets[6]) == 6
    assert len(subsets[4]) == 4
    assert len(subsets[2]) == 2

    # Check nesting property: smaller count is subset of larger count
    assert set(subsets[2]).issubset(set(subsets[4]))
    assert set(subsets[4]).issubset(set(subsets[6]))
    assert set(subsets[6]).issubset(set(subsets[8]))
    assert set(subsets[8]).issubset(set(subsets[10]))
