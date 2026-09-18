"""
Unit tests for gstack CLI runner, configuration, and quality gates.
"""

import pytest
import os
from scripts.gstack import (
    load_gstack_config,
    run_status,
    run_review,
    run_gap_analysis,
)


def test_gstack_config_validity():
    """Validates .gstack/config.json schema and required fields."""
    config = load_gstack_config()
    assert config.get("name") == "gstack-inverse-fno-damage"
    assert "personas" in config
    assert len(config["personas"]) >= 4
    assert "quality_gates" in config
    assert config["quality_gates"]["min_test_count"] >= 10


def test_gstack_status_execution():
    """Validates that gstack status runs without error."""
    code = run_status()
    assert code == 0


def test_gstack_gap_analysis_execution():
    """Validates that gstack gaps runs without error."""
    code = run_gap_analysis()
    assert code == 0


def test_gstack_review_execution():
    """Validates that gstack review runs without error."""
    code = run_review()
    assert code == 0
