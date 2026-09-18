"""
Unit tests for Forward and Inverse Fourier Neural Operators.
"""

import pytest
import torch
from src.forward_fno_model import SpectralConv1d, ForwardFNO
from src.inverse_fno_model import InverseFNO


def test_spectral_conv1d():
    """Tests 1D spectral convolution forward pass and complex multiplication."""
    conv = SpectralConv1d(in_channels=16, out_channels=32, modes1=8)
    x = torch.randn(4, 16, 256)
    out = conv(x)
    assert out.shape == (4, 32, 256), f"Expected shape (4, 32, 256), got {out.shape}"
    assert not torch.isnan(out).any()


def test_forward_fno_forward_pass():
    """Tests ForwardFNO tensor mapping: (B, 11, T) -> (B, 3, T)."""
    model = ForwardFNO(in_channels=11, out_channels=3, modes=16, width=32, n_layers=2)
    x = torch.randn(2, 11, 200)
    y = model(x)
    assert y.shape == (2, 3, 200), f"Expected shape (2, 3, 200), got {y.shape}"
    assert not torch.isnan(y).any()


def test_inverse_fno_forward_pass_and_nonnegativity():
    """Tests InverseFNO mapping: (B, 3, T) -> (B, 9) and non-negative damage output."""
    model = InverseFNO(in_channels=3, num_elements=9, modes=16, width=32, n_layers=2)
    y = torch.randn(4, 3, 200)
    d = model(y)
    assert d.shape == (4, 9), f"Expected shape (4, 9), got {d.shape}"
    assert torch.all(d >= 0.0), "InverseFNO must predict non-negative damage"
    assert not torch.isnan(d).any()
