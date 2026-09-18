"""
Temporal Fourier Neural Operator (FNO-1D) Blocks.

Module: src.ml.temporal_fno
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    1D Fourier spectral convolution layer.
    Computes complex multiplication in frequency domain:
      v_out(k) = R(k) * v_in(k)  for |k| <= modes.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor [Batch, in_channels, Time]

        Returns
        -------
        out : torch.Tensor [Batch, out_channels, Time]
        """
        B, C, T = x.shape

        # 1. Real FFT along time dimension
        x_ft = torch.fft.rfft(x, dim=-1)

        # 2. Spectral filtering on lowest 'modes' frequencies
        out_ft = torch.zeros(
            B, self.out_channels, x_ft.shape[-1],
            device=x.device, dtype=torch.cfloat
        )
        effective_modes = min(self.modes, x_ft.shape[-1])

        out_ft[:, :, :effective_modes] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :effective_modes],
            self.weights[:, :, :effective_modes]
        )

        # 3. Inverse Real FFT
        return torch.fft.irfft(out_ft, n=T, dim=-1)


class TemporalFNOBlock(nn.Module):
    """
    A single 1D FNO block combining spectral convolution, linear bypass (1x1 conv),
    and non-linear activation.
    """

    def __init__(self, width: int, modes: int = 32, activation: str = "gelu"):
        super().__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [Batch, width, Time] -> out : [Batch, width, Time]
        """
        return self.act(self.conv(x) + self.w(x))


class TemporalFNO(nn.Module):
    """
    Multilayer Temporal FNO for arbitrary input channel dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_width: int = 64,
        modes: int = 32,
        n_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.lifting = nn.Sequential(
            nn.Conv1d(in_channels, out_width, kernel_size=1),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.1),
            nn.Conv1d(out_width, out_width, kernel_size=1),
        )
        self.blocks = nn.ModuleList([
            TemporalFNOBlock(out_width, modes=modes, activation=activation)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [Batch, in_channels, Time] -> out : [Batch, out_width, Time]
        """
        h = self.lifting(x)
        for block in self.blocks:
            h = block(h)
        return h
