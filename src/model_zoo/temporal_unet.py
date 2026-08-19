"""Simple 3D+T U-Net.

Input shape  : ``(B, T, C_in, Z, Y, X)``
Output shape : ``(B, T, C_out, Z, Y, X)``

"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt


def _to_3tuple(v) -> tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    t = tuple(v)
    if len(t) != 3:
        raise ValueError(f"Expected a 3-element sequence, got {t}")
    return t


def _conv_block(in_channels: int, out_channels: int, stride: int | tuple[int, int, int] = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class _TemporalAttention(nn.Module):
    """Per-voxel multi-head self-attention across time."""

    # Flash / mem-efficient SDPA CUDA kernels reject batch dims >= 65536
    # (grid.z overflow). Per-voxel reshape uses batch = B * Z * Y * X.
    _SDPA_MAX_BATCH = 65535

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, Z, Y, X)
        B, T, C = x.shape[:3]
        spatial = x.shape[3:]
        S = math.prod(spatial)

        h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)
        h = self.norm(h)
        if h.shape[0] <= self._SDPA_MAX_BATCH:
            h, _ = self.attn(h, h, h, need_weights=False)
        else:
            # Chunk spatial batch so Flash/mem-efficient SDPA can launch.
            chunks = []
            for i in range(0, h.shape[0], self._SDPA_MAX_BATCH):
                chunk = h[i : i + self._SDPA_MAX_BATCH]
                out, _ = self.attn(chunk, chunk, chunk, need_weights=False)
                chunks.append(out)
            h = torch.cat(chunks, dim=0)
        h = h.reshape(B, S, T, C).permute(0, 2, 3, 1).reshape(B, T, C, *spatial)
        return x + h


class TemporalUNet3D(nn.Module):
    """Minimal 3D temporal U-Net.

    Parameters
    ----------
    in_channels : int
        Input channels per frame.
    out_channels : int
        Output feature channels per frame.
    layers : sequence of int
        Encoder channel widths, shallow to deep. Number of stages equals
        ``len(layers)``; spatial size is divided by ``pool_stride`` before
        every stage except the first.
    gradient_checkpointing : bool
        If True (default), wrap encoder/decoder conv blocks with
        ``torch.utils.checkpoint`` during training to reduce activation
        memory at the cost of recomputing activations in the backward
        pass.
    skip_fullres_temporal : bool
        If True (default), replace the temporal-attention block at the
        full-resolution (first) encoder stage with an Identity.
    pool_stride : int or (int, int, int)
        Stride of the encoder MaxPool and decoder Upsample at each level.
        Use ``(1, 2, 2)`` to pool only in Y/X (keeping Z at full
        resolution) so that full-scale 3D images can be fed while the
        output stays at ``(Z, Y//(2**n), X//(2**n))``.
    n_decode_steps : int or None
        Number of decoder upsample stages to apply.  Defaults to
        ``len(layers) - 1`` (fully symmetric U-Net).  Set to 0 to skip
        the decoder entirely — the head then operates on the bottleneck
        features, producing output at the encoder's lowest resolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        layers: Sequence[int] = (32, 64, 128),
        gradient_checkpointing: bool = False,
        skip_fullres_temporal: bool = True,
        pool_stride: int | tuple[int, int, int] = 2,
        n_decode_steps: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        layers = list(layers)
        if len(layers) < 2:
            raise ValueError("layers must contain at least two stages")

        self.gradient_checkpointing = gradient_checkpointing
        self.verbose = verbose

        pool_stride_t = _to_3tuple(pool_stride)
        # scale_factor for Upsample must be a float or sequence of floats
        up_scale = 2  #  tuple(float(s) for s in pool_stride_t)

        n_full_decode = len(layers) - 1

        if n_decode_steps is None:
            n_decode_steps = n_full_decode
        n_decode_steps = max(0, min(n_decode_steps, n_full_decode))

        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(layers):
            self.encoder_blocks.append(_conv_block(prev, ch, stride=pool_stride_t if i == 0 else 1))
            if skip_fullres_temporal and i == 0:
                self.temporal_blocks.append(nn.Identity())
            else:
                self.temporal_blocks.append(_TemporalAttention(ch))
            prev = ch
        self.pool = nn.MaxPool3d(kernel_size=pool_stride_t, stride=pool_stride_t)
        self.pool_anisotropic = nn.MaxPool3d(kernel_size=max(pool_stride_t), stride=max(pool_stride_t))

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_full_decode, n_full_decode - n_decode_steps, -1):
            self.upsamples.append(
                nn.Upsample(scale_factor=up_scale, mode="trilinear", align_corners=False)
            )
            self.decoder_blocks.append(_conv_block(layers[i] + layers[i - 1], layers[i - 1]))

        # Head input channels: bottleneck level when n_decode_steps == 0
        head_in = layers[n_full_decode - n_decode_steps]
        self.head = nn.Conv3d(head_in, out_channels, kernel_size=1)
        

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return _grad_ckpt(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in, Z, Y, X) -> (B, T, C_out, Z, Y, X)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])

        Z, Y, X = x.shape[-3:]

        skips: list[torch.Tensor] = []
        for i, (block, temporal) in enumerate(zip(self.encoder_blocks, self.temporal_blocks)):

            if self.verbose >= 1:
                print(f"TemporalUNet3D - Encoder Block {i} - Input shape: {x.shape}")

            if i > 0:
                if x.shape[-1] == X // 4:  # already downsampled twice
                    x = self.pool_anisotropic(x)
                    # print(f"Downsampled anisotropic: {x.shape}")
                else:
                    x = self.pool(x)
                    # print(f"Downsampled isotropic: {x.shape}")

            x = self._run(block, x)
            x = temporal(x.reshape(B, T, *x.shape[1:])).reshape(B * T, *x.shape[1:])
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)

        if self.verbose >= 1:
            print(f"TemporalUNet3D - Encoder output shape: {x.shape}")

        for i, (up, block, skip) in enumerate(zip(self.upsamples, self.decoder_blocks, skips[::-1])):
            x = up(x)

            if self.verbose >= 1:
                print(f"TemporalUNet3D - Decoder Block {i} upsampled shape: {x.shape} - skip shape: {skip.shape if skip is not None else None}")

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self._run(block, x)

        x = self.head(x)
        return x.reshape(B, T, *x.shape[1:])
