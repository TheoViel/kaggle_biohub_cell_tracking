"""Simple 3D+T U-Net.

Input shape  : ``(B, T, C_in, Z, Y, X)``
Output shape : ``(B, T, C_out, Z // sz, Y // sy, X // sx)``, ``output_stride``
being ``(sz, sy, sx)``.

"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

from model_zoo.resenc_unet import anisotropic_strides, cumulative_stride


def _to_3tuple(v) -> tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    t = tuple(v)
    if len(t) != 3:
        raise ValueError(f"Expected a 3-element sequence, got {t}")
    return t


def _norm(kind: str, channels: int) -> nn.Module:
    """Normalisation layer by name.

    Instance norm rescales every sample on its own, which is what survives the
    heavy per-video brightness augmentation; batch norm mixes statistics across
    the batch and, under DDP without ``sync_bn``, across nothing else.
    """
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "instance":
        return nn.InstanceNorm3d(channels, eps=1e-5, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"unknown norm {kind!r}, expected 'batch' / 'instance' / 'group'")


def _conv_block(
    in_channels: int,
    out_channels: int,
    stride: int | tuple[int, int, int] = 1,
    norm: str = "batch",
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
        _norm(norm, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        _norm(norm, out_channels),
        nn.ReLU(inplace=True),
    )


class _ResBlock(nn.Module):
    """Two 3x3x3 convolutions past a projected identity shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int] = 1,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False
        )
        self.norm1 = _norm(norm, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _norm(norm, out_channels)
        self.act = nn.ReLU(inplace=True)

        strided = _to_3tuple(stride) != (1, 1, 1)
        self.shortcut: nn.Module = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _norm(norm, out_channels),
            )
            if strided or in_channels != out_channels
            else nn.Identity()
        )

        # Zero the last scale so every block starts out as its own shortcut:
        # depth then costs nothing at initialisation and the stage grows into it.
        nn.init.zeros_(self.norm2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))


def _res_stage(
    in_channels: int,
    out_channels: int,
    n_blocks: int,
    stride: int | tuple[int, int, int] = 1,
    norm: str = "batch",
) -> nn.Sequential:
    """``n_blocks`` residual blocks, the first one taking the stride."""
    blocks = [_ResBlock(in_channels, out_channels, stride=stride, norm=norm)]
    blocks += [_ResBlock(out_channels, out_channels, norm=norm) for _ in range(n_blocks - 1)]
    return nn.Sequential(*blocks)


class _PixelUnshuffle3d(nn.Module):
    """Downsample by folding each ``(sz, sy, sx)`` block into the channel axis.

    A strided convolution reads one voxel in every block and drops the rest
    before a single feature has been computed; at ``(1, 2, 2)`` that is three
    quarters of the Y/X detail, gone at the first layer. Unshuffling keeps all
    of it — every voxel reaches the first convolution — for the price of a
    reshape, and the network still runs on the coarser grid afterwards.
    """

    def __init__(self, stride: int | tuple[int, int, int]) -> None:
        super().__init__()
        self.stride = _to_3tuple(stride)

    @property
    def factor(self) -> int:
        return math.prod(self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz, sy, sx = self.stride
        z, y, x_ = x.shape[-3:]
        pz, py, px = (-z) % sz, (-y) % sy, (-x_) % sx
        if pz or py or px:
            x = F.pad(x, (0, px, 0, py, 0, pz))
        n, c = x.shape[:2]
        z, y, x_ = x.shape[-3:]
        x = x.reshape(n, c, z // sz, sz, y // sy, sy, x_ // sx, sx)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
        return x.reshape(n, c * sz * sy * sx, z // sz, y // sy, x_ // sx)


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
        ``len(layers)``; spatial size is divided by the stride schedule
        before every stage except the first, whose step the stem takes.
    blocks_per_stage : sequence of int or None
        Residual blocks per encoder stage. ``None`` (default) keeps the plain
        two-convolution block per stage — no shortcut, no depth knob. Give it
        something like ``(1, 3, 4)`` to put nnU-Net's residual encoder depth
        behind the same geometry, which is the cheapest way to add capacity
        without touching the resolution the head reads.
    gradient_checkpointing : bool
        If True, wrap encoder/decoder blocks with ``torch.utils.checkpoint``
        during training to reduce activation memory at the cost of recomputing
        activations in the backward pass.
    skip_fullres_temporal : bool
        If True (default), replace the temporal-attention block at the
        first encoder stage with an Identity.
    temporal_attention : bool
        If False, drop the temporal attention blocks entirely and treat the
        window as a batch of independent frames.
    pool_stride : int or (int, int, int)
        The anisotropic downsampling stride, applied to the first
        ``n_anisotropic`` steps.  ``(1, 2, 2)`` pools Y/X only, keeping Z at
        full resolution while the coarser Z spacing catches up.
    n_anisotropic : int
        How many downsampling steps use ``pool_stride`` before the rest go
        isotropic.  The stem's own step counts as the first.
    stem : {"stride", "unshuffle"}
        How the first downsampling step is taken. ``"stride"`` is a strided
        convolution, which reads one voxel per block and discards the others
        before any feature exists. ``"unshuffle"`` (recommended) folds the
        block into the channel axis instead, so the first convolution sees
        every voxel of the raw image at no extra full-resolution cost.
    norm : {"batch", "instance", "group"}
        Normalisation used throughout.
    n_decode_steps : int or None
        Number of decoder upsample stages to apply.  Defaults to
        ``len(layers) - 1`` (fully symmetric U-Net).  Set to 0 to skip
        the decoder entirely — the head then operates on the bottleneck
        features, producing output at the encoder's lowest resolution.
    hires_skip_fusion : bool
        When the decoder stops short of the input resolution, the finest
        encoder skips are computed and then thrown away. If True, max-pool
        each of them down to the head's grid and concatenate: the head keeps
        a direct line to the sharpest features the encoder ever held, and
        max-pooling is what carries a local peak through the downsample
        rather than averaging it into its neighbours.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        layers: Sequence[int] = (32, 64, 128),
        blocks_per_stage: Sequence[int] | None = None,
        gradient_checkpointing: bool = False,
        skip_fullres_temporal: bool = True,
        temporal_attention: bool = True,
        pool_stride: int | tuple[int, int, int] = 2,
        n_anisotropic: int = 2,
        stem: str = "stride",
        norm: str = "batch",
        n_decode_steps: int | None = None,
        hires_skip_fusion: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        layers = list(layers)
        if len(layers) < 2:
            raise ValueError("layers must contain at least two stages")
        n_stages = len(layers)

        if blocks_per_stage is not None:
            blocks_per_stage = [int(b) for b in blocks_per_stage]
            if len(blocks_per_stage) != n_stages:
                raise ValueError(
                    f"blocks_per_stage must have {n_stages} entries to match layers, "
                    f"got {len(blocks_per_stage)}"
                )

        self.gradient_checkpointing = gradient_checkpointing
        self.verbose = verbose

        pool_stride_t = _to_3tuple(pool_stride)
        # The stem takes the first downsampling step, so nothing past a single
        # reshape or convolution ever runs at full resolution. Everything below
        # reads the schedule instead of inferring it from tensor shapes.
        stem_stride, stage_strides = anisotropic_strides(
            n_stages,
            pool_stride=pool_stride_t,
            n_anisotropic=n_anisotropic,
            stem_strides=True,
        )

        if stem == "unshuffle":
            self.stem: nn.Module = _PixelUnshuffle3d(stem_stride)
            stem_conv_stride: tuple[int, int, int] = (1, 1, 1)
            prev = in_channels * math.prod(stem_stride)
        elif stem == "stride":
            self.stem = nn.Identity()
            stem_conv_stride = stem_stride  # folded into the first stage's conv
            prev = in_channels
        else:
            raise ValueError(f"unknown stem {stem!r}, expected 'stride' / 'unshuffle'")

        def build_stage(c_in: int, c_out: int, n_blocks: int, stride=(1, 1, 1)) -> nn.Module:
            if blocks_per_stage is None:
                return _conv_block(c_in, c_out, stride=stride, norm=norm)
            return _res_stage(c_in, c_out, n_blocks, stride=stride, norm=norm)

        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        for i, ch in enumerate(layers):
            stride = stem_conv_stride if i == 0 else (1, 1, 1)
            self.encoder_blocks.append(
                build_stage(prev, ch, blocks_per_stage[i] if blocks_per_stage else 1, stride)
            )
            if not temporal_attention or (skip_fullres_temporal and i == 0):
                self.temporal_blocks.append(nn.Identity())
            else:
                self.temporal_blocks.append(_TemporalAttention(ch))
            prev = ch

        # One pool per stage; stage 0's step already belongs to the stem, so its
        # entry is the identity.
        self.pools = nn.ModuleList(
            [
                nn.Identity()
                if tuple(s) == (1, 1, 1)
                else nn.MaxPool3d(kernel_size=tuple(s), stride=tuple(s))
                for s in stage_strides
            ]
        )

        n_full_decode = n_stages - 1
        if n_decode_steps is None:
            n_decode_steps = n_full_decode
        n_decode_steps = max(0, min(n_decode_steps, n_full_decode))

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_full_decode, n_full_decode - n_decode_steps, -1):
            self.upsamples.append(
                nn.Upsample(
                    scale_factor=tuple(float(s) for s in stage_strides[i]),
                    mode="trilinear",
                    align_corners=False,
                )
            )
            # Depth belongs in the encoder; one block per decoder stage is
            # nnU-Net's split too, and these run at the expensive resolutions.
            self.decoder_blocks.append(build_stage(layers[i] + layers[i - 1], layers[i - 1], 1))

        # Which encoder stage the head ends up reading; the bottleneck when
        # n_decode_steps == 0.
        head_stage = n_full_decode - n_decode_steps
        head_in = layers[head_stage]

        # Skips finer than the head's grid are otherwise computed and dropped.
        self.hires_pools = nn.ModuleList()
        self.hires_projs = nn.ModuleList()
        if hires_skip_fusion:
            fuse_channels = max(8, layers[0] // 2)
            for j in range(head_stage):
                factor = cumulative_stride(list(stage_strides[j + 1 : head_stage + 1]))
                self.hires_pools.append(nn.MaxPool3d(kernel_size=factor, stride=factor))
                self.hires_projs.append(nn.Conv3d(layers[j], fuse_channels, kernel_size=1))
                head_in += fuse_channels

        self.head = nn.Conv3d(head_in, out_channels, kernel_size=1)

        self.out_channels = out_channels
        self.n_decode_steps = n_decode_steps
        # Spatial sizes must be divisible by this; the encoder runs every stage
        # regardless of how far back the decoder goes.
        self.divisible_by = cumulative_stride(stem_stride, list(stage_strides))
        # How much coarser the output is than the input, which is what the
        # tracking pipeline measures at runtime as unet_scaling.
        self.output_stride = cumulative_stride(
            stem_stride, list(stage_strides[: head_stage + 1])
        )

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return _grad_ckpt(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in, Z, Y, X) -> (B, T, C_out, Z', Y', X')
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = self.stem(x)

        skips: list[torch.Tensor] = []
        for i, (pool, block, temporal) in enumerate(
            zip(self.pools, self.encoder_blocks, self.temporal_blocks)
        ):
            if self.verbose >= 1:
                print(f"TemporalUNet3D - Encoder Block {i} - Input shape: {x.shape}")

            x = pool(x)
            x = self._run(block, x)
            # Attention mixes the frames of one window, so it needs the time axis back.
            if not isinstance(temporal, nn.Identity):
                x = temporal(x.reshape(B, T, *x.shape[1:])).reshape(B * T, *x.shape[1:])
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)

        if self.verbose >= 1:
            print(f"TemporalUNet3D - Encoder output shape: {x.shape}")

        for i, (up, block, skip) in enumerate(zip(self.upsamples, self.decoder_blocks, skips[::-1])):
            x = up(x)

            if self.verbose >= 1:
                print(f"TemporalUNet3D - Decoder Block {i} upsampled shape: {x.shape} - skip shape: {skip.shape}")

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self._run(block, x)

        if self.hires_projs:
            detail = []
            for proj, pool, skip in zip(self.hires_projs, self.hires_pools, skips):
                d = proj(pool(skip))
                if d.shape[2:] != x.shape[2:]:
                    d = F.interpolate(d, size=x.shape[2:], mode="trilinear", align_corners=False)
                detail.append(d)
            x = torch.cat([x, *detail], dim=1)

        x = self.head(x)
        return x.reshape(B, T, *x.shape[1:])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
