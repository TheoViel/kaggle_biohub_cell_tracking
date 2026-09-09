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


def _heads_for(channels: int, n_heads: int) -> int:
    """The largest head count <= ``n_heads`` that divides ``channels``.

    The stage widths are powers of two and the requested head count usually
    divides them, but ``hires_skip_fusion`` and the unshuffle stem both produce
    widths that are not, and a mismatch there would only surface as a reshape
    error deep in a forward pass.
    """
    for h in range(min(n_heads, channels), 0, -1):
        if channels % h == 0:
            return h
    return 1


def _pad_to_multiple(x: torch.Tensor, multiple: tuple[int, int, int]) -> torch.Tensor:
    """Pad Z/Y/X up so each is a whole number of ``multiple``, by replication.

    Replication rather than zeros: a zero voxel is a *dark* voxel here, which is
    a meaningful value the attention would then average over. Repeating the edge
    adds nothing the volume does not already say.
    """
    sz, sy, sx = x.shape[-3:]
    mz, my, mx = multiple
    pz, py, px = (-sz) % mz, (-sy) % my, (-sx) % mx
    if pz or py or px:
        x = F.pad(x, (0, px, 0, py, 0, pz), mode="replicate")
    return x


class _PooledSelfAttention3d(nn.Module):
    """Self-attention with a convolutional spatial reduction on the keys.

    PVT's spatial-reduction attention, in 3D. Every voxel of the stage is a
    query, but the keys and values are read from a *strided convolution* of the
    same feature map, so the token count they carry falls by ``reduction`` per
    axis. That is what makes a global receptive field affordable: at the default
    geometry the bottleneck is 32x32x32, and full self-attention there is
    32768^2 = 1.07e9 pairs per head per frame, where reducing by 4 leaves
    32768 x 512 = 16.8M.

    The projections are convolutions rather than linear layers, which is the
    "convolutional" half of the name and is not cosmetic: the K/V convolution
    has a 4x4x4 receptive field, so each key summarises a neighbourhood instead
    of sampling one voxel out of 64.

    The output projection is zero-initialised, so the block is exactly the
    identity at initialisation. Every stage of this pipeline fine-tunes from a
    pretrained checkpoint loaded with ``strict=False`` (tracking/main.py), so a
    freshly-inserted block must not perturb what those weights already do.
    """

    def __init__(self, channels: int, n_heads: int = 4, reduction=4) -> None:
        super().__init__()
        self.n_heads = _heads_for(channels, n_heads)
        self.reduction = _to_3tuple(reduction)
        self.scale = (channels // self.n_heads) ** -0.5

        self.norm = _norm("group", channels)
        self.q = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        # The spatial reduction is a learned pooling -- kernel == stride == the
        # reduction -- and it is separable: a depthwise convolution over the
        # 4x4x4 block, then a 1x1 projection to K and V. PVT writes this as one
        # dense convolution, which at these widths costs C * 2C * 64 = 2.1M
        # parameters, over half the backbone again for a single block. The
        # separable form keeps the same receptive field for 41k.
        self.kv = nn.Sequential(
            nn.Conv3d(
                channels, channels, kernel_size=self.reduction, stride=self.reduction,
                groups=channels, bias=False,
            ),
            nn.Conv3d(channels, 2 * channels, kernel_size=1, bias=False),
        )
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, Z, Y, X)
        n, c = x.shape[:2]
        spatial = x.shape[2:]
        h = self.norm(x)

        q = self.q(h).reshape(n, self.n_heads, c // self.n_heads, -1).transpose(-2, -1)
        k, v = self.kv(_pad_to_multiple(h, self.reduction)).chunk(2, dim=1)
        k = k.reshape(n, self.n_heads, c // self.n_heads, -1).transpose(-2, -1)
        v = v.reshape(n, self.n_heads, c // self.n_heads, -1).transpose(-2, -1)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -1).reshape(n, c, *spatial)
        return x + self.proj(out)


class _WindowSelfAttention3d(nn.Module):
    """Self-attention inside non-overlapping windows.

    Swin's partition without the shift: the volume is cut into ``window``-sized
    blocks and every voxel attends to the ones sharing its block. Cost is linear
    in the volume — 4x8x8 windows are 256 tokens each, whatever the stage size —
    but the receptive field stops at the window, and because the windows never
    move, the seams between them are never crossed. It is here to be measured
    against the pooled variant, which buys a global field for a similar price.

    Zero-initialised output projection, for the reason in
    :class:`_PooledSelfAttention3d`.
    """

    def __init__(self, channels: int, n_heads: int = 4, window=(4, 8, 8)) -> None:
        super().__init__()
        self.n_heads = _heads_for(channels, n_heads)
        self.window = _to_3tuple(window)

        self.norm = _norm("group", channels)
        self.qkv = nn.Conv3d(channels, 3 * channels, kernel_size=1, bias=False)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, Z, Y, X)
        n, c = x.shape[:2]
        spatial = x.shape[2:]
        # A stage smaller than the window attends over what is there rather than
        # padding a window's worth of replicated voxels into the softmax.
        window = tuple(min(w, s) for w, s in zip(self.window, spatial))

        h = _pad_to_multiple(self.norm(x), window)
        padded = h.shape[-3:]
        nz, ny, nx = (p // w for p, w in zip(padded, window))
        wz, wy, wx = window
        ws = wz * wy * wx

        qkv = self.qkv(h)
        # (N, 3C, Z, Y, X) -> one attention batch per window
        qkv = qkv.reshape(n, 3 * c, nz, wz, ny, wy, nx, wx)
        qkv = qkv.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(n * nz * ny * nx, 3 * c, ws)
        q, k, v = qkv.chunk(3, dim=1)

        def heads(t):
            return t.reshape(-1, self.n_heads, c // self.n_heads, ws).transpose(-2, -1)

        out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        out = out.transpose(-2, -1).reshape(n, nz, ny, nx, c, wz, wy, wx)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(n, c, *padded)
        out = out[:, :, : spatial[0], : spatial[1], : spatial[2]]
        return x + self.proj(out)


class _CBAM3d(nn.Module):
    """Convolutional channel + spatial attention gating (CBAM), in 3D.

    Not self-attention: there is no voxel-to-voxel interaction at all. A channel
    gate is pooled from the whole volume and a spatial gate from a convolution
    over the channel-mean and channel-max maps, and both re-weight the features
    in place. It is here as the cheap end of the same axis — if the gains of the
    two attention blocks above are really about re-weighting rather than about
    voxels talking to each other, this finds that out for almost no compute.

    The gates enter through a zero-initialised scalar, ``x * (1 + gamma * (g -
    1))``, so the block is the identity at initialisation for the same reason as
    the others. Applying ``g`` directly would multiply a pretrained feature map
    by ~0.25 at step 0.
    """

    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )
        self.spatial_conv = nn.Conv3d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.channel_mlp(F.adaptive_avg_pool3d(x, 1))
        pooled = pooled + self.channel_mlp(F.adaptive_max_pool3d(x, 1))
        gate = torch.sigmoid(pooled)

        h = x * gate
        stats = torch.cat([h.mean(dim=1, keepdim=True), h.amax(dim=1, keepdim=True)], dim=1)
        gate = gate * torch.sigmoid(self.spatial_conv(stats))
        return x * (1 + self.gamma * (gate - 1))


SPATIAL_ATTENTION_KINDS = ("none", "pooled", "window", "cbam")


def spatial_attention_block(
    kind: str,
    channels: int,
    n_heads: int = 4,
    reduction=4,
    window=(4, 8, 8),
) -> nn.Module:
    """One spatial attention block by name, or ``nn.Identity`` for ``"none"``.

    ``"pooled"`` attends globally over convolutionally reduced keys,
    ``"window"`` attends locally inside fixed windows, ``"cbam"`` only
    re-weights. See each class for what it costs and what it can see.
    """
    if kind in (None, "none"):
        return nn.Identity()
    if kind == "pooled":
        return _PooledSelfAttention3d(channels, n_heads=n_heads, reduction=reduction)
    if kind == "window":
        return _WindowSelfAttention3d(channels, n_heads=n_heads, window=window)
    if kind == "cbam":
        return _CBAM3d(channels)
    raise ValueError(
        f"unknown spatial_attention {kind!r}, expected one of {list(SPATIAL_ATTENTION_KINDS)}"
    )


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
    spatial_attention : {"none", "pooled", "window", "cbam"}
        A convolutional attention block on the *spatial* axes, run after each
        stage's convolutions and its temporal block. The temporal blocks mix a
        voxel with itself across the frames of the window and never with its
        neighbours; this is the other half. See ``spatial_attention_block``.
    spatial_attention_stages : int
        How many of the *deepest* encoder stages carry one. 1 (default) is the
        bottleneck alone, which is where the tokens are fewest and the context
        richest; 0 is off. The finer stages are much larger — at the default
        geometry the stage above the bottleneck holds 8x the voxels — so this
        is the knob that decides what the block costs.
    spatial_attention_heads : int
        Attention heads, reduced to a divisor of the stage width if it is not
        one already.
    spatial_attention_reduction : int or (int, int, int)
        ``"pooled"`` only: the stride of the convolution the keys and values are
        read through, per axis.
    spatial_attention_window : (int, int, int)
        ``"window"`` only: the window each voxel attends inside, clamped per
        axis to the stage that carries it.
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
        spatial_attention: str = "none",
        spatial_attention_stages: int = 1,
        spatial_attention_heads: int = 4,
        spatial_attention_reduction: int | tuple[int, int, int] = 4,
        spatial_attention_window: tuple[int, int, int] = (4, 8, 8),
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

        # The deepest `spatial_attention_stages` stages carry a spatial block.
        first_spatial = n_stages - max(0, int(spatial_attention_stages))

        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()
        for i, ch in enumerate(layers):
            stride = stem_conv_stride if i == 0 else (1, 1, 1)
            self.encoder_blocks.append(
                build_stage(prev, ch, blocks_per_stage[i] if blocks_per_stage else 1, stride)
            )
            if not temporal_attention or (skip_fullres_temporal and i == 0):
                self.temporal_blocks.append(nn.Identity())
            else:
                self.temporal_blocks.append(_TemporalAttention(ch))
            self.spatial_blocks.append(
                spatial_attention_block(
                    spatial_attention if i >= first_spatial else "none",
                    ch,
                    n_heads=spatial_attention_heads,
                    reduction=spatial_attention_reduction,
                    window=spatial_attention_window,
                )
            )
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
        for i, (pool, block, temporal, spatial) in enumerate(
            zip(self.pools, self.encoder_blocks, self.temporal_blocks, self.spatial_blocks)
        ):
            if self.verbose >= 1:
                print(f"TemporalUNet3D - Encoder Block {i} - Input shape: {x.shape}")

            x = pool(x)
            x = self._run(block, x)
            # Attention mixes the frames of one window, so it needs the time axis back.
            if not isinstance(temporal, nn.Identity):
                x = temporal(x.reshape(B, T, *x.shape[1:])).reshape(B * T, *x.shape[1:])
            # The spatial block works one frame at a time, so it stays folded.
            if not isinstance(spatial, nn.Identity):
                x = self._run(spatial, x)
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
