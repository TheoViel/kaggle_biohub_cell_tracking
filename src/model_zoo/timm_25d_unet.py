"""2.5D temporal U-Net: a pretrained 2D timm encoder with depth reduction.

Port of the CZII CryoET Object Identification 4th-place architecture (yu4u's
half, ``kaggle-czii-4th/src/timm_3d_decoder3.py``, arXiv:2502.13484) onto this
pipeline's backbone contract.

What the CZII model does, and what is kept here:

* Each Z slice is handed to a **2D ImageNet-pretrained encoder** together with
  its ``slice_size`` neighbours stacked into the channel axis (their
  ``SlidingSlicer``). Z lives in the batch dimension, so the whole ImageNet
  weight stack applies unchanged — that is the entire point of the design, and
  it is what this backbone brings that the hand-built 3D U-Nets here cannot.
* Every pyramid level is folded back to ``(B, C, Z, Y, X)`` and run through a
  small **3D convolution block** before it is used as a skip, so Z stops being
  a batch axis as early as the first level.
* Z is **reduced by average pooling** as Y/X shrink, rather than by the encoder
  — the "depth reduction" of the paper's title.
* A **3D U-Net decoder** climbs back up the pyramid, each block being
  ``interpolate → concat skip → two 3x3x3 convolutions``.

What is deliberately *not* kept: their ``channel_to_spatial_3d`` segmentation
head, which pixel-shuffles the decoder output back to full resolution. This
pipeline's heads (``models_track.UNetNodeTransformer``) read the feature map on
``output_stride``'s grid and do their own thing with it.

Geometry is this repo's, not theirs: ``layers`` / ``pool_stride`` /
``n_anisotropic`` / ``n_decode_steps`` / ``n_extra_decode_steps`` mean exactly
what they mean in ``temporal_unet.TemporalUNet3D``, and the default
``(64, 128, 256)`` with ``pool_stride=(1, 2, 2)`` and ``n_anisotropic=2`` puts
the head on the same ``(1, 4, 4)`` grid and the fine branch on the same
``(1, 2, 2)`` grid as every other backbone in the zoo.

The Y/X half of the stride schedule is the *encoder's*, though, and an ImageNet
backbone only offers the reductions it was built with. The level pyramid is
therefore assembled by matching the schedule's cumulative Y/X strides against
``feature_info.reduction()``: a backbone whose finest feature is at stride 2
(ResNet, EfficientNet) lines up level for level, and one that patchifies to
stride 4 (ConvNeXt, the CZII default) has its missing finest level supplied by
a strided/unshuffling 3D stem on the raw volume. The encoder is then run to its
own bottleneck rather than stopped at ``len(layers)`` levels, because stopping
early would throw away most of the pretrained weights and most of the receptive
field that justify using it; ``layers`` sets the width of the levels the
pipeline actually reads, and anything deeper reuses ``layers[-1]``.

Input shape  : ``(B, T, C_in, Z, Y, X)``
Output shape : ``(B, T, C_out, Z // sz, Y // sy, X // sx)``, ``output_stride``
being ``(sz, sy, sx)``, plus a second and finer map when
``n_extra_decode_steps > 0``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

from model_zoo.resenc_unet import anisotropic_strides
from model_zoo.temporal_unet import (
    _PixelUnshuffle3d,
    _TemporalAttention,
    _norm,
    _to_3tuple,
    spatial_attention_block,
)


def sliding_slices(x: torch.Tensor, slice_size: int) -> torch.Tensor:
    """Give every Z slice its ``slice_size`` neighbours, in the channel axis.

    ``x`` is ``(N, C, Z, Y, X)``, the result ``(N, Z, C * slice_size, Y, X)``.
    The CZII code does this with a frozen ``Conv3d`` whose weights spell out an
    identity; a pad and a slice is the same map with nothing to train, nothing
    to save and no multiply-accumulate over a mostly-zero kernel.
    """
    if slice_size == 1:
        return x.transpose(1, 2)
    pad = slice_size // 2
    # Replicate rather than zero-pad: a zero voxel is a *dark* voxel here, a
    # value the encoder would read as signal at the top and bottom of the stack.
    padded = F.pad(x, (0, 0, 0, 0, pad, slice_size - 1 - pad), mode="replicate")
    # unfold over Z -> (N, C, Z, Y, X, slice_size)
    windows = padded.unfold(2, slice_size, 1)
    n, c, z, y, x = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
    windows = windows.permute(0, 2, 1, 5, 3, 4)  # (N, Z, C, slice_size, Y, X)
    return windows.reshape(n, z, c * slice_size, y, x)


class _Conv3dNormAct(nn.Sequential):
    """CZII's ``Conv3dReLU``, with this repo's normalisation choices."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 norm: str = "batch") -> None:
        super().__init__(
            nn.Conv3d(
                in_channels, out_channels, kernel_size,
                padding=kernel_size // 2, bias=False,
            ),
            _norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )


def _conv3d_stack(in_channels: int, out_channels: int, n_blocks: int = 2,
                  norm: str = "batch") -> nn.Sequential:
    """``n_blocks`` 3x3x3 convolutions, the first one changing the width."""
    n_blocks = max(1, int(n_blocks))
    blocks = [_Conv3dNormAct(in_channels, out_channels, norm=norm)]
    blocks += [
        _Conv3dNormAct(out_channels, out_channels, norm=norm)
        for _ in range(n_blocks - 1)
    ]
    return nn.Sequential(*blocks)


class _DecoderBlock(nn.Module):
    """CZII's ``DecoderBlock``: upsample, concatenate the skip, two convolutions.

    Their ``k`` — 1 for the first block, 2 afterwards — was a hard-coded stand-in
    for a depth schedule. Here the per-axis ``scale`` comes from the stride
    schedule the whole zoo shares, so Z is upsampled exactly where it was pooled.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 scale: Sequence[int], norm: str = "batch") -> None:
        super().__init__()
        self.scale = tuple(float(s) for s in scale)
        self.conv1 = _Conv3dNormAct(in_channels + skip_channels, out_channels, norm=norm)
        self.conv2 = _Conv3dNormAct(out_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if any(s != 1.0 for s in self.scale):
            x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class Timm25DUNet3D(nn.Module):
    """2.5D temporal U-Net over a pretrained 2D timm encoder.

    Parameters
    ----------
    in_channels : int
        Input channels per frame.
    out_channels : int
        Output feature channels per frame.
    layers : sequence of int
        Widths of the pyramid levels the pipeline reads, finest first. Its
        *length* fixes what ``n_decode_steps`` counts against, exactly as in
        ``TemporalUNet3D``; the encoder's own depth past that reuses
        ``layers[-1]``. The encoder's widths are the backbone's and are projected
        onto these by each level's 3D block.
    blocks_per_stage : sequence of int or None
        Convolutions in each level's 3D block. ``None`` is CZII's two.
    gradient_checkpointing : bool
        Checkpoint the 2D encoder and the 3D blocks during training.
    temporal_attention : bool
        Per-voxel attention across the frames of a window, on the 3D level
        features. The 2D encoder is frame-independent by construction, so this
        is the only place the frames of a window meet.
    skip_fullres_temporal : bool
        Leave the finest level without a temporal block, where it is most
        expensive and least informative.
    spatial_attention, spatial_attention_stages, spatial_attention_heads,
    spatial_attention_reduction, spatial_attention_window
        As in ``TemporalUNet3D``: a block on the deepest ``stages`` levels.
    pool_stride, n_anisotropic
        The Z half of the stride schedule — how far Y/X may shrink before Z
        starts being pooled. The Y/X half is the encoder's, see the module
        docstring.
    stem : {"stride", "unshuffle"}
        Only consulted when the encoder's finest feature is coarser than the
        schedule's finest level and a 3D stem has to supply it.
        ``"unshuffle"`` folds the step into the channel axis instead of dropping
        voxels at it.
    norm : {"batch", "instance", "group"}
        Normalisation in the 3D blocks and the decoder. The 2D encoder keeps
        whatever its pretrained weights were trained with.
    n_decode_steps : int or None
        Decoder stages *counted against* ``len(layers)``, so that the head lands
        on the same grid it would with any other backbone. The decoder itself
        always starts at the encoder's true bottleneck.
    n_extra_decode_steps : int
        Decoder stages past the head's grid, feeding a second and finer map.
    hires_skip_fusion : bool
        Max-pool the levels finer than the head's grid back into the head.
    timm_backbone : str
        Any ``features_only``-capable timm model. The CZII default is
        ``"convnext_nano.in12k_ft_in1k"``.
    timm_slice_size : int
        Neighbouring Z slices stacked into the encoder's channel axis, CZII's
        ``model.in_channels``. 1 turns the stacking off.
    timm_pretrained : bool
        Load ImageNet weights. Turn it off when a checkpoint is about to
        overwrite them anyway.
    timm_drop_path_rate : float
        Stochastic depth in the 2D encoder.
    timm_max_levels : int or None
        Cap on how many of the encoder's pyramid levels are used, deepest
        dropped first. ``None`` uses all of them.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        layers: Sequence[int] = (64, 128, 256),
        blocks_per_stage: Sequence[int] | None = None,
        gradient_checkpointing: bool = False,
        skip_fullres_temporal: bool = True,
        temporal_attention: bool = True,
        spatial_attention: str = "none",
        spatial_attention_stages: int = 1,
        spatial_attention_heads: int = 4,
        spatial_attention_reduction: int | tuple[int, int, int] = 4,
        spatial_attention_window: tuple[int, int, int] = (4, 8, 8),
        pool_stride: int | tuple[int, int, int] = (1, 2, 2),
        n_anisotropic: int = 2,
        stem: str = "stride",
        norm: str = "batch",
        n_decode_steps: int | None = None,
        n_extra_decode_steps: int = 0,
        hires_skip_fusion: bool = False,
        timm_backbone: str = "convnext_nano.in12k_ft_in1k",
        timm_slice_size: int = 5,
        timm_pretrained: bool = True,
        timm_drop_path_rate: float = 0.0,
        timm_max_levels: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        layers = list(layers)
        if len(layers) < 2:
            raise ValueError("layers must contain at least two stages")

        pool_stride_t = _to_3tuple(pool_stride)
        if pool_stride_t[1] != pool_stride_t[2]:
            raise ValueError(
                f"the 2D encoder downsamples Y and X together, so pool_stride's "
                f"last two entries must match, got {pool_stride_t}"
            )
        pool_yx = pool_stride_t[1]
        if pool_yx < 2:
            raise ValueError(f"pool_stride must downsample Y/X, got {pool_stride_t}")
        if stem not in ("stride", "unshuffle"):
            raise ValueError(f"unknown stem {stem!r}, expected 'stride' / 'unshuffle'")

        self.gradient_checkpointing = gradient_checkpointing
        self.verbose = verbose
        self.slice_size = max(1, int(timm_slice_size))
        self.in_channels = in_channels

        # --- 2D encoder -----------------------------------------------------
        extra: dict = {}
        if timm_max_levels is not None:
            # How many levels the backbone offers is only knowable from an
            # instance, so probe one without weights before asking for the real
            # thing -- `out_indices` past the last level is an error, not a clamp.
            probe = timm.create_model(timm_backbone, features_only=True, pretrained=False)
            keep = max(1, min(int(timm_max_levels), len(probe.feature_info.channels())))
            extra["out_indices"] = tuple(range(keep))
            del probe
        self.encoder = timm.create_model(
            timm_backbone,
            features_only=True,
            in_chans=in_channels * self.slice_size,
            pretrained=bool(timm_pretrained),
            drop_path_rate=float(timm_drop_path_rate),
            **extra,
        )
        enc_channels = list(self.encoder.feature_info.channels())
        enc_reduction = [int(r) for r in self.encoder.feature_info.reduction()]
        self.n_encoder_levels = len(enc_channels)

        # --- level pyramid --------------------------------------------------
        # The schedule's finest Y/X stride is the stem's. Levels the encoder is
        # too coarse to provide are built in 3D from the raw volume instead.
        stem_stride, _ = anisotropic_strides(
            len(layers), pool_stride=pool_stride_t, n_anisotropic=n_anisotropic
        )
        yx_levels: list[int] = []
        yx = stem_stride[1]
        while yx < enc_reduction[0]:
            yx_levels.append(yx)
            yx *= pool_yx
        self.n_stem_levels = len(yx_levels)
        yx_levels += enc_reduction

        for coarse, fine in zip(yx_levels[1:], yx_levels):
            ratio = coarse / fine
            if ratio < 1 or abs(math.log(ratio, pool_yx) - round(math.log(ratio, pool_yx))) > 1e-6:
                raise ValueError(
                    f"{timm_backbone!r} reduces Y/X by {ratio:g} between pyramid "
                    f"levels, which pool_stride={pool_stride_t} cannot express"
                )

        # Z follows the schedule: it is left alone for the first `n_anisotropic`
        # downsampling steps and then halves with Y/X, which is
        # `anisotropic_strides` read off a chain the encoder chose.
        iso = max(pool_stride_t)
        level_strides: list[tuple[int, int, int]] = []
        for yx in yx_levels:
            steps = round(math.log(yx, pool_yx))
            z = 1
            for step in range(1, steps + 1):
                z *= pool_stride_t[0] if step <= n_anisotropic else iso
            level_strides.append((int(z), int(yx), int(yx)))
        self.level_strides = tuple(level_strides)
        n_levels = len(level_strides)

        def width(i: int) -> int:
            return layers[min(i, len(layers) - 1)]

        if blocks_per_stage is not None:
            blocks_per_stage = [int(b) for b in blocks_per_stage]
            if len(blocks_per_stage) != len(layers):
                raise ValueError(
                    f"blocks_per_stage must have {len(layers)} entries to match "
                    f"layers, got {len(blocks_per_stage)}"
                )

        def n_blocks(i: int) -> int:
            if blocks_per_stage is None:
                return 2  # CZII's two 3x3x3 convolutions per level
            return blocks_per_stage[min(i, len(blocks_per_stage) - 1)]

        # --- 3D stems for the levels the encoder cannot reach ----------------
        self.stems = nn.ModuleList()
        for i in range(self.n_stem_levels):
            stride = level_strides[i]
            if stem == "unshuffle":
                first: nn.Module = nn.Sequential(
                    _PixelUnshuffle3d(stride),
                    _Conv3dNormAct(
                        in_channels * math.prod(stride), width(i), norm=norm
                    ),
                )
            else:
                first = nn.Sequential(
                    nn.Conv3d(
                        in_channels, width(i), kernel_size=3, padding=1,
                        stride=stride, bias=False,
                    ),
                    _norm(norm, width(i)),
                    nn.ReLU(inplace=True),
                )
            self.stems.append(
                nn.Sequential(first, _conv3d_stack(width(i), width(i), n_blocks(i) - 1, norm))
                if n_blocks(i) > 1
                else first
            )

        # --- per-level 3D blocks (CZII's `conv3ds`) --------------------------
        # These are what take a level out of "Z is a batch axis" and into 3D, and
        # they double as the projection from the backbone's widths onto `layers`.
        self.conv3ds = nn.ModuleList(
            [
                _conv3d_stack(enc_channels[j], width(self.n_stem_levels + j), n_blocks(self.n_stem_levels + j), norm)
                for j in range(self.n_encoder_levels)
            ]
        )

        first_spatial = n_levels - max(0, int(spatial_attention_stages))
        self.temporal_blocks = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()
        for i in range(n_levels):
            if not temporal_attention or (skip_fullres_temporal and i == 0):
                self.temporal_blocks.append(nn.Identity())
            else:
                self.temporal_blocks.append(_TemporalAttention(width(i)))
            self.spatial_blocks.append(
                spatial_attention_block(
                    spatial_attention if i >= first_spatial else "none",
                    width(i),
                    n_heads=spatial_attention_heads,
                    reduction=spatial_attention_reduction,
                    window=spatial_attention_window,
                )
            )

        # --- decoder ---------------------------------------------------------
        # `n_decode_steps` is counted against len(layers) so the head lands where
        # it would on any other backbone; the decoder runs from the encoder's own
        # bottleneck regardless of how much deeper that is.
        n_full_decode = len(layers) - 1
        if n_decode_steps is None:
            n_decode_steps = n_full_decode
        n_decode_steps = max(0, min(int(n_decode_steps), n_full_decode))
        head_stage = min(n_full_decode - n_decode_steps, n_levels - 1)

        def decoder_block(i: int) -> _DecoderBlock:
            """The block that lifts level ``i + 1``'s features onto level ``i``."""
            scale = tuple(
                c // f for c, f in zip(level_strides[i + 1], level_strides[i])
            )
            return _DecoderBlock(width(i + 1), width(i), width(i), scale, norm)

        self.decoder_blocks = nn.ModuleList(
            [decoder_block(i) for i in range(n_levels - 2, head_stage - 1, -1)]
        )

        n_extra_decode_steps = max(0, min(int(n_extra_decode_steps), head_stage))
        self.fine_decoder_blocks = nn.ModuleList(
            [
                decoder_block(i)
                for i in range(head_stage - 1, head_stage - n_extra_decode_steps - 1, -1)
            ]
        )
        fine_stage = head_stage - n_extra_decode_steps
        self.fine_head = (
            nn.Conv3d(width(fine_stage), out_channels, kernel_size=1)
            if n_extra_decode_steps
            else None
        )

        head_in = width(head_stage)
        self.hires_pools = nn.ModuleList()
        self.hires_projs = nn.ModuleList()
        if hires_skip_fusion:
            fuse_channels = max(8, layers[0] // 2)
            for j in range(head_stage):
                factor = tuple(
                    c // f for c, f in zip(level_strides[head_stage], level_strides[j])
                )
                self.hires_pools.append(nn.MaxPool3d(kernel_size=factor, stride=factor))
                self.hires_projs.append(nn.Conv3d(width(j), fuse_channels, kernel_size=1))
                head_in += fuse_channels

        self.head = nn.Conv3d(head_in, out_channels, kernel_size=1)

        self.out_channels = out_channels
        self.n_decode_steps = n_decode_steps
        self.n_extra_decode_steps = n_extra_decode_steps
        self.head_stage = head_stage
        self.divisible_by = level_strides[-1]
        self.output_stride = level_strides[head_stage]
        self.fine_output_stride = level_strides[fine_stage]

        if verbose >= 1:
            print(
                f"Timm25DUNet3D - {timm_backbone} - levels {self.level_strides} "
                f"({self.n_stem_levels} from a 3D stem) - output_stride "
                f"{self.output_stride} - fine {self.fine_output_stride}"
            )

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.gradient_checkpointing = enable

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return _grad_ckpt(block, x, use_reentrant=False)
        return block(x)

    def _encode_levels(self, vol: torch.Tensor, B: int, T: int) -> list[torch.Tensor]:
        """Per-level 3D features for a flat ``(B * T, C, Z, Y, X)`` batch."""
        n, _, depth = vol.shape[:3]

        levels: list[torch.Tensor] = [self._run(s, vol) for s in self.stems]

        if self.n_encoder_levels:
            slices = sliding_slices(vol, self.slice_size)  # (N, Z, C * s, Y, X)
            slices = slices.reshape(n * depth, *slices.shape[2:])
            # The encoder is checkpointed as one block rather than through
            # timm's own `set_grad_checkpointing`: the ResNet family reuses its
            # `inplace=True` activations across the stage boundaries timm
            # checkpoints at, which the recompute then trips over.
            feats = self._run(self.encoder, slices)
            for j, (feat, block) in enumerate(zip(feats, self.conv3ds)):
                # (N * Z, c, h, w) -> (N, c, Z, h, w)
                c, h, w = feat.shape[1:]
                feat = feat.reshape(n, depth, c, h, w).transpose(1, 2)
                # Depth reduction: the schedule's Z stride for this level, taken
                # in one pool rather than CZII's cascade of equivalent halvings.
                k = self.level_strides[self.n_stem_levels + j][0]
                k = min(k, feat.shape[2])
                if k > 1:
                    feat = F.avg_pool3d(feat, (k, 1, 1), (k, 1, 1))
                levels.append(self._run(block, feat))

        for i, (temporal, spatial) in enumerate(
            zip(self.temporal_blocks, self.spatial_blocks)
        ):
            if not isinstance(temporal, nn.Identity):
                shape = levels[i].shape[1:]
                levels[i] = temporal(levels[i].reshape(B, T, *shape)).reshape(-1, *shape)
            if not isinstance(spatial, nn.Identity):
                levels[i] = self._run(spatial, levels[i])
            if self.verbose >= 1:
                print(f"Timm25DUNet3D - Level {i} shape: {levels[i].shape}")

        return levels

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C_in, Z, Y, X) -> (B, T, C_out, Z', Y', X'), plus a second
        # and finer map of the same layout when n_extra_decode_steps > 0.
        B, T = x.shape[:2]
        if self.verbose >= 1:
            print(f"Timm25DUNet3D - Input shape: {x.shape}")

        levels = self._encode_levels(x.reshape(B * T, *x.shape[2:]), B, T)

        out = levels[-1]
        for i, block in enumerate(self.decoder_blocks):
            out = self._run_decoder(block, out, levels[len(levels) - 2 - i])

        fine = None
        if self.fine_head is not None:
            fine = out
            for j, block in enumerate(self.fine_decoder_blocks):
                fine = self._run_decoder(block, fine, levels[self.head_stage - 1 - j])
            fine = self.fine_head(fine)

        if self.hires_projs:
            detail = []
            for proj, pool, skip in zip(self.hires_projs, self.hires_pools, levels):
                d = proj(pool(skip))
                if d.shape[2:] != out.shape[2:]:
                    d = F.interpolate(
                        d, size=out.shape[2:], mode="trilinear", align_corners=False
                    )
                detail.append(d)
            out = torch.cat([out, *detail], dim=1)

        out = self.head(out)
        if self.verbose >= 1:
            print(f"Timm25DUNet3D - Output shape: {out.shape}  (stride {self.output_stride})")

        out = out.reshape(B, T, *out.shape[1:])
        if fine is None:
            return out
        return out, fine.reshape(B, T, *fine.shape[1:])

    def _run_decoder(self, block: nn.Module, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return _grad_ckpt(block, x, skip, use_reentrant=False)
        return block(x, skip)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
