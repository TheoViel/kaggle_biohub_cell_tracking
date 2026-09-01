"""nnU-Net ResEnc backbone for the tracking pipeline, as a drop-in for TemporalUNet3D.

Input shape  : ``(B, T, C_in, Z, Y, X)``
Output shape : ``(B, T, C_out, Z // sz, Y // sy, X // sx)``, ``output_stride`` being ``(sz, sy, sx)``

Same interface and geometry as ``model_zoo.temporal_unet.TemporalUNet3D`` -- anisotropic
downsampling for the first steps, partial decoding, per-voxel temporal attention between
encoder stages -- over the residual encoder and decoder of ``model_zoo.resenc_unet`` instead
of the two-conv blocks. See that module for what the ResEnc architecture is.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_ckpt

from model_zoo.resenc_unet import (
    BASE_NUM_FEATURES,
    BLOCKS_PER_STAGE_DECODER,
    BLOCKS_PER_STAGE_ENCODER,
    MAX_FEATURES_3D,
    ResidualEncoder,
    UNetDecoder,
    anisotropic_strides,
    cumulative_stride,
    init_last_norm_before_add_to_zero,
    init_weights_he,
)
from model_zoo.temporal_unet import _TemporalAttention


class TemporalResEncUNet3D(nn.Module):
    """3D+T U-Net with an nnU-Net residual encoder.

    Parameters
    ----------
    in_channels : int
        Input channels per frame.
    out_channels : int
        Output feature channels per frame.
    layers : sequence of int
        Encoder channel widths, shallow to deep.  The number of stages is
        ``len(layers)``.  Defaults to nnU-Net's own schedule, 32 doubling per
        stage and capped at 320.
    blocks_per_stage : sequence of int or None
        Residual blocks per encoder stage.  Defaults to nnU-Net's
        ``(1, 3, 4, 6, 6, ...)``, which is what makes this deeper than the
        plain two-conv ``TemporalUNet3D`` at the same widths.
    gradient_checkpointing : bool
        If True, wrap encoder and decoder stages with
        ``torch.utils.checkpoint`` during training to trade compute for
        activation memory.
    skip_fullres_temporal : bool
        If True (default), the first encoder stage gets no temporal attention
        block — it is the widest feature map and the least informative one to
        attend over.
    temporal_attention : bool
        If False, drop the temporal attention blocks entirely and treat the
        window as a batch of independent frames.
    pool_stride : int or (int, int, int)
        The anisotropic downsampling stride, applied to the first
        ``n_anisotropic`` steps.  ``(1, 2, 2)`` pools Y/X only, keeping Z at
        full resolution while the coarser Z spacing catches up.
    n_anisotropic : int
        How many downsampling steps use ``pool_stride`` before the rest go
        isotropic.  Ignored when ``strides`` is given.
    stem_strides : bool
        If True (default), the stem takes the first downsampling step, so only
        its single convolution ever runs at full resolution.
    strides : sequence or None
        Per-stage strides, overriding ``pool_stride`` / ``n_anisotropic``
        entirely.  Pair with ``stem_stride``.
    stem_stride : int, (int, int, int) or None
        Stride of the stem, overriding ``stem_strides``.
    n_decode_steps : int or None
        Number of decoder stages, counting up from the bottleneck.  None
        (default) decodes all the way back to the stem's resolution.  0 skips
        the decoder entirely and the head reads the bottleneck.  Anything less
        than full leaves the output coarser than the input, by
        ``output_stride``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        layers: Sequence[int] | None = None,
        blocks_per_stage: Sequence[int] | None = None,
        gradient_checkpointing: bool = False,
        skip_fullres_temporal: bool = True,
        temporal_attention: bool = True,
        pool_stride: int | tuple[int, int, int] = 2,
        n_anisotropic: int = 2,
        stem_strides: bool = True,
        strides: Sequence | None = None,
        stem_stride: int | tuple[int, int, int] | None = None,
        n_stages: int = 3,
        n_decode_steps: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()

        if layers is None:
            layers = [min(MAX_FEATURES_3D, BASE_NUM_FEATURES * 2**i) for i in range(n_stages)]
        layers = list(layers)
        if len(layers) < 2:
            raise ValueError("layers must contain at least two stages")
        n_stages = len(layers)

        if blocks_per_stage is None:
            blocks_per_stage = BLOCKS_PER_STAGE_ENCODER[:n_stages]
        if len(blocks_per_stage) != n_stages:
            raise ValueError(f"blocks_per_stage must have {n_stages} entries, got {len(blocks_per_stage)}")

        pool_stride = (pool_stride,) * 3 if isinstance(pool_stride, int) else tuple(pool_stride)
        planned_stem, planned_strides = anisotropic_strides(
            n_stages, pool_stride=pool_stride, n_anisotropic=n_anisotropic, stem_strides=stem_strides
        )
        strides = planned_strides if strides is None else strides
        stem_stride = planned_stem if stem_stride is None else stem_stride

        self.gradient_checkpointing = gradient_checkpointing
        self.verbose = verbose

        # nnU-Net's 3D settings: instance norm, leaky ReLU, biased convolutions.
        self.encoder = ResidualEncoder(
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=layers,
            conv_op=nn.Conv3d,
            kernel_sizes=[3] * n_stages,
            strides=strides,
            n_blocks_per_stage=list(blocks_per_stage),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            return_skips=True,
            stem_stride=stem_stride,
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes=out_channels,
            n_conv_per_stage=BLOCKS_PER_STAGE_DECODER[: n_stages - 1],
            deep_supervision=False,
            n_decode_steps=n_decode_steps,
        )

        self.temporal_blocks = nn.ModuleList(
            [
                _TemporalAttention(ch)
                if temporal_attention and not (skip_fullres_temporal and i == 0)
                else nn.Identity()
                for i, ch in enumerate(layers)
            ]
        )

        self.out_channels = out_channels
        self.n_decode_steps = self.decoder.n_decode_steps
        # Spatial sizes must be divisible by this; the encoder runs every stage regardless of
        # how far back the decoder goes.
        self.divisible_by = cumulative_stride(self.encoder.stem_stride, self.encoder.strides)
        # How much coarser the output is than the input, which is what the tracking pipeline
        # measures at runtime as unet_scaling.
        self.output_stride = cumulative_stride(
            self.encoder.stem_stride, self.encoder.strides[: n_stages - self.n_decode_steps]
        )

        self.apply(init_weights_he)
        self.apply(init_last_norm_before_add_to_zero)

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return _grad_ckpt(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in, Z, Y, X) -> (B, T, C_out, Z', Y', X')
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])

        if self.verbose >= 1:
            print(f"TemporalResEncUNet3D - Input shape: {x.shape}")

        x = self._run(self.encoder.stem, x)

        skips: list[torch.Tensor] = []
        for i, (stage, temporal) in enumerate(zip(self.encoder.stages, self.temporal_blocks)):
            x = self._run(stage, x)
            # Attention mixes the frames of one window, so it needs the time axis back.
            if not isinstance(temporal, nn.Identity):
                x = temporal(x.reshape(B, T, *x.shape[1:])).reshape(B * T, *x.shape[1:])
            skips.append(x)

            if self.verbose >= 1:
                print(f"TemporalResEncUNet3D - Encoder stage {i} output shape: {x.shape}")

        if self.gradient_checkpointing and self.training:
            # Unpack the skips: checkpoint only tracks tensors it receives as its own arguments.
            x = _grad_ckpt(lambda *s: self.decoder(list(s)), *skips, use_reentrant=False)
        else:
            x = self.decoder(skips)

        if self.verbose >= 1:
            print(f"TemporalResEncUNet3D - Output shape: {x.shape}  (stride {self.output_stride})")

        return x.reshape(B, T, *x.shape[1:])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
