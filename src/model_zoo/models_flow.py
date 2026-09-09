"""The Cellpose flow model built on the tracking pipeline's own U-Net.

The tracking model *is* a segmentation backbone with heads bolted on -- see
``UNetNodeTransformer``, which holds a ``self.unet`` and reads detection and
edge features off it. Building the segmenter from the same backbone, with the
same ``unet_*`` configuration names, buys two things:

* One vocabulary. ``unet_layers``, ``unet_pool_stride``, ``unet_n_anisotropic``
  and the rest mean the same thing in ``main_seg`` as in ``main_track``, so a
  geometry that works for one is transferable by copying the numbers across
  rather than translating them.
* Transferable weights. The backbone is stored under ``self.unet``, which is
  exactly the prefix ``tracking.main.load_weights`` pulls out of a checkpoint,
  so a segmentation run's ``ckpt_final.pt`` can be handed straight to a
  tracking run's ``pretraining.weights``. Cellpose flows are a far denser
  training signal than sparse node annotations -- every voxel of every cell
  carries a target -- so that is a pretraining path worth having.

The one place the two heads genuinely differ is output resolution. Detection
reads a heatmap and is happy on the U-Net's own coarse grid; a flow is a vector
*per voxel* and the reconstruction integrates it at voxel scale, so the head has
to come back to full resolution. The backbone never does on its own -- its stem
downsamples and no decoder undoes that step -- so the head carries a learned
upsample sized from ``output_stride``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_zoo.models_track import UNET_BACKBONES

# Cellpose in 3D: interior probability plus the z/y/x flow field.
FLOW_HEADS = {"cellprob_logits": 1, "flows": 3}


class FlowUNet(nn.Module):
    """Tracking backbone + a full-resolution Cellpose head.

    Attributes:
        unet: the shared backbone. Named to match ``UNetNodeTransformer`` so
            ``load_weights`` can move it between the two pipelines.
        divisible_by: per-axis multiple the input spatial size must be padded to.
    """

    def __init__(self, unet, heads=None, dropout: float = 0.0):
        super().__init__()
        from model_zoo.models_seg import split_heads

        self._split_heads = split_heads
        self.unet = unet
        self.heads = heads or FLOW_HEADS
        self.output_stride = tuple(unet.output_stride)
        self.divisible_by = tuple(unet.divisible_by)

        c = unet.out_channels
        nout = sum(self.heads.values())

        # A transposed convolution rather than an interpolation: the flow near a
        # cell centre turns over within a voxel or two, and resampling a coarse
        # field up to the grid it will be integrated on smears exactly the part
        # that decides where the trajectories stop.
        if any(s > 1 for s in self.output_stride):
            self.up = nn.Sequential(
                nn.ConvTranspose3d(
                    c, c, kernel_size=self.output_stride, stride=self.output_stride
                ),
                nn.GELU(),
            )
        else:
            self.up = nn.Identity()

        # Same shape as the tracking model's detect_head, for the same reason:
        # if the two pipelines share a backbone they may as well share the way
        # they read it.
        self.head = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(c, nout, kernel_size=1),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, C, Z, Y, X) -> per-head logits at the input's resolution."""
        # The backbone is temporal and wants a window axis; a still volume is a
        # window of one, which also disables the temporal attention blocks by
        # giving them nothing to attend to.
        feat = self.unet(x[:, None])[:, 0]
        out = self.head(self.up(feat))
        if out.shape[-3:] != x.shape[-3:]:
            raise RuntimeError(
                f"head produced {tuple(out.shape[-3:])} for input "
                f"{tuple(x.shape[-3:])}; flows must be voxel-aligned. Check "
                f"unet_pool_stride / unet_n_decode_steps."
            )
        return self._split_heads(out, self.heads)


def define_flow_model(
    unet_arch: str = "temporal",
    unet_out_channels: int = 32,
    unet_layers=(32, 64, 128),
    unet_pool_stride=(1, 2, 2),
    unet_n_anisotropic: int = 2,
    unet_n_decode_steps: int | None = None,
    unet_temporal_attention: bool = False,
    unet_spatial_attention: str = "none",
    unet_spatial_attention_stages: int = 1,
    unet_spatial_attention_heads: int = 4,
    unet_spatial_attention_reduction=4,
    unet_spatial_attention_window=(4, 8, 8),
    unet_blocks_per_stage=None,
    unet_stem: str = "unshuffle",
    unet_norm: str = "batch",
    unet_hires_fusion: bool = True,
    in_channels: int = 1,
    heads=None,
    dropout: float = 0.0,
    gradient_checkpointing: bool = False,
    verbose: int = 0,
) -> FlowUNet:
    """Build a :class:`FlowUNet`, taking ``define_track_model``'s arguments.

    ``unet_temporal_attention`` defaults to False here and not in the tracking
    model: a static volume is a window of one frame, so the attention has
    nothing to mix and only adds parameters that will never see a gradient
    signal worth anything. Leave it True if the weights are headed for a
    tracking run that wants those blocks initialised.
    """
    if unet_arch not in UNET_BACKBONES:
        raise ValueError(
            f"unknown unet_arch {unet_arch!r}, expected one of {sorted(UNET_BACKBONES)}"
        )

    backbone = UNET_BACKBONES[unet_arch]
    # Stem layout, normalisation and skip fusion are TemporalUNet3D's; the
    # ResEnc encoder fixes all three by construction.
    extra = (
        {}
        if unet_arch == "resenc"
        else {
            "stem": unet_stem,
            "norm": unet_norm,
            "hires_skip_fusion": unet_hires_fusion,
        }
    )
    # Both backbones take these; "none" leaves the model unchanged. Kept here so
    # a segmentation run can pretrain the same blocks the tracking model wants.
    extra["spatial_attention"] = unet_spatial_attention
    extra["spatial_attention_stages"] = unet_spatial_attention_stages
    extra["spatial_attention_heads"] = unet_spatial_attention_heads
    extra["spatial_attention_reduction"] = unet_spatial_attention_reduction
    extra["spatial_attention_window"] = tuple(unet_spatial_attention_window)

    unet = backbone(
        in_channels=in_channels,
        out_channels=unet_out_channels,
        layers=list(unet_layers),
        blocks_per_stage=unet_blocks_per_stage,
        gradient_checkpointing=gradient_checkpointing,
        pool_stride=tuple(unet_pool_stride) if unet_pool_stride is not None else 2,
        n_anisotropic=unet_n_anisotropic,
        n_decode_steps=unet_n_decode_steps,
        temporal_attention=unet_temporal_attention,
        verbose=verbose,
        **extra,
    )

    model = FlowUNet(unet, heads=heads, dropout=dropout)
    if verbose:
        print(
            f"\n-> FlowUNet ({unet_arch})  {model.count_parameters() / 1e6:.1f}M parameters"
            f"\n   layers={list(unet_layers)}  pool_stride={tuple(unet_pool_stride)}  "
            f"n_anisotropic={unet_n_anisotropic}"
            f"\n   backbone output_stride={model.output_stride} -> head upsamples to "
            f"full resolution;  divisible_by={model.divisible_by}\n"
        )
    return model
