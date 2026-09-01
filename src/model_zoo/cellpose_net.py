"""CPnet, the Cellpose network, in native 3D.

A port of ``cellpose/resnet_torch.py`` from https://github.com/MouseLand/cellpose
(Stringer et al., Nat. Methods 2021) with every 2-D operator replaced by its 3-D
counterpart. The parts that make it CPnet rather than a generic U-Net, all kept:

* **Pre-activation residual blocks.** Each stage is four convolutions ordered
  BatchNorm -> ReLU -> Conv, wired as two residual pairs over a 1x1 projection
  (``resdown`` / ``resup``).
* **The style vector.** The deepest feature map is global-average-pooled and
  L2-normalised into a single vector, which is projected by a per-block Linear
  and *added* to the features of every decoder convolution. It is what lets a
  decision anywhere in the image depend on the image as a whole -- how bright
  the field is, how crowded, how large the cells run -- which for a flow field
  matters more than it would for a class label, because the correct vector at a
  voxel depends on how far away its cell's centre is.
* **Additive skips.** ``concatenation=False`` is the paper's default: the
  encoder feature is added into the decoder block rather than concatenated, so
  the decoder width does not double at every level.

The one deliberate departure is pooling. Cellpose halves both axes at every
stage because its images are isotropic; here a voxel is 1.625 um in z against
0.40625 in y/x, so isotropic pooling would spend the z axis four times faster
than it should and, on a 16-slice patch, exhaust it after four stages. Strides
are therefore per stage and default to pooling y/x first, the same shape of
schedule the repo's ResEnc U-Net plans for this data. Pass ``strides`` to get
the original isotropic behaviour.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cellpose's own default widths and kernel size.
CELLPOSE_NBASE = (32, 64, 128, 256)
CELLPOSE_SZ = 3

# Anisotropy-aware pooling: y/x only for the first stage, then isotropic. On a
# (16, 64, 64) patch this leaves 2 slices at the bottleneck instead of 1.
DEFAULT_STRIDES_3D = ((1, 2, 2), (2, 2, 2), (2, 2, 2))


def plan_strides(patch_size, spacing=(1.625, 0.40625, 0.40625), n_stages=3,
                 min_size=2):
    """Pooling schedule that equalises physical spacing before going isotropic.

    An axis is pooled only once its spacing is within 2x of the finest axis, so
    the coarse z axis waits while y/x catch up. On this data -- 1.625 um in z
    against 0.40625 in y/x -- that pools y/x once alone, after which a voxel is
    1.625 um cubed and the rest of the schedule is isotropic, which is where
    ``DEFAULT_STRIDES_3D`` comes from. An axis with too little left to give is
    held at stride 1 rather than pooled to nothing.
    """
    sp = [float(s) for s in spacing]
    size = [int(s) for s in patch_size]
    strides = []
    for _ in range(n_stages):
        finest = min(sp)
        s = [
            2 if (sp[a] <= 2 * finest and size[a] // 2 >= min_size) else 1
            for a in range(3)
        ]
        strides.append(tuple(s))
        for a in range(3):
            sp[a] *= s[a]
            size[a] //= s[a]
    return tuple(strides)


def batchconv(in_channels, out_channels, sz):
    """BatchNorm -> ReLU -> Conv, the pre-activation order Cellpose uses."""
    return nn.Sequential(
        nn.BatchNorm3d(in_channels, eps=1e-5, momentum=0.05),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz):
    """The projection path: no ReLU, so the residual branch stays linear."""
    return nn.Sequential(
        nn.BatchNorm3d(in_channels, eps=1e-5, momentum=0.05),
        nn.Conv3d(in_channels, out_channels, sz, padding=sz // 2),
    )


class ResDown(nn.Module):
    """Four convolutions as two residual pairs, over a 1x1x1 projection."""

    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.proj = batchconv0(in_channels, out_channels, 1)
        self.conv = nn.Sequential(
            batchconv(in_channels, out_channels, sz),
            batchconv(out_channels, out_channels, sz),
            batchconv(out_channels, out_channels, sz),
            batchconv(out_channels, out_channels, sz),
        )

    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        return x + self.conv[3](self.conv[2](x))


class ConvDown(nn.Module):
    """The non-residual encoder stage, for ``residual_on=False``."""

    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential(
            batchconv(in_channels, out_channels, sz),
            batchconv(out_channels, out_channels, sz),
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    """The encoder: one block per stage, max-pooled between them."""

    def __init__(self, nbase, sz, residual_on=True, strides=DEFAULT_STRIDES_3D):
        super().__init__()
        block = ResDown if residual_on else ConvDown
        self.strides = tuple(tuple(s) for s in strides)
        self.pools = nn.ModuleList(
            [nn.MaxPool3d(kernel_size=s, stride=s) for s in self.strides]
        )
        self.down = nn.ModuleList(
            [block(nbase[n], nbase[n + 1], sz) for n in range(len(nbase) - 1)]
        )

    def forward(self, x):
        feats = []
        for n, stage in enumerate(self.down):
            y = x if n == 0 else self.pools[n - 1](feats[n - 1])
            feats.append(stage(y))
        return feats


class MakeStyle(nn.Module):
    """The image's global descriptor: pooled bottleneck features, L2-normalised.

    Normalising is the point -- it makes the vector a *direction*, describing
    what kind of image this is rather than how bright it happens to be, so the
    decoder cannot read overall intensity off it and use it as a shortcut.
    """

    def forward(self, x):
        style = F.avg_pool3d(x, kernel_size=x.shape[-3:]).flatten(1)
        return style / (style.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)


class BatchConvStyle(nn.Module):
    """A conv whose input has the projected style vector added to it."""

    def __init__(self, in_channels, out_channels, style_channels, sz,
                 concatenation=False):
        super().__init__()
        self.concatenation = concatenation
        self.conv = batchconv(in_channels, out_channels, sz)
        self.full = nn.Linear(style_channels, in_channels)

    def forward(self, style, x, y=None):
        if y is not None:
            x = torch.cat((y, x), dim=1) if self.concatenation else x + y
        # Broadcast one scalar per channel across the whole volume.
        x = x + self.full(style)[..., None, None, None]
        return self.conv(x)


class ResUp(nn.Module):
    """The decoder stage: ``ResDown``'s shape, with style added to each conv."""

    def __init__(self, in_channels, out_channels, style_channels, sz,
                 concatenation=False):
        super().__init__()
        self.proj = batchconv0(in_channels, out_channels, 1)
        self.conv = nn.ModuleList([
            batchconv(in_channels, out_channels, sz),
            BatchConvStyle(
                out_channels * (2 if concatenation else 1), out_channels,
                style_channels, sz, concatenation=concatenation,
            ),
            BatchConvStyle(out_channels, out_channels, style_channels, sz),
            BatchConvStyle(out_channels, out_channels, style_channels, sz),
        ])

    def forward(self, x, y, style):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y)
        return x + self.conv[3](style, self.conv[2](style, x))


class ConvUp(nn.Module):
    """The non-residual decoder stage, for ``residual_on=False``."""

    def __init__(self, in_channels, out_channels, style_channels, sz,
                 concatenation=False):
        super().__init__()
        self.conv = nn.ModuleList([
            batchconv(in_channels, out_channels, sz),
            BatchConvStyle(
                out_channels * (2 if concatenation else 1), out_channels,
                style_channels, sz, concatenation=concatenation,
            ),
        ])

    def forward(self, x, y, style):
        return self.conv[1](style, self.conv[0](x), y=y)


class Upsample(nn.Module):
    """The decoder, walking back up the same stride schedule."""

    def __init__(self, nbase, sz, residual_on=True, concatenation=False,
                 strides=DEFAULT_STRIDES_3D):
        super().__init__()
        block = ResUp if residual_on else ConvUp
        self.strides = tuple(tuple(s) for s in strides)
        self.up = nn.ModuleList([
            block(nbase[n], nbase[n - 1], nbase[-1], sz, concatenation)
            for n in range(1, len(nbase))
        ])

    def forward(self, style, feats):
        x = self.up[-1](feats[-1], feats[-1], style)
        for n in range(len(self.up) - 2, -1, -1):
            # Interpolate to the skip's own size rather than scaling by the
            # stride: an odd spatial size does not survive a round trip through
            # pooling, and the skip add would then be off by a voxel.
            x = F.interpolate(x, size=feats[n].shape[-3:], mode="nearest")
            x = self.up[n](x, feats[n], style)
        return x


class CPnet3D(nn.Module):
    """Cellpose's CPnet in 3D.

    Args:
        nbase: channel widths, input channels first, e.g. ``(1, 32, 64, 128, 256)``.
        nout: output channels. 4 here -- 3 flow components and cell probability.
        sz: convolution kernel size.
        residual_on: use the residual blocks. Off gives the plain U-Net Cellpose
            ablates against.
        style_on: inject the style vector. Off zeroes it, which is Cellpose's
            own ablation and worth running before trusting it.
        concatenation: concatenate skips instead of adding them.
        strides: per-stage pooling, one tuple per transition.
        heads: name -> channel count, to split the output like the rest of the
            model zoo.
    """

    def __init__(self, nbase=(1,) + CELLPOSE_NBASE, nout=4, sz=CELLPOSE_SZ,
                 residual_on=True, style_on=True, concatenation=False,
                 strides=DEFAULT_STRIDES_3D, heads=None):
        super().__init__()
        nbase = tuple(int(n) for n in nbase)
        strides = tuple(tuple(int(v) for v in s) for s in strides)
        if len(strides) != len(nbase) - 2:
            raise ValueError(
                f"need one stride per transition: {len(nbase) - 2} for nbase={nbase}, "
                f"got {len(strides)}"
            )

        self.nbase = nbase
        self.nout = nout
        self.heads = heads
        self.style_on = style_on
        self.name = "cpnet3d"
        self.n_channels = nbase[0]
        self.num_classes = nout

        self.downsample = Downsample(nbase, sz, residual_on, strides)
        # Cellpose's decoder widths: the encoder's, with the deepest repeated so
        # the bottleneck block maps 256 -> 256 before the first upsample.
        nbaseup = list(nbase[1:]) + [nbase[-1]]
        self.upsample = Upsample(nbaseup, sz, residual_on, concatenation, strides)
        self.make_style = MakeStyle()
        self.output = batchconv(nbaseup[0], nout, 1)

        # What the input has to be padded to for every pooling to be exact.
        self.divisible_by = tuple(
            int(np.prod([s[a] for s in strides])) for a in range(3)
        )

    def count_parameters(self):
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        from model_zoo.models_seg import split_heads

        feats = self.downsample(x)
        style = self.make_style(feats[-1])
        out = self.output(self.upsample(style if self.style_on else style * 0, feats))
        return split_heads(out, self.heads)


# Channel widths. "cellpose" is the paper's own; the others scale it, since a
# 3-D volume asks more of the bottleneck than a 2-D image does.
CPNET_PRESETS = {
    "cellpose": (32, 64, 128, 256),
    "cellpose_l": (64, 128, 256, 512),
    "cellpose_xl": (64, 128, 256, 512, 512),
}


def define_cpnet(preset="cellpose", nout=4, n_channels=1, heads=None,
                 patch_size=None, spacing=None, verbose=0, **kwargs):
    """Build a :class:`CPnet3D`, planning its pooling from the patch size."""
    if preset not in CPNET_PRESETS:
        raise ValueError(f"unknown CPnet preset {preset!r}, expected {list(CPNET_PRESETS)}")

    widths = CPNET_PRESETS[preset]
    nbase = (n_channels,) + widths
    n_stages = len(widths) - 1
    strides = kwargs.pop("strides", None)
    if strides is None:
        strides = (
            plan_strides(patch_size, spacing or (1.625, 0.40625, 0.40625), n_stages)
            if patch_size is not None
            else DEFAULT_STRIDES_3D[:n_stages]
        )

    model = CPnet3D(nbase=nbase, nout=nout, strides=strides, heads=heads, **kwargs)
    model.preset = preset
    model.patch_size = tuple(patch_size) if patch_size is not None else None
    if verbose:
        print(f"\n-> Cellpose CPnet3D {preset} on patch {model.patch_size}  "
              f"({model.count_parameters() / 1e6:.1f}M parameters)")
        print(f"   nbase={nbase}  strides={strides}  "
              f"divisible_by={model.divisible_by}\n")
    return model
