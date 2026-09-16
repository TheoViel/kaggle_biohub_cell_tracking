"""Temporal U-Net v2: a per-frame encoder and a motion-tolerant temporal fusion.

Input shape  : ``(B, T, C_in, Z, Y, X)``
Output shape : ``(B, T, C_out, Z // sz, Y // sy, X // sx)`` -- the same contract
as ``TemporalUNet3D``, including the ``(coarse, fine)`` pair when
``n_extra_decode_steps > 0``, so it drops into ``UNetNodeTransformer``
unchanged.

What differs from v1, and why
-----------------------------

**1. The encoder never mixes frames.** v1 interleaves ``_TemporalAttention``
between the encoder stages, which makes a frame's features a function of the
whole window it was encoded in. Two consequences, and the second is the
expensive one:

* the same frame encoded in window ``[t-1, t]`` and in ``[t, t+1]`` gives two
  different feature maps, so an overlapped sliding window has to recompute the
  encoder for every frame of every window. At an overlap of one frame that is
  exactly twice the encoder work;
* there is no single "features of frame t" to cache.

Here the encoder is a plain per-frame 3D U-Net encoder. ``encode_frames`` takes
``(N, C, Z, Y, X)`` -- any flat batch of frames, in any grouping -- and returns
features that depend on that frame alone. An overlapped inference loop encodes
each frame once, keeps the skips in a cache, and calls ``decode_window`` per
window. See ``FrameFeatureCache``.

**2. The temporal interaction lives in the decoder, and is local.** Its job is
narrow: decide whether a candidate detection is corroborated by the frames
around it. Cells do not appear for one frame and vanish, so a peak with no
counterpart in the neighbouring frames is evidence of a false positive, and a
peak that is corroborated should be reinforced. That is a local question -- the
corroborating evidence sits near the same place -- so it does not need global
spatial attention, and global attention at decoder resolution would be
unaffordable anyway.

**3. "Near the same place" is not "at the same voxel".** This is what v1's
per-voxel temporal attention gets wrong: it mixes voxel ``p`` at ``t`` with
voxel ``p`` at ``t+1``, which is only the same cell if the cell did not move.
Measured over 17,882 consecutive-frame ground-truth links in 24 stacks:

    displacement (um)      median 1.68    p90 3.63    p99 7.56    max 21.05
    |delta| in voxels      z: p90 1, p99 4    y/x: p90 4, p99 12

At the ``(1, 4, 4)`` node grid that is a p99 of ``(4, 3, 3)`` feature voxels,
and at the ``(1, 2, 2)`` detection grid ``(4, 6, 6)``. So a same-voxel
comparison is wrong for roughly a third of all cells at the median and badly
wrong in the tail, and the axis that hurts most is ``z`` -- the anisotropic
schedule never pools it, so its displacement in feature voxels does not shrink
as the decoder goes coarser the way ``y``/``x`` does.

``_MotionTolerantFusion`` therefore matches before it compares: a small
dilated correlation window finds where the cell went, a soft-argmax over that
window gives a sub-voxel displacement, and the neighbour frame is warped by it
before the features are combined. The default window spans the measured p99 on
each axis with 45 taps rather than the 693 a dense window of the same reach
would need.

The fusion emits an explicit scalar per voxel -- the peak matching score --
and gates on it, so the block can *attenuate* a feature that nothing
corroborates instead of only adding to one. An additive-only block cannot
suppress a false positive; it can only fail to boost it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

from model_zoo.temporal_unet import TemporalUNet3D, _norm, _to_3tuple


# The default correlation window, as (taps, dilation) per axis. The product of
# taps is the cost of the block, so the reach is bought with dilation and the
# soft-argmax below interpolates between the taps rather than the window
# enumerating every voxel:
#
#   z: 5 taps, dilation 2 -> {-4, -2, 0, +2, +4}   spans the p99 of 4
#   y: 3 taps, dilation 3 -> {-3, 0, +3}           spans the p99 of 3
#   x: 3 taps, dilation 3 -> {-3, 0, +3}           spans the p99 of 3
#
# 45 taps against the 9*7*7 = 441 a dense window of the same reach would cost.
# These are sized for the (1, 4, 4) node grid; at the (1, 2, 2) detection grid
# the same p99 is (4, 6, 6) voxels, so a fusion placed there wants the y/x
# dilation at 6 -- see ``dilation_for_stride``.
DEFAULT_TAPS = (5, 3, 3)
DEFAULT_DILATION = (2, 3, 3)

# p99 of |delta| per axis in *original* voxels, from the measurement quoted in
# the module docstring. ``dilation_for_stride`` divides this by the feature
# stride to size a window at whatever grid a fusion block is placed on.
P99_DISPLACEMENT_VOXELS = (4.0, 12.0, 12.0)


def dilation_for_stride(
    output_stride: Sequence[int], taps: Sequence[int] = DEFAULT_TAPS
) -> tuple[int, int, int]:
    """Per-axis dilation whose window reaches the p99 displacement at a stride.

    A fusion block placed deeper in the decoder sees smaller displacements in
    feature voxels and wants a tighter window; one placed at the finest grid
    sees larger ones. Rather than leave that to be rediscovered per
    configuration, derive it: reach = dilation * (taps - 1) / 2, so

        dilation = ceil( p99_voxels / stride / ((taps - 1) / 2) )

    Never returns 0, which would collapse the window onto a single tap.
    """
    stride = _to_3tuple(output_stride)
    taps = _to_3tuple(taps)
    out = []
    for p99, s, t in zip(P99_DISPLACEMENT_VOXELS, stride, taps):
        reach = max(1, (t - 1) // 2)
        out.append(max(1, int(-(-(p99 / max(1, s)) // reach))))
    return tuple(out)  # type: ignore[return-value]


def _offset_grid(taps: Sequence[int], dilation: Sequence[int]) -> torch.Tensor:
    """The (D, 3) integer displacements of a dilated window, centred on zero."""
    tz, ty, tx = _to_3tuple(taps)
    dz, dy, dx = _to_3tuple(dilation)
    axes = [
        torch.arange(n, dtype=torch.float32).sub((n - 1) / 2.0).mul(d)
        for n, d in ((tz, dz), (ty, dy), (tx, dx))
    ]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], dim=1)


def _shift(t: torch.Tensor, d: Sequence[int]) -> torch.Tensor:
    """``t`` translated by integer ``(dz, dy, dx)``, edge-replicated.

    Replication rather than zeros, for the same reason ``_pad_to_multiple``
    uses it in v1: a zero feature is a meaningful value that the correlation
    would happily match against. Repeating the border says nothing new instead.
    """
    dz, dy, dx = (int(v) for v in d)
    if (dz, dy, dx) == (0, 0, 0):
        return t
    Z, Y, X = t.shape[-3:]
    # out[r] = t[r + d]: for d > 0 that means dropping the first d entries, so
    # the padding goes on the *right* and the slice starts at d. Getting this
    # backwards silently negates every displacement the correlation recovers --
    # the cost volume still peaks at the right magnitude, so it looks like it
    # works, and the warp then moves each feature the wrong way.
    pad = (max(-dx, 0), max(dx, 0), max(-dy, 0), max(dy, 0), max(-dz, 0), max(dz, 0))
    t = F.pad(t, pad, mode="replicate")
    sz, sy, sx = max(dz, 0), max(dy, 0), max(dx, 0)
    return t[..., sz : sz + Z, sy : sy + Y, sx : sx + X]


class _MotionTolerantFusion(nn.Module):
    """Corroborate each voxel against the other frames of the window.

    For every frame ``t`` and every other frame ``t'`` in the window:

    1. **Match.** Correlate a low-dimensional embedding of ``t`` against the
       same embedding of ``t'`` over a dilated window of candidate
       displacements. The embeddings are L2-normalised, so each correlation is
       a cosine in ``[-1, 1]`` and the scores are comparable between voxels --
       which matters, because the gate below reads their magnitude and not just
       their argmax.
    2. **Locate.** Soft-argmax over the window gives a displacement that is not
       restricted to the tap positions, so a dilation of 3 does not quantise
       motion to multiples of 3.
    3. **Warp.** Sample ``t'``'s value features at that displacement.
    4. **Agree.** Keep the peak cosine as an explicit per-voxel agreement
       score.

    The aggregate over ``t'`` of the warped values is the context, and the
    aggregate of the agreement scores drives a gate. ``agree="min"`` asks a
    voxel to be corroborated by *every* other frame, which is the strict
    reading of "cells do not disappear"; ``"mean"`` (default) tolerates one bad
    neighbour, which matters at a window edge and when a cell is genuinely
    occluded for a frame.

    The block is residual and its gate is initialised closed
    (``gate_scale = 0``), so at initialisation it is the identity and training
    starts from the per-frame model rather than from noise injected into it.
    """

    def __init__(
        self,
        channels: int,
        match_channels: int = 32,
        value_channels: int | None = None,
        taps: Sequence[int] = DEFAULT_TAPS,
        dilation: Sequence[int] = DEFAULT_DILATION,
        temperature: float = 0.07,
        agree: str = "mean",
        norm: str = "batch",
        max_neighbours: int | None = None,
    ) -> None:
        super().__init__()
        if agree not in ("mean", "min"):
            raise ValueError(f"unknown agree {agree!r}, expected 'mean' / 'min'")
        value_channels = value_channels or max(8, channels // 2)

        self.query = nn.Conv3d(channels, match_channels, kernel_size=1, bias=False)
        self.key = nn.Conv3d(channels, match_channels, kernel_size=1, bias=False)
        self.value = nn.Conv3d(channels, value_channels, kernel_size=1, bias=False)

        # +1 for the agreement score, which is concatenated as a feature as
        # well as used as a gate: the gate can only scale the whole update, and
        # how a voxel should react to weak corroboration may depend on what it
        # is looking at.
        self.fuse = nn.Sequential(
            nn.Conv3d(channels + value_channels + 1, channels, kernel_size=1, bias=False),
            _norm(norm, channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

        # The gate is an affine function of the agreement score through a
        # sigmoid. Starting at scale 0 / bias 0 makes it a constant 0.5, and
        # `gate_scale` at 0 also zeroes the residual below, so the block starts
        # as the identity either way.
        self.gate_logit_scale = nn.Parameter(torch.tensor(4.0))
        self.gate_logit_bias = nn.Parameter(torch.tensor(0.0))
        self.gate_scale = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

        self.register_buffer("offsets", _offset_grid(taps, dilation), persistent=False)
        self.temperature = float(temperature)
        self.agree = agree
        self.max_neighbours = max_neighbours

    # -- helpers ----------------------------------------------------------

    def _neighbours(self, t: int, T: int) -> list[int]:
        """Which frames corroborate frame ``t``, nearest first."""
        order = sorted((s for s in range(T) if s != t), key=lambda s: (abs(s - t), s))
        if self.max_neighbours is not None:
            order = order[: self.max_neighbours]
        return order

    def _match(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(warped value, peak cosine) for one ordered frame pair.

        ``q`` is the frame being corroborated; ``k``/``v`` belong to the
        corroborating frame.
        """
        D = self.offsets.shape[0]
        N = q.shape[0]
        spatial = q.shape[-3:]

        cost = q.new_empty((N, D, *spatial))
        for i in range(D):
            # Sum over the (already normalised) matching channels -> cosine.
            cost[:, i] = (q * _shift(k, self.offsets[i].tolist())).sum(dim=1)

        weights = torch.softmax(cost / self.temperature, dim=1)
        # (N, 3, Z, Y, X): the expected displacement under that distribution.
        flow = torch.einsum("ndzyx,dc->nczyx", weights, self.offsets.to(q.dtype))
        # The peak is the honest "is this corroborated at all" number: the
        # softmax weights sum to 1 whether or not anything matched, so they
        # cannot say "nothing here", while max cosine can.
        agree = cost.amax(dim=1, keepdim=True)
        return _warp(v, flow), agree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, Z, Y, X)
        B, T, C = x.shape[:3]
        if T < 2:
            return x
        spatial = x.shape[3:]
        flat = x.reshape(B * T, C, *spatial)

        q = F.normalize(self.query(flat), dim=1).reshape(B, T, -1, *spatial)
        k = F.normalize(self.key(flat), dim=1).reshape(B, T, -1, *spatial)
        v = self.value(flat).reshape(B, T, -1, *spatial)

        out = []
        for t in range(T):
            ctxs, agrees = [], []
            for s in self._neighbours(t, T):
                warped, agree = self._match(q[:, t], k[:, s], v[:, s])
                ctxs.append(warped)
                agrees.append(agree)
            ctx = torch.stack(ctxs).mean(dim=0)
            stacked = torch.stack(agrees)
            agree = stacked.mean(dim=0) if self.agree == "mean" else stacked.amin(dim=0)

            update = self.fuse(torch.cat([x[:, t], ctx, agree], dim=1))
            gate = torch.sigmoid(self.gate_logit_scale * agree + self.gate_logit_bias)
            out.append(x[:, t] + self.gate_scale * gate * update)
        return torch.stack(out, dim=1)


def _warp(v: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """``v`` sampled at ``p + flow``, trilinear, edge-clamped.

    ``flow`` is in voxels, ``(N, 3, Z, Y, X)`` ordered ``(dz, dy, dx)``;
    ``grid_sample`` wants normalised ``(x, y, z)``, which is the axis flip
    below and the usual place this goes wrong.
    """
    N, _, Z, Y, X = flow.shape
    dev, dtype = flow.device, flow.dtype
    zs = torch.arange(Z, device=dev, dtype=dtype)
    ys = torch.arange(Y, device=dev, dtype=dtype)
    xs = torch.arange(X, device=dev, dtype=dtype)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")

    pz = gz.unsqueeze(0) + flow[:, 0]
    py = gy.unsqueeze(0) + flow[:, 1]
    px = gx.unsqueeze(0) + flow[:, 2]

    def norm(p, n):
        return (2.0 * p / max(n - 1, 1)) - 1.0 if n > 1 else torch.zeros_like(p)

    grid = torch.stack([norm(px, X), norm(py, Y), norm(pz, Z)], dim=-1)
    return F.grid_sample(
        v, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


class FrameFeatureCache:
    """Per-frame encoder outputs, keyed by whatever identifies a frame.

    The point of the per-frame encoder is that this is *sound*: features of a
    frame do not depend on the window it was encoded in, so a sliding window
    with overlap can encode each frame once. Hold one of these across the
    windows of a video and the encoder cost is ``n_frames`` forwards rather
    than ``n_frames * window_size``.

        cache = FrameFeatureCache()
        for window in windows:                       # e.g. [(0,1), (1,2), ...]
            missing = [t for t in window if t not in cache]
            if missing:
                cache.update(missing, model.encode_frames(images[missing]))
            out = model.decode_window(cache.gather(window))

    It is a plain dict with a tensor-aware ``gather``; nothing here is clever,
    and it is separate from the model so a caller can bound it (a video is far
    too large to keep every frame's skips resident).
    """

    def __init__(self) -> None:
        self._store: dict[object, tuple[list[torch.Tensor], torch.Tensor]] = {}

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def update(self, keys: Sequence[object], feats: tuple[list[torch.Tensor], torch.Tensor]) -> None:
        skips, bottleneck = feats
        for i, key in enumerate(keys):
            self._store[key] = ([s[i] for s in skips], bottleneck[i])

    def evict(self, keys: Sequence[object]) -> None:
        for key in keys:
            self._store.pop(key, None)

    def gather(self, keys: Sequence[object]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """The cached frames stacked into one window, shaped ``(1, T, C, ...)``.

        The leading 1 is the batch axis ``decode_window`` expects: a cache
        holds single frames, and a window assembled from it is one sample.
        """
        entries = [self._store[k] for k in keys]
        n_skips = len(entries[0][0])
        skips = [
            torch.stack([e[0][j] for e in entries]).unsqueeze(0) for j in range(n_skips)
        ]
        return skips, torch.stack([e[1] for e in entries]).unsqueeze(0)


class TemporalUNet3Dv2(TemporalUNet3D):
    """A ``TemporalUNet3D`` whose encoder is per-frame and whose decoder fuses time.

    Every geometry parameter -- ``layers``, ``pool_stride``, ``n_anisotropic``,
    ``stem``, ``n_decode_steps``, ``n_extra_decode_steps``, ``hires_skip_fusion``
    -- means what it means in v1 and is inherited unchanged, as are
    ``output_stride`` / ``fine_output_stride`` / ``divisible_by``. What v2
    replaces is where the frames meet.

    Parameters
    ----------
    temporal_fusion_at : sequence of {"bottleneck", "decoder", "head", "fine"}
        Where a fusion block is inserted.

        * ``"bottleneck"`` -- on the encoder output, before the decoder. The
          cheapest place by far (the smallest tensor) and the one where a
          fixed-voxel window reaches furthest in real distance, but its
          corroboration is blurred by the stride before the head ever sees it.
        * ``"decoder"`` -- after every decoder stage.
        * ``"head"`` (default) -- on the features ``self.head`` reads. This is
          the one that answers the question as posed: it is the last thing
          between the network and the detection map, so a suppressed voxel is
          suppressed in the output.
        * ``"fine"`` -- likewise for the extra fine branch, if there is one.

    temporal_match_channels : int
        Width of the embedding the correlation runs in. It only has to make
        cells distinguishable from each other locally, not carry the
        representation, so this stays small.
    temporal_taps, temporal_dilation : (int, int, int)
        The correlation window. ``temporal_dilation="auto"`` sizes it per
        fusion site from the measured p99 displacement and that site's stride
        -- see ``dilation_for_stride`` -- which is the intended way to use it,
        since the right window at the bottleneck is not the right window at the
        head.
    temporal_agree : {"mean", "min"}
        How the per-neighbour agreement scores combine.
    temporal_max_neighbours : int or None
        Cap on corroborating frames per frame, nearest first. ``None`` uses the
        whole window; 2 keeps it at ``t +- 1`` for a long window.
    """

    def __init__(
        self,
        *args,
        temporal_fusion_at: Sequence[str] = ("head",),
        temporal_match_channels: int = 32,
        temporal_value_channels: int | None = None,
        temporal_taps: Sequence[int] = DEFAULT_TAPS,
        temporal_dilation: Sequence[int] | str = "auto",
        temporal_temperature: float = 0.07,
        temporal_agree: str = "mean",
        temporal_max_neighbours: int | None = None,
        **kwargs,
    ) -> None:
        # The whole point is that the encoder is frame-independent; v1's
        # temporal blocks would silently break the caching contract.
        if kwargs.pop("temporal_attention", False):
            raise ValueError(
                "TemporalUNet3Dv2 has no encoder temporal attention -- that is "
                "what makes its per-frame features cacheable. Use "
                "temporal_fusion_at to place the temporal interaction, or "
                "TemporalUNet3D if you want v1's behaviour."
            )
        kwargs.pop("skip_fullres_temporal", None)
        super().__init__(*args, temporal_attention=False, **kwargs)

        sites = tuple(dict.fromkeys(temporal_fusion_at))
        known = {"bottleneck", "decoder", "head", "fine"}
        unknown = set(sites) - known
        if unknown:
            raise ValueError(
                f"unknown temporal_fusion_at {sorted(unknown)}, expected {sorted(known)}"
            )
        if "fine" in sites and self.fine_head is None:
            raise ValueError(
                "temporal_fusion_at includes 'fine' but this model has no fine "
                "branch; set n_extra_decode_steps > 0 or drop it"
            )
        self.temporal_fusion_at = sites

        # Everything a subclass needs to build a different fusion block at the
        # same sites, kept as one dict so `_build_fusion` can be overridden
        # without re-listing the constructor.
        self._fusion_kwargs = dict(
            match_channels=temporal_match_channels,
            value_channels=temporal_value_channels,
            taps=tuple(temporal_taps),
            temperature=temporal_temperature,
            agree=temporal_agree,
            norm=kwargs.get("norm", "batch"),
            max_neighbours=temporal_max_neighbours,
        )
        self._temporal_dilation = temporal_dilation

        def build(channels: int, stride: Sequence[int]) -> nn.Module:
            return self._build_fusion(channels, stride)

        # Channel widths and strides per site, read off the geometry the parent
        # already computed rather than re-derived.
        widths = self._stage_widths()

        self.fuse_bottleneck = (
            build(widths[-1], self._stride_at(len(widths) - 1))
            if "bottleneck" in sites
            else None
        )
        self.fuse_decoder = nn.ModuleList()
        if "decoder" in sites:
            for i, blk in enumerate(self.decoder_blocks):
                stage = len(widths) - 1 - (i + 1)
                self.fuse_decoder.append(build(widths[stage], self._stride_at(stage)))
        self.fuse_head = (
            build(widths[self.head_stage], self.output_stride) if "head" in sites else None
        )
        self.fuse_fine = (
            build(widths[self.head_stage - self.n_extra_decode_steps], self.fine_output_stride)
            if "fine" in sites
            else None
        )

    def _build_fusion(self, channels: int, stride: Sequence[int]) -> nn.Module:
        """The fusion block placed at one site. Overridden by v3.

        ``stride`` is that site's cumulative input-to-feature stride, which is
        what sizes the correlation window under ``temporal_dilation="auto"``
        and, in v3, what converts a displacement in voxels to one in microns.
        """
        kw = dict(self._fusion_kwargs)
        dil = (
            dilation_for_stride(stride, kw["taps"])
            if self._temporal_dilation == "auto"
            else self._temporal_dilation
        )
        return _MotionTolerantFusion(channels, dilation=dil, **kw)

    # -- geometry helpers -------------------------------------------------

    def _stage_widths(self) -> list[int]:
        """Output width of every encoder stage."""
        widths = []
        for block in self.encoder_blocks:
            conv = [m for m in block.modules() if isinstance(m, nn.Conv3d)][-1]
            widths.append(conv.out_channels)
        return widths

    def _stride_at(self, stage: int) -> tuple[int, int, int]:
        """Cumulative input-to-feature stride at an encoder stage.

        Derived from ``output_stride`` rather than recomputed from the stem:
        the parent already resolved the stem's step, and ``pools[0]`` is the
        identity precisely because the stem took it. So walking the pool strides
        between the head stage and ``stage`` -- in either direction -- scales
        the one absolute stride that is known to be right.
        """
        strides = [
            tuple(int(v) for v in p.stride) if isinstance(p, nn.MaxPool3d) else (1, 1, 1)
            for p in self.pools
        ]
        lo, hi = sorted((self.head_stage, stage))
        rel = [1, 1, 1]
        for s in strides[lo + 1 : hi + 1]:
            for a in range(3):
                rel[a] *= s[a]
        if stage >= self.head_stage:
            return tuple(int(o) * r for o, r in zip(self.output_stride, rel))
        # Finer than the head: the same factors divide instead. They always
        # divide exactly, being the very products that built `output_stride`.
        return tuple(int(o) // r for o, r in zip(self.output_stride, rel))

    # -- the two halves ---------------------------------------------------

    def encode_frames(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Per-frame encoder. ``(N, C_in, Z, Y, X)`` -> ``(skips, bottleneck)``.

        ``N`` is a flat batch of frames with no temporal meaning: nothing here
        reads across it, which is exactly the property that makes the result
        cacheable and reusable between overlapping windows.
        """
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for i, (pool, block, spatial) in enumerate(
            zip(self.pools, self.encoder_blocks, self.spatial_blocks)
        ):
            x = pool(x)
            x = self._run(block, x)
            if not isinstance(spatial, nn.Identity):
                x = self._run(spatial, x)
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)
        return skips, x

    def _fuse(self, fusion: nn.Module | None, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Run a fusion block on a ``(B*T, ...)`` tensor, keeping it folded."""
        if fusion is None:
            return x
        shaped = x.reshape(B, T, *x.shape[1:])
        if self.gradient_checkpointing and self.training:
            shaped = _grad_ckpt(fusion, shaped, use_reentrant=False)
        else:
            shaped = fusion(shaped)
        return shaped.reshape(B * T, *x.shape[1:])

    def decode_window(
        self, feats: tuple[list[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decoder + temporal fusion over one window of encoded frames.

        ``feats`` is ``(skips, bottleneck)`` shaped ``(B, T, C, ...)`` -- what
        ``FrameFeatureCache.gather`` returns, or ``encode_frames`` reshaped.
        """
        skips, x = feats
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        skips = [s.reshape(B * T, *s.shape[2:]) for s in skips]

        x = self._fuse(self.fuse_bottleneck, x, B, T)

        for i, (up, block, skip) in enumerate(
            zip(self.upsamples, self.decoder_blocks, skips[::-1])
        ):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = self._run(block, torch.cat([x, skip], dim=1))
            if self.fuse_decoder:
                x = self._fuse(self.fuse_decoder[i], x, B, T)

        x = self._fuse(self.fuse_head, x, B, T)

        fine = None
        if self.fine_head is not None:
            fine = x
            for j, (up, block) in enumerate(zip(self.fine_upsamples, self.fine_decoder_blocks)):
                skip = skips[self.head_stage - 1 - j]
                fine = up(fine)
                if fine.shape[2:] != skip.shape[2:]:
                    fine = F.interpolate(
                        fine, size=skip.shape[2:], mode="trilinear", align_corners=False
                    )
                fine = self._run(block, torch.cat([fine, skip], dim=1))
            fine = self._fuse(self.fuse_fine, fine, B, T)
            fine = self.fine_head(fine)

        if self.hires_projs:
            detail = []
            for proj, pool, skip in zip(self.hires_projs, self.hires_pools, skips):
                d = proj(pool(skip))
                if d.shape[2:] != x.shape[2:]:
                    d = F.interpolate(d, size=x.shape[2:], mode="trilinear", align_corners=False)
                detail.append(d)
            x = torch.cat([x, *detail], dim=1)

        x = self.head(x).reshape(B, T, -1, *x.shape[2:])
        if fine is None:
            return x
        return x, fine.reshape(B, T, *fine.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C_in, Z, Y, X) -- the v1 contract, encode then decode.
        B, T = x.shape[:2]
        skips, bottleneck = self.encode_frames(x.reshape(B * T, *x.shape[2:]))
        return self.decode_window(
            (
                [s.reshape(B, T, *s.shape[1:]) for s in skips],
                bottleneck.reshape(B, T, *bottleneck.shape[1:]),
            )
        )
