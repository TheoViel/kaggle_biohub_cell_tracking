"""Temporal U-Net v3: the neighbour proposes, so a missed cell can be recovered.

Same contract as v1 and v2 -- ``(B, T, C_in, Z, Y, X)`` in, one or two maps at
``output_stride`` / ``fine_output_stride`` out -- and the same per-frame
encoder as v2, so ``encode_frames`` / ``decode_window`` / ``FrameFeatureCache``
work unchanged. What differs is the fusion block.

Why v2's block cannot recover a false negative
----------------------------------------------

v2's ``_MotionTolerantFusion`` is a *gather*: voxel ``p`` of frame ``t`` forms a
query, correlates it against the neighbouring frame, and is reinforced when it
finds a match. That is the right shape for suppressing a false positive -- a
peak nothing corroborates gets its gate closed -- and it is structurally unable
to do the opposite. Where the detector missed a cell, ``q_t(p)`` is not a weak
description of a cell, it is a description of background: there is nothing to
match with, the correlation finds nothing, and the gate closes exactly where it
should have opened. An architecture that asks "does my evidence check out?"
cannot manufacture evidence it does not have.

v3 reverses the direction. The *neighbouring* frame, where the cell was
detected, proposes: it estimates where each of its confident voxels went, and
deposits its feature there. The blind frame contributes nothing to that
decision, which is the point.

What this can and cannot reach
------------------------------

Measured on 199 out-of-fold stacks of ``logs/2026-09-13/1/preds_r6_0.40``, for
every ground-truth node the detector missed, whether the same track was
detected at ``t-1`` or ``t+1``:

    embryo    GT nodes   missed   recall   recoverable from t+-1
    44b6        20 197      219   0.9892    183   (83.6% of misses)
    6bba       113 121     7271   0.9357   3067   (42.2% of misses)
    ALL        133 318     7490   0.9438   3250   (43.4% of misses)

So the ceiling on *any* temporal recovery mechanism is node recall 0.9438 ->
0.9682. The other 56.6% of misses are missed in every neighbouring frame too
and nothing here invents them. Worth keeping in view before reading a result:
more than half the FN budget is out of reach by construction, and 97% of the
misses are in 6bba, the embryo with the weaker recall to begin with.

The three changes
-----------------

**1. The neighbour's objectness enters the match.** The displacement search
scores a correspondence as

    alpha * <q_t'(r), k_t(r + d)>  +  beta * o_t'(r)  -  gamma * ||d (um)||^2

``o_t'`` is a learned "there is a cell here" scalar read off the *neighbour*,
and the last term is a motion prior in microns -- the measured median
frame-to-frame displacement is 1.68 um, so a correspondence that requires a
large jump should have to earn it. When the appearance term is uninformative,
because the target frame has nothing there to match against, the search falls
back on the prior and lands the proposal near where the cell was. That is the
correct behaviour for a miss, and it is what v2 gets wrong.

**2. A splat, not a warp.** Each source voxel deposits into the target grid at
its own predicted position, trilinearly, with the weights accumulated and
divided out. Gathering cannot localise a cell the target frame has no feature
for; splatting does not need it to. This is the expensive change: it is a
scatter, so it is slower than ``grid_sample`` and non-deterministic on CUDA
(``index_add_`` accumulates in nondeterministic order). Set
``torch.use_deterministic_algorithms`` off, or accept the run-to-run jitter.

**3. Two paths, and only one of them is gated by the target frame.**

    out = x * (1 + m * 2 * (agree_gate - 0.5))      suppress / reinforce
            + a * support_gate * recover_term       inject what is missing

``agree_gate`` is v2's appearance corroboration and keeps its FP-suppression
job. ``support_gate`` is driven by the splatted objectness alone -- nothing in
it reads the target frame's own confidence, which is precisely what lets it
fire where that frame is blind. In v2 a single gate multiplied the whole
update, and that one multiplication is what made the block FP-only.

Both ``m`` and ``a`` start at zero, so the block is the identity at
initialisation and training starts from the per-frame model. One consequence
looks alarming and is not: at step 0 every parameter *inside* the two branches
gets exactly zero gradient, because its contribution is multiplied by a zero
scale. The scales themselves do get gradient, so they move on the first step
and the branches start learning on the second -- this is ReZero, and the branch
is inert for exactly one optimiser step, not forever.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_zoo.temporal_unet import _norm, _to_3tuple
from model_zoo.temporal_unet_v2 import (
    TemporalUNet3Dv2,
    _offset_grid,
    _shift,
    _warp,
    dilation_for_stride,
)

# Voxel size of the raw data in microns, (z, y, x) -- ``params.SCALE``. Restated
# here rather than imported so ``model_zoo`` stays free of the pipeline's
# configuration; ``voxel_um`` overrides it for data with another spacing.
SCALE_UM = (1.625, 0.40625, 0.40625)

# Median frame-to-frame displacement, microns, from the measurement in the
# module docstring. The motion prior is scaled so that a proposal at the median
# distance is penalised by a fixed, small amount and the tail is what pays.
MEDIAN_DISPLACEMENT_UM = 1.68


def _splat(
    src: torch.Tensor, pos: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    """Deposit ``src`` at fractional positions ``pos``, trilinearly, normalised.

    Parameters
    ----------
    src : (N, C, Z, Y, X)
        What each source voxel carries.
    pos : (N, 3, Z, Y, X)
        Where that voxel lands in the target grid, in voxels, ``(z, y, x)``.

    Returns
    -------
    (N, C, Z, Y, X)
        The accumulated contributions divided by the accumulated weight, so a
        target voxel that two source voxels landed on gets their mean rather
        than their sum. Where nothing landed the weight is 0 and the output is
        0 -- which is the honest value: no neighbour proposed anything here.

    A weight channel of ones rides along with ``src`` so the normaliser costs
    one extra channel rather than a second scatter.
    """
    N, C, Z, Y, X = src.shape
    S = Z * Y * X
    dev = src.device

    payload = torch.cat([src, src.new_ones(N, 1, Z, Y, X)], dim=1)  # (N, C+1, ...)
    payload = payload.reshape(N, C + 1, S)

    # Clamp rather than drop: a proposal that leaves the crop is pinned to the
    # face, which is the same convention `_warp`'s border padding uses. The
    # alternative -- discarding it -- silently makes the block weaker at exactly
    # the edges where a cell is most likely to be missed.
    p = pos.reshape(N, 3, S)
    pz = p[:, 0].clamp(0, Z - 1)
    py = p[:, 1].clamp(0, Y - 1)
    px = p[:, 2].clamp(0, X - 1)

    z0, y0, x0 = pz.floor(), py.floor(), px.floor()
    fz, fy, fx = pz - z0, py - y0, px - x0
    z0, y0, x0 = z0.long(), y0.long(), x0.long()

    # n * S puts every sample in its own slice of one flat buffer, so the whole
    # batch is a single index_add_ instead of one per sample.
    base = (torch.arange(N, device=dev) * S).view(N, 1)

    out = src.new_zeros(C + 1, N * S)
    flat = payload.permute(1, 0, 2).reshape(C + 1, N * S)
    for dz in (0, 1):
        wz = fz if dz else (1 - fz)
        zz = (z0 + dz).clamp(0, Z - 1)
        for dy in (0, 1):
            wy = fy if dy else (1 - fy)
            yy = (y0 + dy).clamp(0, Y - 1)
            for dx in (0, 1):
                wx = fx if dx else (1 - fx)
                xx = (x0 + dx).clamp(0, X - 1)
                w = (wz * wy * wx).reshape(N * S)
                idx = (base + zz * (Y * X) + yy * X + xx).reshape(N * S)
                out.index_add_(1, idx, flat * w)

    acc = out[:C].reshape(C, N, S).permute(1, 0, 2)
    weight = out[C].reshape(N, 1, S)
    return (acc / weight.clamp_min(eps)).reshape(N, C, Z, Y, X)


class _RecoveryFusion(nn.Module):
    """Let every frame propose detections into its neighbours, and corroborate.

    One block, two jobs, deliberately kept on separate paths so that neither
    can silence the other:

    * **recover** -- source frames splat their features and their objectness
      into the target frame. Gated by the splatted objectness only.
    * **corroborate** -- v2's gather, which scores how well the target frame's
      own feature is matched in its neighbours. Modulates what is already there.

    Parameters
    ----------
    channels : int
        Width of the feature map this sits on.
    voxel_um : (float, float, float)
        Physical size of one voxel at this site, used by the motion prior. The
        anisotropy matters: at the ``(1, 4, 4)`` grid a voxel is ``1.625`` um on
        every axis, but at ``(1, 2, 2)`` it is ``1.625`` in z against ``0.8125``
        in y/x, so a displacement of one voxel does not mean the same thing on
        each axis and a prior in voxels would be wrong.
    motion_prior : float
        Weight of ``-motion_prior * (dist_um / median_um)^2`` in the matching
        score. 0 turns the prior off and leaves the search to appearance alone,
        which is what v2 does.
    objectness_weight : float
        Initial weight of the neighbour's objectness in the matching score. It
        is a parameter, not a constant, so the balance between "looks like the
        same thing" and "looks like a cell" is learned.
    """

    def __init__(
        self,
        channels: int,
        match_channels: int = 32,
        value_channels: int | None = None,
        taps: Sequence[int] = (5, 3, 3),
        dilation: Sequence[int] = (2, 3, 3),
        temperature: float = 0.07,
        agree: str = "mean",
        norm: str = "batch",
        max_neighbours: int | None = None,
        voxel_um: Sequence[float] = SCALE_UM,
        motion_prior: float = 1.0,
        objectness_weight: float = 1.0,
        flow_smooth_iters: int = 3,
        flow_smooth_kernel: Sequence[int] = (3, 7, 7),
    ) -> None:
        super().__init__()
        if agree not in ("mean", "min"):
            raise ValueError(f"unknown agree {agree!r}, expected 'mean' / 'min'")
        value_channels = value_channels or max(8, channels // 2)

        self.query = nn.Conv3d(channels, match_channels, kernel_size=1, bias=False)
        self.key = nn.Conv3d(channels, match_channels, kernel_size=1, bias=False)
        self.value = nn.Conv3d(channels, value_channels, kernel_size=1, bias=False)
        # Objectness. 3x3x3 rather than 1x1x1: "is there a cell here" is a
        # question about a neighbourhood, and this is the one signal the
        # recovery path depends on entirely.
        self.object = nn.Sequential(
            nn.Conv3d(channels, max(8, channels // 4), kernel_size=3, padding=1, bias=False),
            _norm(norm, max(8, channels // 4)),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(8, channels // 4), 1, kernel_size=1),
        )

        # The recovery path sees the splatted feature and the splatted
        # objectness, and the target frame's own feature for context -- but the
        # *gate* below does not, which is what lets it fire on a blind voxel.
        self.recover = nn.Sequential(
            nn.Conv3d(channels + value_channels + 1, channels, kernel_size=1, bias=False),
            _norm(norm, channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
        )
        self.corroborate = nn.Sequential(
            nn.Conv3d(channels + value_channels + 1, channels, kernel_size=1, bias=False),
            _norm(norm, channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

        self.support_logit_scale = nn.Parameter(torch.tensor(4.0))
        self.support_logit_bias = nn.Parameter(torch.tensor(0.0))
        self.agree_logit_scale = nn.Parameter(torch.tensor(4.0))
        self.agree_logit_bias = nn.Parameter(torch.tensor(0.0))
        # Both start at 0: identity at initialisation, so training begins from
        # the per-frame model rather than from noise added to it.
        self.recover_scale = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.modulate_scale = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

        self.objectness_weight = nn.Parameter(torch.tensor(float(objectness_weight)))
        self.motion_prior = nn.Parameter(torch.tensor(float(motion_prior)))

        offsets = _offset_grid(taps, dilation)
        self.register_buffer("offsets", offsets, persistent=False)
        # Squared physical length of each candidate displacement, in units of
        # the median step, precomputed once.
        um = torch.tensor(_to_3tuple(voxel_um), dtype=torch.float32)
        d_um = (offsets * um).pow(2).sum(dim=1).sqrt() / MEDIAN_DISPLACEMENT_UM
        self.register_buffer("offset_cost", d_um.pow(2), persistent=False)

        self.temperature = float(temperature)
        self.agree = agree
        self.max_neighbours = max_neighbours
        self.flow_smooth_iters = int(flow_smooth_iters)
        self.flow_smooth_kernel = _to_3tuple(flow_smooth_kernel)

    # -- pieces -----------------------------------------------------------

    def _neighbours(self, t: int, T: int) -> list[int]:
        order = sorted((s for s in range(T) if s != t), key=lambda s: (abs(s - t), s))
        if self.max_neighbours is not None:
            order = order[: self.max_neighbours]
        return order

    def _propagate_flow(self, flow: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        """Spread the displacement field out of confident voxels into blind ones.

        A confidence-weighted box blur, iterated: ``sum(w*flow) / sum(w)`` over
        a growing neighbourhood. A voxel that matched well keeps its own answer
        (it dominates its own average); a voxel that matched nothing inherits
        the average of whatever matched near it. ``flow_smooth_iters=0`` turns
        this off, which is the ablation -- it is the difference between "where
        did *I* go" and "where did things *around here* go", and only the
        second is answerable for a cell the destination frame cannot see.
        """
        if self.flow_smooth_iters <= 0:
            return flow
        k = self.flow_smooth_kernel
        pad = tuple(v // 2 for v in k)
        w = conf.clamp_min(0)                      # a negative cosine is no evidence
        num, den = flow * w, w
        for _ in range(self.flow_smooth_iters):
            num = F.avg_pool3d(num, k, stride=1, padding=pad, count_include_pad=False)
            den = F.avg_pool3d(den, k, stride=1, padding=pad, count_include_pad=False)
        return num / den.clamp_min(1e-4)

    def _correlate(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """(N, D, Z, Y, X) cosine of ``q`` against ``k`` over the offset window."""
        D = self.offsets.shape[0]
        cost = q.new_empty((q.shape[0], D, *q.shape[-3:]))
        for i in range(D):
            cost[:, i] = (q * _shift(k, self.offsets[i].tolist())).sum(dim=1)
        return cost

    def _propose(
        self,
        q_src: torch.Tensor,
        k_dst: torch.Tensor,
        v_src: torch.Tensor,
        o_src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Source frame proposes into the destination grid. -> (value, support).

        The search is run *from* the source: its query is a real cell's
        description, so it stays informative even when the destination frame
        has nothing at the answer.
        """
        cost = self._correlate(q_src, k_dst)
        # The objectness of the *source* voxel is constant across d -- it says
        # how much this proposal is worth, not where it goes -- so it does not
        # belong in the softmax. It rides along as the payload instead.
        prior = self.offset_cost.view(1, -1, 1, 1, 1).to(cost.dtype)
        # The prior is subtracted *after* the temperature divide, not before.
        # Before, it is scaled by 1/temperature along with the cosine -- and
        # since the cosine lives in [-1, 1] while `offset_cost` reaches ~30 at
        # the window's corner, that puts a +-14 appearance term against a 0..450
        # prior and the search collapses to zero displacement whatever the
        # images say. Here one unit of `motion_prior` costs one logit at the
        # median step, against an appearance range of ~2/temperature.
        logits = cost / self.temperature - self.motion_prior * prior
        weights = torch.softmax(logits, dim=1)
        flow = torch.einsum("ndzyx,dc->nczyx", weights, self.offsets.to(cost.dtype))
        peak = cost.amax(dim=1, keepdim=True)

        # Where a cell is *missing* from the destination frame there is no
        # correspondence to find, and the flow this search returns for it is
        # noise -- which is fatal, because those are precisely the voxels the
        # recovery path exists to serve. So the displacement is taken from the
        # cells that did match: confidence-weighted blurring propagates flow
        # out of the reliable voxels and into the blind ones, and neighbouring
        # cells move coherently enough for that to be the right guess (it is
        # the same assumption `postprocessing/recover.py` makes when it removes
        # the sample's own drift before interpolating a gap).
        #
        # Weighted by objectness *and* match quality, not match quality alone:
        # background correlates with background perfectly well, so a peak cosine
        # on its own would let the empty regions -- whose flow is meaningless --
        # outvote the handful of cells that actually know where they went.
        cellness = torch.sigmoid(o_src)
        flow = self._propagate_flow(flow, cellness * peak.clamp_min(0))

        N, _, Z, Y, X = flow.shape
        dev, dtype = flow.device, flow.dtype
        gz, gy, gx = torch.meshgrid(
            torch.arange(Z, device=dev, dtype=dtype),
            torch.arange(Y, device=dev, dtype=dtype),
            torch.arange(X, device=dev, dtype=dtype),
            indexing="ij",
        )
        pos = torch.stack(
            [gz.unsqueeze(0) + flow[:, 0],
             gy.unsqueeze(0) + flow[:, 1],
             gx.unsqueeze(0) + flow[:, 2]],
            dim=1,
        )

        # Objectness is splatted as a payload channel, weighted by how good the
        # correspondence was: a confident cell whose landing spot is a guess
        # should propose more weakly than one that matched cleanly.
        support = cellness * torch.sigmoid(self.objectness_weight * peak)
        out = _splat(torch.cat([v_src, support], dim=1), pos)
        return out[:, :-1], out[:, -1:]

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
        o = self.object(flat).reshape(B, T, 1, *spatial)

        out = []
        for t in range(T):
            props, supports, ctxs, agrees = [], [], [], []
            for s in self._neighbours(t, T):
                # Recovery: s proposes into t.
                prop, support = self._propose(q[:, s], k[:, t], v[:, s], o[:, s])
                props.append(prop)
                supports.append(support)

                # Corroboration: t asks s, v2's direction.
                cost = self._correlate(q[:, t], k[:, s])
                w = torch.softmax(cost / self.temperature, dim=1)
                fl = torch.einsum("ndzyx,dc->nczyx", w, self.offsets.to(cost.dtype))
                ctxs.append(_warp(v[:, s], fl))
                agrees.append(cost.amax(dim=1, keepdim=True))

            # A proposal from any neighbour is enough to recover a cell -- the
            # cell need not be visible in all of them -- so these take the max,
            # not the mean. `agree` keeps v2's configurable reduction, where
            # "min" is the strict "corroborated by every frame".
            prop = torch.stack(props).amax(dim=0)
            support = torch.stack(supports).amax(dim=0)
            ctx = torch.stack(ctxs).mean(dim=0)
            st = torch.stack(agrees)
            agree = st.mean(dim=0) if self.agree == "mean" else st.amin(dim=0)

            xt = x[:, t]
            support_gate = torch.sigmoid(
                self.support_logit_scale * support + self.support_logit_bias
            )
            agree_gate = torch.sigmoid(
                self.agree_logit_scale * agree + self.agree_logit_bias
            )
            recovered = self.recover(torch.cat([xt, prop, support], dim=1))
            modulated = self.corroborate(torch.cat([xt, ctx, agree], dim=1))

            out.append(
                xt
                + self.modulate_scale * 2.0 * (agree_gate - 0.5) * modulated
                + self.recover_scale * support_gate * recovered
            )
        return torch.stack(out, dim=1)


class TemporalUNet3Dv3(TemporalUNet3Dv2):
    """v2's per-frame encoder and caching, with ``_RecoveryFusion`` at the sites.

    Everything ``TemporalUNet3Dv2`` documents still applies -- the geometry
    arguments, ``temporal_fusion_at``, ``encode_frames`` / ``decode_window`` /
    ``FrameFeatureCache``, ``temporal_dilation="auto"``. Two additions:

    temporal_flow_smooth_iters : int
        Iterations of confidence-weighted blurring applied to the displacement
        field before the splat, so a voxel with no correspondence inherits the
        motion of the cells around it. 0 is the ablation.
    temporal_motion_prior : float
        Weight of the squared-distance prior on a proposal's displacement, in
        units of the median frame-to-frame step. 0 reproduces v2's
        appearance-only search, which is the ablation to run before believing
        the prior does anything.
    temporal_voxel_um : (float, float, float)
        Voxel size of the *input* data in microns. Each site scales it by its
        own stride, so a block at the bottleneck gets a physically larger voxel
        than one at the head without that having to be configured twice.

    The default ``temporal_fusion_at`` is ``("fine",)`` rather than v2's
    ``("head",)``: a recovered cell has to exist on the grid the detection is
    read from, and with an extra decode step that is the fine branch. On a
    model without one the two are the same place; ``TemporalUNet3Dv2``'s check
    will say so.
    """

    def __init__(
        self,
        *args,
        temporal_motion_prior: float = 1.0,
        temporal_objectness_weight: float = 1.0,
        temporal_voxel_um: Sequence[float] = SCALE_UM,
        temporal_flow_smooth_iters: int = 3,
        temporal_flow_smooth_kernel: Sequence[int] = (3, 7, 7),
        **kwargs,
    ) -> None:
        # Stashed before super().__init__, which builds the blocks and so calls
        # `_build_fusion` on the way through.
        self._motion_prior = float(temporal_motion_prior)
        self._objectness_weight = float(temporal_objectness_weight)
        self._voxel_um = tuple(float(v) for v in temporal_voxel_um)
        self._flow_smooth_iters = int(temporal_flow_smooth_iters)
        self._flow_smooth_kernel = tuple(int(v) for v in temporal_flow_smooth_kernel)
        kwargs.setdefault("temporal_fusion_at", ("fine",) if kwargs.get(
            "n_extra_decode_steps", 0) else ("head",))
        super().__init__(*args, **kwargs)

    def _build_fusion(self, channels: int, stride: Sequence[int]) -> nn.Module:
        kw = dict(self._fusion_kwargs)
        dil = (
            dilation_for_stride(stride, kw["taps"])
            if self._temporal_dilation == "auto"
            else self._temporal_dilation
        )
        return _RecoveryFusion(
            channels,
            dilation=dil,
            voxel_um=tuple(
                float(u) * int(s) for u, s in zip(self._voxel_um, _to_3tuple(stride))
            ),
            motion_prior=self._motion_prior,
            objectness_weight=self._objectness_weight,
            flow_smooth_iters=self._flow_smooth_iters,
            flow_smooth_kernel=self._flow_smooth_kernel,
            **kw,
        )
