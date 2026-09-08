"""HOCT: the Higher-Order Cell Tracking Transformer, as an edge classifier.

A reimplementation of the architecture royerlab ships as a JIT blob in ``hoct/``
(``hoct.load_model`` -> ``slot_cell_tracking.modules.edge_model.EdgeModel``).
The published package is inference-only and its weights are TorchScript, so
this file is the trainable source form; the module and parameter names follow
the checkpoint's so a published model can still be read into it.

What makes it "higher-order" is the second attention stack. Most learned
linkers score a candidate edge from its two endpoints and whatever context the
*nodes* gathered -- an edge never sees another edge, so nothing in the model can
represent "these two cells cannot both come from that parent" or "this link
disagrees with the motion every link around it agrees on". Here the candidate
edges are themselves tokens in a second transformer, and they attend to each
other with a bias read off their geometry, so the score of a link is a function
of the other links competing with and surrounding it.

The stack, in order:

1. ``input_proj``: node features -> ``d_model``.
2. ``node_blocks``: self-attention over the nodes of the window. Attention is
   masked to pairs closer than ``max_distance``, which is what keeps the cost
   sane on a frame of several hundred detections and also states the prior that
   a cell's context is local.
3. ``edge_gatherer``: an MLP over ``[node_i, node_j]`` turns each candidate edge
   into a token. Explicit per-edge features (distance, dt, appearance
   similarity, ...) are projected by ``edge_input_proj`` and added here.
4. ``edge_blocks``: self-attention over the *edges*, biased by the distance
   between the two candidate links seen as segments in space -- see
   ``segment_distance``. Half the heads get the bias positively and half
   negatively (``dist_scaling_head_direction``), so some heads look at links
   near this one and others at links far from it, with the strength learned per
   head and per block.
5. ``head``: one logit per edge. ``orphan_head`` is optional and off by default,
   as it is in the published ``general_v1`` weights, which return zeros there.

Positions enter through rotary embeddings on the queries and keys rather than as
added features, so what the attention sees is *relative* position -- the right
invariance for a field of cells that has no origin.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelPrediction(NamedTuple):
    """The four tensors ``hoct.inference.EdgeModel`` contracts to return."""

    edge_logits: torch.Tensor      # (B, E, 1)
    node_features: torch.Tensor    # (B, N, d_model)
    edge_features: torch.Tensor    # (B, E, d_model)
    orphan_logits: torch.Tensor    # (B, N, 1)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def masked_attn_bias(
    pos: torch.Tensor,           # (B, S, D)
    mask: torch.Tensor,          # (B, S) bool, True where the token is real
    max_distance: float | None,
) -> torch.Tensor:               # (B, 1, S, S) additive, 0 or -inf
    """Who may attend to whom: padding out, and anything past ``max_distance``.

    Returned as an additive bias rather than a boolean mask so the edge blocks
    can add their own distance term to it without a second masking pass.

    A row with nothing left to attend to is reopened onto everything. Softmax
    over an all -inf row is NaN, and that NaN would not stay in its own row: a
    padded token's value vector is multiplied by an exactly-zero weight in every
    other row, and ``0 * NaN`` is NaN, so one dead row poisons the whole batch.
    The caller masks these tokens out downstream anyway, so the reopened row's
    output is never read.
    """
    B, S, _ = pos.shape
    pair = mask.unsqueeze(2) & mask.unsqueeze(1)                 # (B, S, S)
    if max_distance is not None:
        # Squared distance in float32: at fp16 a distance of a few hundred
        # microns squares into the tail of the format's range.
        sq = torch.cdist(pos.float(), pos.float()).pow_(2)
        pair = pair & (sq <= float(max_distance) ** 2)
    pair = pair | ~pair.any(dim=-1, keepdim=True)
    bias = torch.zeros(B, S, S, dtype=torch.float32, device=pos.device)
    return bias.masked_fill_(~pair, float("-inf")).unsqueeze(1)


def segment_distance(
    node_pos: torch.Tensor,      # (B, N, D)
    edge_indices: torch.Tensor,  # (B, E, 2) long
) -> torch.Tensor:               # (B, E, E)
    """Distance between every pair of candidate links, taken as line segments.

    A candidate edge is the segment from where a cell is to where it might have
    gone. Two such segments being close means the two links describe the same
    piece of motion -- which is precisely the relation the edge attention needs:
    a link is judged against the links it competes with (same endpoint, distance
    0) and against the links that share its neighbourhood.

    This is the standard closest-approach of two finite segments, with the
    parameters clamped into [0, 1] so it is the distance between the segments
    rather than between the infinite lines they lie on, and with the degenerate
    near-parallel case (``D`` below vanishing) handled separately -- there the
    two lines have no unique closest pair and the projection of one endpoint
    onto the other segment is taken instead.
    """
    i, j = edge_indices.unbind(-1)
    d = node_pos.shape[-1]
    p0 = node_pos.gather(1, i.unsqueeze(-1).expand(-1, -1, d)).float()   # (B, E, D)
    p1 = node_pos.gather(1, j.unsqueeze(-1).expand(-1, -1, d)).float()
    u = p1 - p0                                                          # direction

    p0_i, p0_j = p0.unsqueeze(2), p0.unsqueeze(1)
    u_i, u_j = u.unsqueeze(2), u.unsqueeze(1)
    v = p0_i - p0_j

    a = (u_i * u_i).sum(-1)
    b = (u_i * u_j).sum(-1)
    c = (u_j * u_j).sum(-1)
    e_ = (u_i * v).sum(-1)
    f_ = (u_j * v).sum(-1)
    denom = a * c - b * b

    parallel = denom < 1e-4
    t_parallel = (f_ / c.clamp(min=1e-6)).clamp(0.0, 1.0)
    s = ((b * f_ - c * e_) / denom.clamp(min=1e-6)).clamp(0.0, 1.0)
    t = ((b * s + f_) / c.clamp(min=1e-6)).clamp(0.0, 1.0)
    s = ((b * t - e_) / a.clamp(min=1e-6)).clamp(0.0, 1.0)

    s = torch.where(parallel, torch.zeros_like(s), s)
    t = torch.where(parallel, t_parallel, t)

    closest_i = p0_i + s.unsqueeze(-1) * u_i
    closest_j = p0_j + t.unsqueeze(-1) * u_j
    return (closest_i - closest_j).norm(dim=-1)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class RoPEmbedding(nn.Module):
    """N-dimensional rotary position embedding, with a learned frequency set.

    Each head's ``head_dim`` channels are cut into ``n_freqs * n_pos_dims``
    pairs, and the pair for (frequency f, axis d) is rotated by
    ``pos_d * exp(log_freq_f) / sqrt(n_pos_dims)``. Rotating q and k this way
    makes their inner product a function of the *difference* of positions, which
    is the invariance wanted here: a tracking decision does not depend on where
    in the volume the cells happen to sit.

    ``log_freq`` is a parameter, per head and per frequency, so the model can
    move a head onto the length scale it needs. It is initialised as a geometric
    spread between two wavelengths, which is easier to reason about than raw
    frequencies: ``min_wavelength_um`` should be around the finest spatial
    detail worth resolving and ``max_wavelength_um`` around the field of view.

    ``reflect_vec`` reproduces the published model's Householder reflection,
    ``M = I - 2 v v^T``, applied to q and k after the rotation. Note that the
    *same* orthogonal M is applied to both, so it cancels exactly in ``q . k``
    and cannot change any attention weight: it is kept for parameter-level
    compatibility with the released checkpoints and can be switched off with
    ``householder=False`` without altering what the model computes.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        n_pos_dims: int = 3,
        pos_scale: float = 20.0,
        min_wavelength: float = 4.0,
        max_wavelength: float = 200.0,
        householder: bool = True,
    ):
        super().__init__()
        if head_dim % (2 * n_pos_dims) != 0:
            raise ValueError(
                f"head_dim={head_dim} must be a multiple of 2 * n_pos_dims="
                f"{2 * n_pos_dims}: every (frequency, axis) pair takes two "
                f"channels. With {n_pos_dims} axes and {n_heads} heads, try "
                f"d_model={n_heads * 2 * n_pos_dims * (head_dim // (2 * n_pos_dims) or 1)}."
            )
        self.n_pos_dims = int(n_pos_dims)
        self.pos_scale = float(pos_scale)
        n_freqs = head_dim // (2 * n_pos_dims)

        # Wavelength lambda satisfies (lambda / pos_scale) * f / sqrt(D) = 2 pi.
        norm = 2.0 * math.pi * math.sqrt(n_pos_dims) * pos_scale
        f_hi = norm / float(min_wavelength)
        f_lo = norm / float(max_wavelength)
        freqs = torch.logspace(math.log10(f_hi), math.log10(f_lo), n_freqs)
        # (1, H, 1, F, 1): broadcast against a (B, 1, N, 1, D) position.
        self.log_freq = nn.Parameter(
            freqs.log().view(1, 1, 1, n_freqs, 1).repeat(1, n_heads, 1, 1, 1)
        )

        self.householder = bool(householder)
        if self.householder:
            self.reflect_vec = nn.Parameter(torch.randn(n_heads, head_dim))
            self.register_buffer("eye", torch.eye(head_dim), persistent=True)

    def _cos_sin(self, pos: torch.Tensor, n_heads: int):
        B, N, D = pos.shape
        p = (pos.float() / self.pos_scale).reshape(B, 1, N, 1, D)
        angle = p * self.log_freq.exp() / math.sqrt(self.n_pos_dims)
        cos, sin = angle.cos(), angle.sin()
        # A position that arrives non-finite (an unset padding sentinel) would
        # otherwise put NaN into every token it is compared with.
        cos = torch.where(cos.isnan(), torch.ones_like(cos), cos)
        sin = torch.where(sin.isnan(), torch.zeros_like(sin), sin)
        cos = cos.repeat_interleave(2, dim=-1).reshape(B, n_heads, N, -1)
        sin = sin.repeat_interleave(2, dim=-1).reshape(B, n_heads, N, -1)
        return cos, sin

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """(B, H, N, head_dim) queries or keys, rotated by ``pos`` (B, N, D)."""
        cos, sin = self._cos_sin(pos, x.shape[1])
        # Interleaved rotation: channels (0, 1), (2, 3), ... are the 2-vectors.
        rotated = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape_as(x)
        out = x * cos.to(x.dtype) + rotated * sin.to(x.dtype)
        if self.householder:
            v = F.normalize(self.reflect_vec, dim=-1, eps=1e-12)
            m = self.eye - 2.0 * (v.unsqueeze(-1) * v.unsqueeze(-2))
            out = torch.einsum("bhnd,hde->bhne", out, m.to(out.dtype))
        return out


class MLP(nn.Module):
    """The published model's two-layer MLP -- note the activation on the output.

    GELU after ``fc2`` as well as after ``fc1`` is unusual and is what the
    checkpoint does; it bounds the block's contribution below at ~-0.17 and is
    kept rather than tidied away.
    """

    def __init__(self, dim: int, hidden: int, out: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc2 = nn.Linear(hidden, out or dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        return self.drop2(self.act2(self.fc2(self.drop1(self.act1(self.fc1(x))))))


class Attention(nn.Module):
    """Gated multi-head attention with rotary positions and an additive bias.

    Two departures from a textbook block, both from the published model:

    * ``q_proj`` emits twice ``d_model``; the second half is a **gate**, and the
      attention output is multiplied by its sigmoid. A head can therefore
      abstain on a token -- useful when a node's neighbourhood is empty or a
      candidate edge has no competitor, where the attended value is whatever the
      mask left behind and should not be trusted.
    * queries and keys carry no bias, and the RoPE is applied to them after
      projection rather than added to the input.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        pos_enc: RoPEmbedding,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"d_model={dim} must divide by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.pos_enc = pos_enc
        self.q_proj = nn.Linear(dim, dim * 2, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = float(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,                  # (B, S, C) query source
        y: torch.Tensor | None,           # (B, S, C) key/value source, or x
        pos: torch.Tensor,                # (B, S, D)
        attn_mask: torch.Tensor,          # (B, 1|H, S, S) additive
    ) -> torch.Tensor:
        B, S, C = x.shape
        y = x if y is None else y
        q, gate = (
            self.q_proj(x).reshape(B, S, self.n_heads, -1).permute(0, 2, 1, 3).chunk(2, dim=-1)
        )
        k, v = (
            self.kv_proj(y)
            .reshape(B, y.shape[1], 2, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q = self.pos_enc(q, pos)
        k = self.pos_enc(k, pos)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask.to(q.dtype),
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        out = out * torch.sigmoid(gate)
        out = out.transpose(1, 2).reshape(B, S, C)
        return self.proj_drop(self.proj(out))


class AttnBlock(nn.Module):
    """Pre-norm attention + MLP, optionally with the higher-order distance bias.

    ``dist_bias`` (the edge stack) adds ``direction * softplus(scale) * dist``
    to the attention logits, with ``scale`` learned per head and ``direction``
    fixed at +1 for the first half of the heads and -1 for the second. The sign
    is what makes the two halves ask different questions: a -1 head attends to
    the links *closest* to this one -- its direct competitors -- and a +1 head to
    the ones furthest away, which is the frame-wide context. Leaving the sign
    free would let every head collapse onto the same one, and fixing it costs
    nothing since the magnitude is still learned.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        pos_enc: RoPEmbedding,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        dist_bias: bool = False,
        separate_kv_norm: bool = False,
    ):
        super().__init__()
        self.norm1_x = nn.RMSNorm(dim)
        # Queries and keys come from the same tensor, so a second norm is a free
        # extra gain rather than a second input. The published model has one in
        # the edge blocks and shares norm1_x in the node blocks (its node-block
        # norm1_y is still at its all-ones init, i.e. never used), and that is
        # reproduced here so its weights load with the same meaning.
        self.norm1_y = nn.RMSNorm(dim) if separate_kv_norm else None
        self.attn = Attention(dim, n_heads, pos_enc, dropout=dropout)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

        self.dist_bias = bool(dist_bias)
        if self.dist_bias:
            self.dist_scaling = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
            direction = torch.ones(n_heads)
            direction[n_heads // 2:] = -1.0
            self.register_buffer(
                "dist_scaling_head_direction", direction.view(1, n_heads, 1, 1),
                persistent=True,
            )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        attn_mask: torch.Tensor,
        dist: torch.Tensor | None = None,      # (B, S, S)
    ) -> torch.Tensor:
        if self.dist_bias:
            if dist is None:
                raise ValueError("a dist_bias block needs the pairwise distance")
            scale = self.dist_scaling_head_direction * F.softplus(self.dist_scaling)
            attn_mask = attn_mask + scale * dist.unsqueeze(1)
        normed = self.norm1_x(x)
        x = x + self.attn(
            normed, self.norm1_y(x) if self.norm1_y is not None else normed,
            pos, attn_mask,
        )
        return x + self.mlp(self.norm2(x))


class EdgeGatherer(nn.Module):
    """A candidate edge's token, from the two node tokens it joins."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(dim * 2, dim, out=dim, dropout=dropout)

    def forward(self, edge_indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        i, j = edge_indices.unbind(-1)
        c = x.shape[-1]
        e_i = x.gather(1, i.unsqueeze(-1).expand(-1, -1, c))
        e_j = x.gather(1, j.unsqueeze(-1).expand(-1, -1, c))
        return self.mlp(torch.cat([e_i, e_j], dim=-1))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class HOCT(nn.Module):
    """The edge transformer.

    Args:
        node_feat_dim: width of the per-node feature vector the dataset builds.
        edge_feat_dim: width of the per-edge feature vector, or 0 for none. The
            edge tokens are otherwise a function of the two node tokens alone,
            and an MLP over a concatenation is a poor way to compute things like
            a cosine similarity or a distance rank -- so the pairwise quantities
            that are known to carry this task are handed over directly.
        d_model / n_heads / n_node_blocks / n_edge_blocks / mlp_ratio: the
            published model's 288 / 4 / 4 / 4 / 2.
        n_pos_dims: 3 (z, y, x, in microns). Time is a node *feature*, not a
            position, as in the published model: the rotary encoding is spatial
            and the edge geometry below is spatial, so putting time in here
            would silently redefine both.
        max_distance: attention cutoff, in microns, for both stacks.
        orphan_head: per-node "this cell has no parent" logit. Off by default,
            matching ``general_v1``, which returns zeros for it, and **not
            supervised by this repo's training pipeline** -- turning it on
            without adding a target for it gives a head that stays at its init.
            It is here because it is part of the model's published output
            contract and because a solver can use it to open a track
            deliberately rather than by failing to find a link.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int = 0,
        d_model: int = 288,
        n_heads: int = 4,
        n_node_blocks: int = 4,
        n_edge_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        n_pos_dims: int = 3,
        pos_scale: float = 20.0,
        min_wavelength: float = 4.0,
        max_wavelength: float = 200.0,
        max_distance: float | None = 40.0,
        householder: bool = True,
        orphan_head: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_distance = max_distance
        self.edge_feat_dim = int(edge_feat_dim)

        # Dropout1d over the feature axis: it drops whole input *channels*
        # rather than scattered scalars, so the model cannot lean on any single
        # feature -- the appearance embedding and the geometric ranks are each
        # strong enough alone to be leaned on.
        self.input_dropout = nn.Dropout1d(input_dropout) if input_dropout else nn.Identity()
        self.edge_feat_dropout = (
            nn.Dropout1d(input_dropout) if input_dropout and edge_feat_dim else nn.Identity()
        )
        self.input_proj = nn.Linear(node_feat_dim, d_model)
        self.edge_input_proj = (
            nn.Linear(edge_feat_dim, d_model) if edge_feat_dim else None
        )

        def rope():
            return RoPEmbedding(
                n_heads, d_model // n_heads, n_pos_dims=n_pos_dims,
                pos_scale=pos_scale, min_wavelength=min_wavelength,
                max_wavelength=max_wavelength, householder=householder,
            )

        self.node_blocks = nn.ModuleList([
            AttnBlock(d_model, n_heads, rope(), mlp_ratio, dropout, dist_bias=False)
            for _ in range(n_node_blocks)
        ])
        self.edge_gatherer = EdgeGatherer(d_model, dropout=dropout)
        self.edge_blocks = nn.ModuleList([
            AttnBlock(d_model, n_heads, rope(), mlp_ratio, dropout,
                      dist_bias=True, separate_kv_norm=True)
            for _ in range(n_edge_blocks)
        ])

        self.head_norm = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.head = nn.Linear(d_model, 1)
        self.orphan_head = nn.Linear(d_model, 1) if orphan_head else None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        node_feats: torch.Tensor,        # (B, N, node_feat_dim)
        node_pos: torch.Tensor,          # (B, N, 3) microns
        edge_pos: torch.Tensor,          # (B, E, 3) microns, the edge midpoint
        edge_indices: torch.Tensor,      # (B, E, 2) long, into the node axis
        node_mask: torch.Tensor,         # (B, N) bool
        edge_mask: torch.Tensor,         # (B, E) bool
        edge_feats: torch.Tensor | None = None,   # (B, E, edge_feat_dim)
    ) -> ModelPrediction:
        x = self.input_proj(self.input_dropout(node_feats))

        node_bias = masked_attn_bias(node_pos, node_mask, self.max_distance)
        for block in self.node_blocks:
            x = block(x, node_pos, node_bias)

        e = self.edge_gatherer(edge_indices, x)
        if self.edge_input_proj is not None:
            if edge_feats is None:
                raise ValueError(
                    f"the model was built with edge_feat_dim={self.edge_feat_dim} "
                    "but forward() was given no edge_feats"
                )
            e = e + self.edge_input_proj(self.edge_feat_dropout(edge_feats))

        edge_bias = masked_attn_bias(edge_pos, edge_mask, self.max_distance)
        # Computed once and shared by every edge block, which is the whole
        # reason the blocks take it as an argument: it is (B, E, E) and by far
        # the most expensive tensor in the forward pass.
        dist = segment_distance(node_pos, edge_indices)
        for block in self.edge_blocks:
            e = block(e, edge_pos, edge_bias, dist=dist)

        edge_logits = self.head(self.head_dropout(self.head_norm(e)))
        orphan_logits = (
            self.orphan_head(x) if self.orphan_head is not None
            else x.new_zeros(x.shape[0], x.shape[1], 1)
        )
        return ModelPrediction(edge_logits, x, e, orphan_logits)
