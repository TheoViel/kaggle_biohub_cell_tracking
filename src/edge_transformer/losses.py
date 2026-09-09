"""The edge objective: focal BCE, on the edges the ground truth can judge.

Per-edge binary classification rather than the tracking pipeline's
softmax-over-parents. The two encode different priors. A softmax says "this
child has exactly one parent among the candidates", which is convenient but
false at the boundaries -- a cell entering the field of view has none, and the
normalisation makes every score depend on the competitors, so the same link
scores differently depending on how crowded its neighbourhood is. Independent
per-edge scores leave the one-to-one decision to the linker, where it can be
made globally over a whole frame, and let the model say "none of these" by
scoring them all low. The higher-order attention is what makes that affordable:
the competition between candidate links is already represented *inside* the
model, so the loss does not have to impose it.

Focal weighting is kept from the tracking pipeline. Among supervised edges
roughly one in ten is real, and without it the gradient is dominated by the
easy negatives -- the candidate on the far side of the cell that nothing was
ever going to link.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class EdgeLossOut(NamedTuple):
    loss: torch.Tensor
    n_supervised: torch.Tensor
    n_positive: torch.Tensor
    degree: torch.Tensor | None = None


def _segment_sum(values: torch.Tensor, seg: torch.Tensor, n_seg: int):
    """Sum ``values`` within each segment id in ``seg``."""
    out = torch.zeros(n_seg, device=values.device, dtype=values.dtype)
    return out.scatter_add(0, seg, values)


def _segment_logsumexp(values: torch.Tensor, seg: torch.Tensor, n_seg: int):
    """logsumexp of ``values`` within each segment id in ``seg``."""
    m = torch.full((n_seg,), float("-inf"), device=values.device, dtype=values.dtype)
    m = m.scatter_reduce(0, seg, values, reduce="amax", include_self=True)
    z = torch.zeros(n_seg, device=values.device, dtype=values.dtype)
    z = z.scatter_add(0, seg, (values - m[seg]).exp())
    return m + z.clamp_min(1e-12).log()


def competitor_rank_loss(
    logits: torch.Tensor,        # (B, E)
    target: torch.Tensor,        # (B, E) float
    supervised: torch.Tensor,    # (B, E) bool
    edge_index: torch.Tensor,    # (B, E, 2) long, into the window's node axis
    n_nodes: int,
) -> torch.Tensor:
    """Cross-entropy over the candidates competing for one endpoint.

    Per-edge BCE scores a link without ever comparing it to the links it loses
    to, yet that comparison is the decision the linker actually makes: 61% of
    this model's missed links are *substitutions*, where the Hungarian spent the
    source on a candidate the model ranked above the true one. Each of those
    costs the metric twice -- a false negative and the false positive that took
    its place -- so the term below asks the question the linker asks: given that
    this cell has a successor among its candidates, which one is it?

    Restricted to endpoints the ground truth says *do* have a partner, which is
    what keeps the module docstring's objection to a plain softmax from biting:
    a cell entering the field, or one whose track simply ends, is never asked to
    name a parent it does not have. Applied in both directions, since a division
    gives a parent two valid children but each child exactly one parent.
    """
    device = logits.device
    B, E = logits.shape
    pos = (target > 0.5) & supervised
    if not bool(pos.any()):
        return logits.sum() * 0.0

    batch_id = torch.arange(B, device=device).unsqueeze(1).expand(B, E)
    total, n_terms = logits.sum() * 0.0, 0

    for endpoint in (0, 1):
        node = edge_index[..., endpoint]                       # (B, E)
        seg = (batch_id * n_nodes + node).reshape(-1)          # (B*E,)
        flat_logits = logits.reshape(-1)
        n_seg = B * n_nodes

        # Only segments holding a positive are asked to name a winner; edges
        # outside the supervised set must not contribute to the denominator.
        usable = supervised.reshape(-1)
        masked = torch.where(usable, flat_logits, torch.full_like(flat_logits, -1e4))
        lse = _segment_logsumexp(masked, seg, n_seg)

        pos_flat = pos.reshape(-1)
        if not bool(pos_flat.any()):
            continue
        idx = pos_flat.nonzero(as_tuple=True)[0]
        nll = -(flat_logits[idx] - lse[seg[idx]])
        total = total + nll.sum()
        n_terms += idx.numel()

    return total / max(n_terms, 1)


def degree_loss(
    probs: torch.Tensor,         # (B, E) in [0, 1]
    edge_index: torch.Tensor,    # (B, E, 2) long, into the window's node axis
    edge_mask: torch.Tensor,     # (B, E) bool, real candidates (not padding)
    n_nodes: int,
    max_parents: int = 1,
    max_children: int = 2,
) -> torch.Tensor:
    """Penalise a node whose candidate links promise it an impossible degree.

    Biology fixes two hard bounds, and this ground truth obeys them exactly:
    across 25 stacks, no annotated cell has more than one parent (6358/6358)
    and none more than two children. So for every node,

        sum of p over its candidate parents  <= 1
        sum of p over its candidate children <= 2

    and the penalty is a squared hinge on the excess. One-sided on purpose: the
    bound above is a fact, the bound below is not. A cell entering the field of
    view, or one whose track simply begins, legitimately has no parent, and
    nothing here can tell that apart from a link the model missed -- so the
    minimum is left to the supervised term, which knows.

    What makes this worth more than its arithmetic is *where it applies*. The
    focal term only reaches edges adjacent to an annotated cell -- roughly one
    per frame against ~260 detections -- so the overwhelming majority of the
    graph receives no gradient at all. This constraint needs no annotation: it
    is just as true of a candidate pair the annotation has never heard of, and
    it says the one thing per-edge classification structurally cannot, which is
    that these candidates are competing rather than independent.

    It is also the objective the *linker* optimises. Inference solves a
    one-to-one assignment; training scores each edge alone. Two parents at 0.9
    for the same child cost the BCE nothing and cost the Hungarian a
    substitution. This is the relaxation of that constraint, so the two stages
    are finally asking for the same thing.

    ``max_children=2`` rather than 1 keeps a real division from being trained
    away -- they are only 0.09% of parents here, but they carry a tenth of the
    competition score and the model currently predicts none at all.
    """
    B, E = probs.shape
    device = probs.device
    p = probs * edge_mask
    batch_id = torch.arange(B, device=device).unsqueeze(1).expand(B, E)

    total = probs.sum() * 0.0
    for endpoint, cap in ((0, max_children), (1, max_parents)):
        seg = (batch_id * n_nodes + edge_index[..., endpoint]).reshape(-1)
        n_seg = B * n_nodes
        degree = _segment_sum(p.reshape(-1), seg, n_seg)
        # Padding shares node 0's segment, but contributes 0 to both sums
        # because `p` and this count are already masked -- so a padded row
        # neither adds degree nor makes an inactive node look active.
        active = _segment_sum(edge_mask.reshape(-1).to(p.dtype), seg, n_seg) > 0
        if not bool(active.any()):
            continue
        excess = (degree - cap).clamp_min(0.0)
        total = total + (excess[active] ** 2).mean()

    return total / 2.0


def edge_loss(
    logits: torch.Tensor,        # (B, E) or (B, E, 1)
    target: torch.Tensor,        # (B, E) float, 1 on ground-truth links
    supervised: torch.Tensor,    # (B, E) bool
    gamma: float = 2.0,
    pos_weight: float = 1.0,
    rank_weight: float = 0.0,
    degree_weight: float = 0.0,
    edge_index: torch.Tensor | None = None,   # (B, E, 2), needed by both extras
    edge_mask: torch.Tensor | None = None,    # (B, E) bool, needed by degree_weight
    weight: torch.Tensor | None = None,       # (B, E) per-edge, 1.0 = annotated
    n_nodes: int | None = None,
) -> EdgeLossOut:
    """Mean over samples of each sample's mean over its supervised edges.

    Two levels rather than one pooled mean over the batch's edges: windows differ
    several-fold in how many edges the annotation reaches -- a frame pair with
    three annotated cells supervises three times the edges of one with a single
    cell -- and pooling would weight those windows by their annotation density,
    which is a property of the labelling effort, not of the data.

    ``weight`` prices each edge, which is how pseudo-labels enter at less than
    an annotated edge's worth. It is applied at *both* levels, and has to be:
    weighting only within a window would let a window carrying nothing but
    pseudo-labels count as much as a fully annotated one, which is the opposite
    of discounting them. So the inner mean is weighted, and each window then
    enters the outer mean by its own mean weight -- 1 for an annotated window,
    ``pl_weight`` for a purely pseudo-labelled one, between for a mixture. With
    no ``weight`` every supervised edge is worth 1 and both levels collapse back
    to the plain means above, exactly.
    """
    logits = logits.squeeze(-1) if logits.dim() == 3 else logits
    supervised = supervised & torch.isfinite(logits.detach())

    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.sigmoid(logits) * target + torch.sigmoid(-logits) * (1 - target)
        loss = ((1 - p_t) ** gamma) * bce
        if pos_weight != 1.0:
            loss = loss * (1.0 + (pos_weight - 1.0) * target)

        n = supervised.sum(dim=1)
        w = supervised.to(loss.dtype)
        if weight is not None:
            w = w * weight.to(loss.dtype)
        w_sum = w.sum(dim=1)
        per_sample = (loss * w).sum(dim=1) / w_sum.clamp(min=1e-8)
        # A window the annotation cannot reach contributes nothing and is not
        # counted; the sum still keeps every parameter attached to the graph,
        # which is what DDP needs on a step that would otherwise be skipped.
        #
        # `scored` is the window's mean weight -- 1 for an annotated window,
        # pl_weight for a purely pseudo-labelled one -- and the denominator
        # counts supervised windows rather than summing those weights. That
        # distinction is the whole discount: dividing by the sum would cancel
        # the weights out of any batch whose windows are *all* pseudo-labelled,
        # which is most of them once pseudo-labels unlock the windows the
        # annotation never reached. Counting instead lets such a batch
        # contribute pl_weight as much as an annotated one, which is what
        # trusting it less means. With no `weight` every scored window is 1 and
        # this is the plain mean it always was.
        scored = w_sum / n.clamp(min=1).to(loss.dtype)
        n_scored = (n > 0).to(loss.dtype).sum()
        total = (per_sample * scored).sum() / n_scored.clamp(min=1e-8)

        if rank_weight and edge_index is not None and n_nodes:
            total = total + rank_weight * competitor_rank_loss(
                logits, target, supervised, edge_index, int(n_nodes)
            )

        degree = None
        if degree_weight and edge_index is not None and n_nodes:
            # Unsupervised, so it reads edge_mask -- every real candidate -- and
            # not `supervised`, which is the whole point of it.
            mask = edge_mask if edge_mask is not None else torch.isfinite(logits)
            degree = degree_loss(
                torch.sigmoid(logits), edge_index, mask, int(n_nodes)
            )
            total = total + degree_weight * degree

    return EdgeLossOut(total, n.sum(), (target * supervised).sum(), degree)
