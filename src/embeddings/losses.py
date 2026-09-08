"""NT-Xent, as ASCENT uses it: SimCLR's loss over the cells of one frame.

The negatives are the *other cells in the pair of views*, in both views. That is
what makes the task the right one: the embedding does not have to separate this
cell from cells in some unrelated image, it has to separate it from its own
neighbours, which is exactly the decision the matcher makes frame to frame.

Which cell is the positive is left to the caller, as a pair of index maps. Two
views of one frame pair cell i with cell i; a real pair of consecutive frames
pairs whatever the ground-truth graph links, and leaves the rest of the rows
unpaired -- the annotation covers ~1 cell a frame here. An unpaired row still
serves as a *negative* for the rows that are paired; it just contributes no
term of its own. That is what makes partial annotation usable without any
notion of an "unlabelled" class: a negative needs no label, only the knowledge
that it is not the positive, and within a frame that is free.

Temperature is the paper's tau = 0.05, sharper than the 0.1 the repo's
supervised contrastive stage uses. At this temperature the loss is dominated by
the hardest negatives -- the nearby cell that looks almost the same -- which is
where the errors are.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class NTXentOut(NamedTuple):
    loss: torch.Tensor
    top1: torch.Tensor          # scalar, averaged over samples
    n_pairs: torch.Tensor       # scalar, scored rows in the batch
    sample_top1: torch.Tensor   # (B,) per sample, 0 where it scored nothing
    sample_rows: torch.Tensor   # (B,) scored rows per sample


def nt_xent(
    z1: torch.Tensor,          # (B, N, D) L2-normalised
    z2: torch.Tensor,          # (B, N, D) L2-normalised
    mask_1: torch.Tensor,      # (B, N) bool: a real cell, present in view 1
    mask_2: torch.Tensor,      # (B, N) bool: same for view 2
    pos_1to2: torch.Tensor,    # (B, N) int64: row in view 2, or -1
    pos_2to1: torch.Tensor,    # (B, N) int64: row in view 1, or -1
    temperature: float = 0.05,
) -> NTXentOut:
    """Returns loss, top-1 and the per-sample breakdown of both.

    ``top1`` is the fraction of scored cells whose positive is its nearest
    neighbour among all candidates. It is the number to watch: the loss falls as
    soon as the easy negatives are pushed away, while top-1 only moves when the
    cells that actually confuse the matcher come apart.

    Every sample in the batch is its own contrastive problem -- cells from
    different frames are not negatives for each other, since they are different
    scenes and telling them apart is trivial and teaches nothing.

    The reduction is a mean **over samples** of each sample's mean over its
    scored rows, not one mean over every row in the batch. The two agree when
    every sample scores a comparable number of rows, which was true when the
    only pairing was synthetic. It is emphatically not true once real pairs are
    mixed in: a synthetic pair scores every cell of the frame (~500 rows) and a
    real one scores its annotated link (~2). Pooling the rows would let a
    ``real_prop`` of 0.5 buy 0.4% of the gradient, and the knob would be a lie.
    """
    N = z1.shape[1]
    device = z1.device

    # Candidates are both views: 2N per frame. A row's positive is wherever the
    # caller's map points; itself in its own view is excluded.
    z = torch.cat([z1, z2], dim=1)                       # (B, 2N, D)
    present = torch.cat([mask_1, mask_2], dim=1)         # (B, 2N)
    # View 2's rows are offset by N in the concatenated candidate axis; -1 stays
    # -1, so it is still recognisable as "no positive" below.
    target = torch.cat(
        [torch.where(pos_1to2 >= 0, pos_1to2 + N, pos_1to2), pos_2to1], dim=1
    )                                                    # (B, 2N)

    sim = torch.bmm(z, z.transpose(1, 2)) / temperature  # (B, 2N, 2N)
    eye = torch.eye(2 * N, dtype=torch.bool, device=device).unsqueeze(0)
    sim = sim.masked_fill(eye, float("-inf"))            # never match yourself
    sim = sim.masked_fill(~present.unsqueeze(1), float("-inf"))  # nor an absent cell

    safe = target.clamp(min=0)
    # A row is scored only if it is present, has a positive, and that positive
    # is itself present. The last is belt and braces -- the dataset already
    # drops edges into a cell the augmentation removed -- but a positive masked
    # to -inf would otherwise contribute an infinite loss rather than nothing.
    rows = present & (target >= 0) & present.gather(1, safe)

    log_p = torch.log_softmax(sim, dim=-1).nan_to_num()
    picked = log_p.gather(2, safe.unsqueeze(2)).squeeze(2)           # (B, 2N)

    sample_rows = rows.sum(dim=1)                                    # (B,)
    denom = sample_rows.clamp(min=1)
    sample_loss = -(picked * rows).sum(dim=1) / denom
    # Samples that scored nothing are dropped from the mean rather than counted
    # as zero, and never divided by zero. With no scored row anywhere the loss
    # is zero and every parameter still reaches the graph, which is what DDP
    # needs on a step an early return would have skipped.
    scored = (sample_rows > 0).to(sample_loss.dtype)
    loss = (sample_loss * scored).sum() / scored.sum().clamp(min=1)

    hit = (sim.argmax(dim=-1) == target) & rows
    sample_top1 = hit.sum(dim=1) / denom
    top1 = (sample_top1 * scored).sum() / scored.sum().clamp(min=1)
    # n_pairs stays a device tensor: an ``int()`` here would block the host on
    # the whole step just to read a diagnostic, once per iteration.
    return NTXentOut(loss, top1.detach(), sample_rows.sum(),
                     sample_top1.detach(), sample_rows)


@torch.no_grad()
def alignment_uniformity(z1, z2, pos_1to2, t: float = 2.0):
    """Wang & Isola's two diagnostics, which pull apart what the loss merges.

    Alignment is how close the two views of one cell sit; uniformity is how
    spread the embeddings are overall. A collapsing model -- every cell mapping
    to the same vector -- has perfect alignment and no uniformity, and its loss
    can look reasonable while its embeddings are worthless, so the pair is worth
    logging even though neither is optimised directly.

    Only the rows with a known positive are used, which for a real pair is the
    handful the ground truth annotates.
    """
    m = pos_1to2 >= 0
    if not m.any():
        return float("nan"), float("nan")
    a = z1[m]
    b = z2.gather(1, pos_1to2.clamp(min=0).unsqueeze(2).expand(-1, -1, z2.shape[-1]))[m]
    align = float((a - b).norm(dim=-1).pow(2).mean())
    sq = torch.cdist(a, a).pow(2)
    off = ~torch.eye(len(a), dtype=torch.bool, device=a.device)
    unif = float(sq[off].mul(-t).exp().mean().log()) if len(a) > 1 else float("nan")
    return align, unif
