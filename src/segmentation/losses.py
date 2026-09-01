"""The Cellpose objective: regress the flow field, classify cell interior."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellposeLoss(nn.Module):
    """``w_flow * MSE(flows) + w_cellprob * BCE(cellprob)``.

    Two things follow Stringer et al. rather than being tuned here.

    The flow term is a plain MSE over the *whole* volume, background included,
    where the target is zero. Masking it to the cell interior would leave the
    background flow unconstrained, and at inference every voxel above the
    probability threshold gets followed -- including the ones that only just
    cross it. A model with no opinion outside cells sends those on a random
    walk into whichever cell they drift toward.

    Flows are trained at ``FLOW_SCALE`` (5x unit length). Regressing raw unit
    vectors puts the MSE an order of magnitude below the BCE, and the model
    then buys its loss reduction almost entirely from cell probability -- which
    is the half of the output that does not separate touching cells.
    """

    def __init__(self, w_flow: float = 1.0, w_cellprob: float = 1.0):
        super().__init__()
        self.w_flow = w_flow
        self.w_cellprob = w_cellprob

    def forward(self, preds, targets):
        flows = preds["flows"]                                  # (B, 3, D, H, W)
        cellprob_logits = preds["cellprob_logits"].squeeze(1)   # (B, D, H, W)

        loss_flow = F.mse_loss(flows, targets["flows"]) / 2.0
        loss_cellprob = F.binary_cross_entropy_with_logits(
            cellprob_logits, targets["cellprob"]
        )
        total = self.w_flow * loss_flow + self.w_cellprob * loss_cellprob
        return total, {
            "flow": loss_flow.item(),
            "cellprob": loss_cellprob.item(),
            "total": total.item(),
        }


@torch.no_grad()
def flow_error(preds, targets, eps: float = 1e-6) -> float:
    """Mean cosine error between predicted and target flow, inside cells only.

    The training MSE is dominated by background, where both are zero, so it
    keeps falling long after the field inside the cells has stopped improving.
    This is the number that tracks whether reconstruction will work: a cosine
    below ~0.9 in the interior means trajectories will leave for the wrong
    attractor no matter how good the probability map looks.
    """
    fg = targets["cellprob"] > 0.5
    if not fg.any():
        return float("nan")
    p = preds["flows"].permute(0, 2, 3, 4, 1)[fg]
    t = targets["flows"].permute(0, 2, 3, 4, 1)[fg]
    cos = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1) + eps)
    return float(cos.mean())
