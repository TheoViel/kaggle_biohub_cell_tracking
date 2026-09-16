"""Combining several detectors' detection logits into one heatmap.

The public pipeline's headline ensemble is a second, independently seeded
detector whose *detection* logits are mixed into the primary's before peaks are
found (`PublicChanges.md`, "A adds a dual-seed ensemble"): the secondary is
aligned to the primary by mean and standard deviation, the scale ratio clamped
to `[0.5, 2.0]`, and mixed at weight 0.475.

Alignment is the part that carries the idea. Two networks trained separately put
their logits on different scales -- one confident at +6 where the other saturates
at +3 -- and averaging raw logits then silently weights the louder one. Matching
the second's mean and spread to the first's puts them on a common footing. The
clamp stops that correction running away on a frame where the secondary is
nearly flat and its standard deviation is close to zero, which would otherwise
multiply its noise by a huge factor.

One consequence is easy to miss and decides how these are used: **alignment
shifts the whole logit volume**, and the detection threshold is applied after
it. Measured over six fold-0 stacks with `2026-09-11/1` as primary and
`2026-09-15/0` as secondary, at the threshold tuned for the unblended primary
(0.40) the aligned blend looks like a heavy loss -- node ratio 0.114 against
0.237, adjusted Jaccard 0.8177 against 0.8684. Given its own threshold it is the
only rule that wins:

    mode          best thr   adj J    vs primary
    aligned           0.30   0.8735      +0.0050
    primary           0.40   0.8684            -
    max_logit         0.40   0.8618      -0.0066
    mean_logit        0.30   0.8585      -0.0099
    mean_prob         0.30   0.8583      -0.0101
    secondary         0.40   0.8343      -0.0342

So a blended run has to re-sweep the threshold; inheriting the single model's is
what makes the rule look broken. `aligned` reaches the highest edge Jaccard of
any arm at any threshold (0.9008 against the primary's best 0.8884) and pays
part of it back through the node-count adjustment.
"""

from __future__ import annotations

import torch

CLAMP = (0.5, 2.0)          # the public pipeline's scale-ratio clamp
WEIGHT = 0.475              # and its mix weight on the secondary
MODES = ("aligned", "mean_logit", "mean_prob", "max_logit", "primary")


def align(sec: torch.Tensor, pri: torch.Tensor, clamp=CLAMP) -> torch.Tensor:
    """``sec`` put on ``pri``'s scale: match the mean, match the spread."""
    ratio = torch.clamp(pri.std() / (sec.std() + 1e-6), clamp[0], clamp[1])
    return (sec - sec.mean()) * ratio + pri.mean()


def blend_logits(pri: torch.Tensor, secs, mode: str = "aligned",
                 weight: float = WEIGHT, clamp=CLAMP) -> torch.Tensor:
    """One detection-logit volume from a primary and any number of secondaries.

    ``weight`` is the total mass on the secondaries, split equally between them,
    so two secondaries at 0.475 contribute 0.2375 each and the primary keeps
    0.525 whatever their number.

    ``max_logit`` ignores ``weight`` -- it is an OR over the models and has no
    mixing to weight. It is also the only rule here that is symmetric in its
    arguments, which is worth knowing when reading a comparison: swapping
    primary and secondary changes every other mode's answer and not this one.
    """
    secs = [s for s in secs if s is not None]
    if mode == "primary" or not secs:
        return pri
    if mode == "max_logit":
        out = pri
        for s in secs:
            out = torch.maximum(out, s)
        return out

    if mode == "mean_prob":
        # Averaged after the sigmoid, then mapped back: everything downstream --
        # the threshold, the NMS, the node probabilities -- expects a logit.
        p = (1.0 - weight) * torch.sigmoid(pri)
        for s in secs:
            p = p + (weight / len(secs)) * torch.sigmoid(s)
        return torch.logit(p.clamp(1e-6, 1.0 - 1e-6))

    if mode == "mean_logit":
        parts = secs
    elif mode == "aligned":
        parts = [align(s, pri, clamp) for s in secs]
    else:
        raise ValueError(f"unknown blend mode {mode!r}; one of {MODES}")

    out = (1.0 - weight) * pri
    for s in parts:
        out = out + (weight / len(parts)) * s
    return out
