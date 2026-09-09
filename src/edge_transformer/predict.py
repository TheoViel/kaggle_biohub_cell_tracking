"""Whole-video inference: score every candidate link, then commit to a set.

Two steps, deliberately separate. The model scores each candidate edge
independently -- it never has to decide that a cell has exactly one successor --
and the linker then makes that decision over a whole frame pair at once, where
it is a global assignment rather than a per-edge argmax.

The assignment is one-to-one (Hungarian over the frame pair), which is the same
choice the tracking pipeline makes and carries the same known limitation: a
division cannot be represented, so a daughter starts a new track and
``division_jaccard`` stays NaN. The alternative -- the ILP the published HOCT
solves, which admits a parent taking two children under a cost -- is a
post-processing change, not a model change, and the per-edge probabilities this
writes are exactly what it would consume.
"""

from __future__ import annotations

import numpy as np
import torch

from edge_transformer.dataset import collate_graphs

# A forward materialises (B, H, E, E) attention scores in the edge blocks, so
# its memory goes as B * E^2, not B * E. Batches are therefore bounded by this
# budget on B * E_max^2 as well as by MAX_WINDOWS_PER_BATCH, which keeps the
# footprint flat whether a video's windows carry 1k or 4k candidate edges.
EDGE_PAIR_BUDGET = 32_000_000
MAX_WINDOWS_PER_BATCH = 16


def fuse_directions(
    forward: list[float],
    reverse: list[float],
    mode: str = "harmonic",
    weight: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """One probability for a pair, from its forward and time-reversed readings.

    Averaging happens twice, and the two are not the same question. Several
    windows may cover the same frame pair (``window_size > 2``), and those are
    the same model asked the same way under different context -- so they are
    meaned within a direction. The two *directions* are the model asked
    differently, and how to combine those is the choice ``mode`` makes:

    ``"mean"``      the weighted arithmetic mean, which is what plain averaging
                    of both passes does, and what this did before.
    ``"harmonic"``  the weighted harmonic mean, ``1 / ((1-w)/p_f + w/p_r)``.

    The harmonic mean is the one worth having. It is dominated by the *smaller*
    of the two -- a pair one direction likes and the other does not comes out
    near the doubter's value rather than halfway -- so it asks for mutual
    support instead of an average opinion. That matters here precisely because
    the two passes agree closely on average: an arithmetic mean of two nearly
    equal numbers is nearly the number itself and changes almost nothing, while
    a disagreement-penalising rule still moves the pairs that disagree, which
    are the ones in doubt.

    A pair seen in only one direction keeps that direction's value. It is not
    evidence against the pair: the reverse pass re-estimates the drift from the
    other end, so a borderline pair can fall either side of the candidate radius
    and simply not be proposed. Scoring the missing view as zero would delete
    links over a rounding difference in the drift estimate.
    """
    f = float(np.mean(forward)) if forward else None
    r = float(np.mean(reverse)) if reverse else None
    if f is None or r is None:
        return f if f is not None else (r if r is not None else 0.0)

    if mode == "mean":
        return (1.0 - weight) * f + weight * r
    if mode == "harmonic":
        return 1.0 / ((1.0 - weight) / max(f, eps) + weight / max(r, eps))
    raise ValueError(f"unknown tta fusion {mode!r}; use 'harmonic' or 'mean'")


def _plan_batches(sizes: list[int], max_windows: int, budget: int) -> list[list[int]]:
    """Group window indices into forwards: near-equal sizes, bounded memory.

    Windows are scored independently and accumulate into a dict, so they may be
    visited in any order. Taking them in order of edge count puts windows of
    near-equal width in the same forward, which is what keeps the padding to a
    few percent instead of the widest window in the video.
    """
    batches, batch = [], []
    for i in sorted(range(len(sizes)), key=sizes.__getitem__):
        # Ascending, so sizes[i] is what this batch would be padded out to.
        e = sizes[i]
        if batch and (len(batch) >= max_windows or (len(batch) + 1) * e * e > budget):
            batches.append(batch)
            batch = []
        batch.append(i)
    return batches + ([batch] if batch else [])


@torch.no_grad()
def score_video(
    model,
    dataset,
    video_idx: int,
    device: torch.device,
    fp16: bool = True,
    tta: bool = False,
    tta_fusion: str = "harmonic",
    tta_weight: float = 0.5,
    max_windows_per_batch: int = MAX_WINDOWS_PER_BATCH,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Every candidate edge of a video, as ``(src, tgt) -> (probability, distance)``.

    Indices are rows of the video's own ``coords`` array, so the result can be
    handed straight to ``build_graph``.

    With a window longer than two frames the same frame pair is scored in
    several windows, under different context each time; the probabilities are
    averaged rather than taken from one arbitrary window. At the default window
    of 2 there is exactly one score per pair and the averaging is a no-op.

    ``tta`` adds a second reading of every window with time reversed. The
    candidate set is symmetric under the reversal -- ``build_candidates`` unions
    the k-nearest in both directions, so the same pairs are proposed either way
    -- and a flipped window's edges are mapped back to real time before they are
    accumulated. It costs one extra forward per window and is worth most on a
    model trained with ``time_flip_p``, which is what makes the reversed view
    something it has seen before. ``tta_fusion`` and ``tta_weight`` decide how
    the two directions are combined; see :func:`fuse_directions`.

    Windows are batched dynamically: sorted by candidate count, then grouped so
    a batch is padded only out to its *own* widest member rather than to any
    fixed cap. A window here is small -- a few hundred nodes, ~1-2k candidate
    edges -- so one forward per window leaves the device latency-bound: batching
    is worth ~10x wall clock, far more than the few percent of padding that
    size-sorted grouping costs. ``_plan_batches`` bounds the batch by memory as
    well as by count, since the edge blocks cost ``B * E_max^2``.
    """
    video = dataset.videos[video_idx]
    frames = video.frames.astype(int)
    starts = [
        int(t) for t in frames
        if all(int(t) + k in video.rows_by_frame for k in range(dataset.window_size))
    ]

    was_training = model.training
    model.eval()
    # Forward and reversed readings kept apart: several windows may cover one
    # frame pair and those are meaned within a direction, while the two
    # directions are combined by `fuse_directions`, which is a different rule.
    total: dict[tuple[int, int], tuple[list[float], list[float]]] = {}
    dist_of: dict[tuple[int, int], float] = {}

    # keep_annotated=False: a real deployment has no ground truth to spare an
    # annotated node from the per-frame cap, and scoring this the way training
    # validates it would credit the model with a recall it cannot reach
    # standalone. See EdgeDataset.build's docstring.
    items = [
        dataset.build(video_idx, t, keep_annotated=False, time_flip=flip)
        for t in starts
        for flip in ((False, True) if tta else (False,))
    ]
    items = [it for it in items if it["edge_index"].shape[0]]

    sizes = [int(it["edge_index"].shape[0]) for it in items]
    for group in _plan_batches(sizes, max_windows_per_batch, EDGE_PAIR_BUDGET):
        batch = collate_graphs([items[i] for i in group])
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=fp16 and device.type == "cuda",
        ):
            out = model(
                batch["node_feats"].to(device),
                batch["node_pos"].to(device),
                batch["edge_pos"].to(device),
                batch["edge_index"].to(device),
                batch["node_mask"].to(device),
                batch["edge_mask"].to(device),
                edge_feats=batch["edge_feats"].to(device),
            )
        probs = torch.sigmoid(out.edge_logits.float().squeeze(-1)).cpu().numpy()

        for slot, i in enumerate(group):
            item = items[i]
            edges = item["edge_index"].numpy()
            glob = item["node_global"].numpy()
            pos = item["node_pos"].numpy()
            d = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
            # A reversed window proposes the same pair the other way round, so
            # its edges are turned back before they are keyed -- otherwise the
            # two views would accumulate into two separate, half-observed edges
            # and `link_video` would see a graph pointing both ways in time.
            flipped = item["time_flip"]
            # Rows past this window's own edge count are padding; ignore them.
            for e, (src, tgt) in enumerate(edges):
                a, b = int(glob[src]), int(glob[tgt])
                key = (b, a) if flipped else (a, b)
                seen = total.setdefault(key, ([], []))
                seen[1 if flipped else 0].append(float(probs[slot, e]))
                dist_of[key] = float(d[e])

    if was_training:
        model.train()
    return {
        k: (fuse_directions(fwd, rev, mode=tta_fusion, weight=tta_weight),
            dist_of[k])
        for k, (fwd, rev) in total.items()
    }


def link_video(
    video,
    scored: dict[tuple[int, int], tuple[float, float]],
    threshold: float = 0.5,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """One-to-one assignment per frame pair, over the scored candidates.

    Returns ``(coords, edges)`` in the form ``build_graph`` reads: ``coords`` is
    the video's own ``(N, 4)`` node array and each edge indexes its rows.

    Only consecutive-frame candidates are linked. A longer ``max_dt`` proposes
    gap-closing edges, and they are still scored and still shape the attention,
    but committing to one would produce an edge the competition metric does not
    score and a track the metric reads as two.
    """
    from scipy.optimize import linear_sum_assignment

    frame_of = video.coords[:, 0].astype(int)
    by_pair: dict[int, list[tuple[int, int, float, float]]] = {}
    for (a, b), (p, d) in scored.items():
        t = int(frame_of[a])
        if int(frame_of[b]) != t + 1:
            continue
        by_pair.setdefault(t, []).append((a, b, p, d))

    edges: list[tuple[int, int, float, float]] = []
    for cands in by_pair.values():
        parents = sorted({a for a, _, _, _ in cands})
        children = sorted({b for _, b, _, _ in cands})
        row = {a: i for i, a in enumerate(parents)}
        col = {b: j for j, b in enumerate(children)}

        # Cost is the log-odds of the link *against the threshold*:
        # log(threshold) - log(p), which is negative exactly when the model
        # scores the pair above the threshold. Not linking is free (0), which
        # is the whole point.
        #
        # The obvious alternative -- a large finite cost for the pairs that were
        # never candidates -- is wrong, and expensively so. Every row must be
        # assigned something, so a large dummy cost makes the solver maximise
        # the *number* of real assignments before caring about their quality: it
        # will happily drop a link the model scored 0.66 to pick up two mediocre
        # ones elsewhere. That cost this 19% of all ground-truth links, which is
        # more than the difference between any two scorers here.
        log_th = float(np.log(min(max(threshold, 1e-6), 1 - 1e-6)))
        cost = np.zeros((len(parents), len(children)), np.float32)
        prob = np.zeros_like(cost)
        dist = np.zeros_like(cost)
        for a, b, p, d in cands:
            i, j = row[a], col[b]
            cost[i, j] = log_th - float(np.log(min(max(p, 1e-9), 1 - 1e-9)))
            prob[i, j] = p
            dist[i, j] = d

        r, c = linear_sum_assignment(cost)
        # Strictly below zero: a pair exactly at the threshold is worth no more
        # than leaving the cell unlinked, and the solver is indifferent there.
        keep = cost[r, c] < 0
        for i, j in zip(r[keep], c[keep]):
            edges.append((parents[i], children[j], float(prob[i, j]), float(dist[i, j])))

    return video.coords.astype(np.float32), edges


def predict_video(
    model,
    dataset,
    video_idx: int,
    device: torch.device,
    threshold: float = 0.5,
    fp16: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """``score_video`` then ``link_video``, for one video."""
    scored = score_video(model, dataset, video_idx, device, fp16=fp16)
    return link_video(dataset.videos[video_idx], scored, threshold=threshold)
