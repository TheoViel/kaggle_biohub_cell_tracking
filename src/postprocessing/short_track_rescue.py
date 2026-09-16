"""Put back the short tracks the length filter should not have deleted.

`prune_short` drops every connected component shorter than ``min_track``, and
that is the single most powerful knob in the chain -- on one stack, raising it
from 3 to 20 moved the score 0.11 with edge Jaccard *flat*, because the metric
charges for node count and the annotation is too sparse for the deleted tracks
to show up as lost edges. But length is a crude proxy for "spurious". A cell
that enters late, divides, or is lost to a detector gap produces a genuinely
short component, and the filter removes it alongside the noise.

This is the public notebooks' `ADAPTIVE_SHORT_TRACK_RESCUE`, ported from
`946-biohub-harmonic-fusion.ipynb`'s `filter_short_track_components`. It re-admits
components the length filter removed when their *quality* says they are real,
which is orthogonal evidence: a fragment of five detections linked at mean
probability 0.9 over 2.5 um steps is a cell, and a fragment of five linked at
0.5 over 8 um is not.

Four things the public version does that a naive re-admission would not, and
all four are load-bearing:

* **Only components in the band** ``[min_len, min_track)`` are candidates. A
  component of one or two nodes is not rescued at any probability -- there is
  no such thing as a confidently-linked single node.
* **A trigger.** The rescue only runs when the filter removed at least
  ``trigger_removed_frac`` of the nodes. On a stack where pruning took almost
  nothing, what it took was almost certainly noise.
* **A budget**, ``min(max_nodes_abs, max_nodes_frac * n_nodes)``, so the stage
  cannot undo the filter wholesale. v19 tightened this from 180 to 120.
* **A ranking**, ``mean_prob - 0.02 * mean_dist + 0.004 * len``, with proposals
  taken greedily until the budget is spent -- so under pressure the best
  evidence survives rather than whichever component the iteration order reached
  first.

One deliberate difference from the notebook: it keeps any component containing
a division regardless of length (``OUTPUT_KEEP_DIVISION_COMPONENTS``). That is
kept here as `keep_division_components`, because the division term is 0.1 of
the score and a forked component is exactly what the chain is trying to
produce.

This replaces `edge_pp.prune_short` in the chain rather than running after it:
the rescue needs the components and their edge statistics *before* anything is
deleted.
"""

from __future__ import annotations

import numpy as np

DEFAULTS = dict(
    # The length filter this stage wraps. Same meaning as `prune_short`'s.
    min_track=5,
    # Candidates are components of at least this many nodes. Public: 4 (946) /
    # 5 (the earlier B notebook).
    min_len=4,
    # Quality gates. Public: 0.88 and 3.0 in 946, 0.90 and 2.75 in B.
    min_mean_edge_prob=0.88,
    max_mean_edge_dist_um=3.0,
    # Budget. Public: 0.012 and 120.
    max_nodes_frac=0.012,
    max_nodes_abs=120,
    # Do not bother unless the filter actually took something substantial.
    trigger_removed_frac=0.0,
    # A component holding a fork survives the length filter outright.
    keep_division_components=1,
    enabled=1,
)


def _components(n_nodes: int, edges: np.ndarray):
    """Union-find over the edge list. -> (root per node, members per root)."""
    parent = np.arange(n_nodes)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, *_ in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n_nodes)])
    members: dict[int, list[int]] = {}
    for i, r in enumerate(roots):
        members.setdefault(int(r), []).append(i)
    return roots, members


def prune_short_with_rescue(coords, edges, cfg=None, stats=None):
    """`prune_short`, but re-admitting the short components that look real.

    ``edges`` is (M, 4) ``(src, tgt, prob, dist_um)`` -- the probability and the
    physical step length are the two quality signals, and both are already in
    the array the chain carries.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 4)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1, 4)
    n = len(coords)
    if not n or not len(edges) or cfg["min_track"] <= 1:
        return coords, edges

    roots, members = _components(n, edges)

    out_deg = np.zeros(n, dtype=np.int64)
    np.add.at(out_deg, edges[:, 0].astype(int), 1)

    by_root: dict[int, list[int]] = {}
    for k, (a, b, p, d) in enumerate(edges):
        by_root.setdefault(int(roots[int(a)]), []).append(k)

    keep: set[int] = set()
    for root, mem in members.items():
        long_enough = len(mem) >= cfg["min_track"]
        has_div = cfg["keep_division_components"] and out_deg[mem].max(initial=0) >= 2
        if long_enough or has_div:
            keep.update(mem)

    removed = n - len(keep)
    if removed <= 0 or not keep:
        # Everything survives, or nothing does -- in the second case deleting
        # the whole graph is worse than leaving it, which is the notebook's
        # `short_track_filter_skipped_all` branch.
        return coords, edges

    rescued_nodes = rescued_comps = 0
    if cfg["enabled"] and removed / max(n, 1) >= cfg["trigger_removed_frac"]:
        budget = min(int(cfg["max_nodes_abs"]),
                     max(0, int(round(n * cfg["max_nodes_frac"]))))
        proposals = []
        for root, mem in members.items():
            if keep & set(mem):
                continue
            if len(mem) < cfg["min_len"] or len(mem) >= cfg["min_track"]:
                continue
            eidx = by_root.get(root, [])
            if not eidx:
                continue                      # a lone node has no evidence
            probs = edges[eidx, 2]
            dists = edges[eidx, 3]
            mean_prob = float(np.mean(probs)) if len(probs) else 0.0
            finite = dists[np.isfinite(dists)]
            mean_dist = float(np.mean(finite)) if len(finite) else float("inf")
            if mean_prob < cfg["min_mean_edge_prob"]:
                continue
            if mean_dist > cfg["max_mean_edge_dist_um"]:
                continue
            # The notebook's ranking: confidence first, short steps second, and
            # a mild preference for longer fragments as a tie-break.
            score = mean_prob - 0.02 * mean_dist + 0.004 * len(mem)
            proposals.append((score, len(mem), mem))
        proposals.sort(key=lambda p: -p[0])
        for _, size, mem in proposals:
            if budget <= 0 or rescued_nodes + size > budget:
                continue
            keep.update(mem)
            rescued_nodes += size
            rescued_comps += 1

    if len(keep) >= n:
        if stats is not None:
            stats.update(rescued_nodes=rescued_nodes, rescued_components=rescued_comps,
                         removed_nodes=0)
        return coords, edges

    rows = np.array(sorted(keep), dtype=np.int64)
    remap = -np.ones(n, dtype=np.int64)
    remap[rows] = np.arange(len(rows))
    ok = (remap[edges[:, 0].astype(int)] >= 0) & (remap[edges[:, 1].astype(int)] >= 0)
    out_edges = edges[ok].copy()
    out_edges[:, 0] = remap[out_edges[:, 0].astype(int)]
    out_edges[:, 1] = remap[out_edges[:, 1].astype(int)]
    if stats is not None:
        stats.update(rescued_nodes=rescued_nodes, rescued_components=rescued_comps,
                     removed_nodes=int(n - len(rows)))
    return coords[rows], out_edges
