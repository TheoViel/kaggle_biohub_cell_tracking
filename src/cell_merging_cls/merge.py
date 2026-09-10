"""Turning pair decisions into a graph: which node survives, and what moves onto it.

**Merge, never delete.** The oracle measured both: deleting the surplus
detection is worth +0.0050 cv, re-pointing its links onto the survivor is worth
+0.0129. The gap is the fork bucket -- in 34 of the 151 annotated divisions the
duplicate is the copy that carries the *second daughter*, so deleting it deletes
the division, while merging keeps the link and lets the survivor fork. Deleting
also costs node recall (-0.0045 against -0.0005), because a shortened track
falls under `prune_short` and takes annotated nodes with it.

So a merge here is: drop the duplicate's row, and rewrite every candidate and
edge index that pointed at it to point at the survivor instead.
"""

from __future__ import annotations

import numpy as np


def choose_merges(n_nodes, pairs, probs, threshold, keep_score=None):
    """`(drop, onto)` from scored pairs, taking the confident ones first.

    Greedy in descending probability, and each node participates in at most one
    merge: a chain of near-threshold merges could otherwise collapse a row of
    genuinely distinct cells into one, which is the failure mode that makes a
    distance rule fire 10 508 times.

    `keep_score` (higher = keep) decides which member survives. At inference the
    caller passes how well connected each node is, so the merge keeps the copy
    the assignment was already using and moves the weaker one onto it.
    """
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    probs = np.asarray(probs, float).reshape(-1)
    onto = np.full(n_nodes, -1, np.int64)
    if len(pairs) == 0:
        return np.zeros(0, int), onto

    fire = probs >= threshold
    pairs, probs = pairs[fire], probs[fire]
    order = np.argsort(-probs)

    spent = np.zeros(n_nodes, bool)
    drop = []
    for k in order:
        a, b = int(pairs[k, 0]), int(pairs[k, 1])
        if spent[a] or spent[b]:
            continue
        if keep_score is not None and keep_score[b] > keep_score[a]:
            a, b = b, a            # a survives, b is merged onto it
        onto[b] = a
        spent[a] = spent[b] = True
        drop.append(b)
    return np.array(sorted(drop), dtype=int), onto


def connectivity(n_nodes, cand):
    """How well linked each node is: the sum of its candidate probabilities.

    Used to pick the survivor. The copy the assignment can actually use is the
    one to keep -- the other is the one whose links have to move.
    """
    cand = np.asarray(cand, float).reshape(-1, 5)
    out = np.zeros(n_nodes, float)
    if len(cand) == 0:
        return out
    np.add.at(out, cand[:, 0].astype(int), cand[:, 2])
    np.add.at(out, cand[:, 1].astype(int), cand[:, 2])
    return out


def apply_merges(coords, edges, cand, drop, onto=None):
    """Remove the dropped rows and renumber everything that referred to them.

    With `onto`, a dropped row's links are re-pointed onto its survivor; without
    it they are discarded, which is the delete variant the oracle measured and
    rejected.
    """
    coords = np.asarray(coords)
    keep = np.ones(len(coords), bool)
    keep[drop] = False
    remap = np.full(len(coords), -1, np.int64)
    remap[keep] = np.arange(keep.sum())
    if onto is not None:
        for dup in drop:
            s = int(onto[dup])
            if s < 0:
                continue
            while not keep[s]:      # resolve a chain down to a surviving row
                s = int(onto[s])
            remap[dup] = remap[s]

    def _filter(arr, key_cols):
        arr = np.asarray(arr, float)
        if len(arr) == 0:
            return arr
        s, t = remap[arr[:, 0].astype(int)], remap[arr[:, 1].astype(int)]
        # s == t is a link that became a self-loop when both ends merged
        ok = (s >= 0) & (t >= 0) & (s != t)
        out = arr[ok].copy()
        out[:, 0], out[:, 1] = s[ok], t[ok]
        if onto is None or len(out) == 0:
            return out
        # Re-pointing can collide two links onto one key; keep the most
        # confident, which is what the assignment would have seen.
        order = np.lexsort((-out[:, 2], *[out[:, c] for c in reversed(key_cols)]))
        out = out[order]
        _, idx = np.unique(out[:, key_cols], axis=0, return_index=True)
        return out[np.sort(idx)]

    return (coords[keep],
            _filter(edges, [0, 1]),
            _filter(cand, [0, 1, 4]))
