"""The target: which nodes of a post-processed graph carry a correct link.

One label per **detection** in a `main_edge_postprocess.py` output. A node is
positive when it is an endpoint of a predicted edge that matched a ground-truth
edge -- so it is both covered by the annotation *and* linked the way the
annotation says. Everything else is negative.

Two things about that definition are worth stating plainly, because they decide
how the result should be read.

**A negative is usually a real cell.** The annotation follows roughly one
lineage per stack: measured over the first six fold-0 stacks of
`edge_run/2026-09-15/0/preds_0.45_pp`, 1209 of 259428 detections are positive,
a rate of 0.466% and an imbalance of 214:1. That matches the 223:1 that
`cell_merging_cls/README.md` measures over the whole set. So this is not a
"real cell / spurious detection" label however much it is used as one, and a
model fitted to it is being asked which cells the *annotators* followed.

**They followed a region.** Annotated cells are spatially clustered, which is
what makes the question answerable from a crop at all -- and is also the reason
to distrust a good score. Distance from the frame's centroid, alone, separates
the classes at d' = 0.89, 0.67, 0.53 on three of those six stacks and d' = 3.18
on `44b6_144b256d`, where the annotated lineage sits at 13 um from the centroid
against 50 um for everything else. A crop model can score well on that without
learning anything about a cell.

The fold split is what tests whether it did: folds are whole embryos, and every
stack carries its own lineage in its own place, so a model that has only learnt
one neighbourhood cannot transfer. Read the out-of-fold AUC, not the training
one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from params import SCALE, TRAIN_DIR
from postprocessing.fast_score import GroundTruth, match_nodes

SCALE_ARR = np.asarray(SCALE, dtype=np.float64)


def load_graph(preds_dir, sample):
    """`(coords, edges)` from a post-processed npz."""
    z = np.load(Path(preds_dir) / f"{sample}.npz")
    return (z["coords"].astype(np.int64),
            np.asarray(z["edges"], np.float64).reshape(-1, 4))


def ground_truth(sample, data_dir=TRAIN_DIR):
    import tracksdata as td

    g = td.graph.IndexedRXGraph.from_geff(Path(data_dir) / f"{sample}.geff")
    return GroundTruth(g[0] if isinstance(g, tuple) else g)


def node_labels(coords, edges, gt: GroundTruth):
    """`(labels, matched)` over the rows of ``coords``.

    ``labels`` is the target: an endpoint of a true-positive edge. ``matched``
    is the weaker fact -- this detection matched *some* annotated cell -- and is
    returned only so the two can be told apart in a diagnostic, since a node can
    be annotated and still linked wrongly.
    """
    m = match_nodes(coords, gt)
    gid = np.full(len(coords), -1, np.int64)
    for row, node in m.items():
        gid[row] = node
    matched = gid >= 0

    labels = np.zeros(len(coords), bool)
    if len(edges):
        a, b = edges[:, 0].astype(int), edges[:, 1].astype(int)
        tp = np.fromiter(((int(x), int(y)) in gt.pairs
                          for x, y in zip(gid[a], gid[b])), bool, len(a))
        labels[a[tp]] = True
        labels[b[tp]] = True
    return labels, matched
