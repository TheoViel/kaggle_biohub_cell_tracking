"""The classifier as a drop-in replacement for the pipeline's third step.

The gate the sweeps settled on — a cut on the assignment margin — turned out to
remove roughly one true positive for every false positive it removed, where the
metric needs about 1.23 to break even. That is a ranking failure, not a
threshold failure, which is what a fitted model is for.

``classifier_postprocess`` is ``postprocessing.pipeline.postprocess`` with that
step swapped: re-link, prune short tracks, then keep a link only when the model
puts it above ``threshold``.
"""

from __future__ import annotations

import numpy as np

from edge_classifier.features import link_features
from edge_classifier.prepare import prepare

# Where the metric's own arithmetic puts the decision.
#
#   J = TP / (n_gt + FP).  Keeping a link contributes one TP with probability p
#   and one FP with probability 1 - p, so keeping beats dropping when
#
#       p / (n_gt + FP)  >  (1 - p) * TP / (n_gt + FP)^2
#   <=> p  >  TP / (TP + n_gt + FP)
#
# At the pipeline's operating point across the 71 validation stacks
# (TP = 17875, FP = 2211, n_gt = 19826) that is 0.45.
#
# An earlier note in postprocessing/README.md put this at 0.54, later 0.62, by
# charging a wrong decision two false positives. That conflated two different
# things: a ground-truth link we get wrong does show up as two false-positive
# *edges* — the parent linked elsewhere and the child linked from elsewhere —
# but each of those is a separate keep-or-drop decision worth one FP, not two.
BREAK_EVEN = 0.45

# What the sweep actually prefers (evaluate.py). Lower than break-even because
# the model's ranking inside [0.30, 0.45) is weaker than its calibration
# suggests: those links come true about half the time, not a third.
DEFAULT_THRESHOLD = 0.30


def link_probability(coords, npz_edges, linked, model):
    """P(this link reproduces a ground-truth link), per link."""
    if len(linked) == 0:
        return np.empty(0)
    X = link_features(coords, npz_edges, linked)
    return model.predict_proba(X)[:, 1]


def classifier_postprocess(coords, edges, model, threshold=DEFAULT_THRESHOLD, **kw):
    """Raw tracker output -> (coords, edges), gated by the classifier."""
    coords, linked, npz_edges = prepare(coords, edges, **kw)
    if len(linked) == 0:
        return coords, np.empty((0, 4))

    p = link_probability(coords, npz_edges, linked, model)
    return coords, linked[p >= threshold][:, :4]
