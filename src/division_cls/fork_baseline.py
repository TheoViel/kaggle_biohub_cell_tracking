"""A features-only baseline on the fork candidates, before spending a GPU.

The classifier is meant to rank the chain's fork proposals. Before asking
whether a CNN reading pixels can do that, ask whether the features the fork was
*proposed* with already can -- parent/sister distance, the two edge
probabilities, the size of the split. If they can, the CNN has to beat them to
be worth anything; if they cannot, that is the evidence the appearance model is
needed, which is the same argument `README_edge.md` makes for the original
classifier ("precision is the constraint and no re-weighting of existing
features supplies it").

Leave-one-embryo-out, same split as everywhere else, reporting AUC and AP
against the positive rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from division_cls.gen_fork_candidates import COLUMNS  # noqa: E402
from params import EMBRYOS  # noqa: E402


def load(path, labelled_only=True):
    blob = np.load(path, allow_pickle=True)
    stems, table = blob["stem"], blob["table"]
    if labelled_only:
        keep = table[:, 4] >= 0
        stems, table = stems[keep], table[keep]
    return np.asarray([str(s) for s in stems]), table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True)
    p.add_argument("--score_out", default=None,
                   help="write out-of-fold scores for EVERY fork, in the layout "
                        "`score_forks.py` writes, so `tune_fork_threshold.py` can "
                        "read either interchangeably")
    args = p.parse_args()

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    stems, table = load(args.candidates)
    y = table[:, 4].astype(int)
    X = table[:, 5:].astype(np.float64)
    names = list(COLUMNS[5:])
    embryo = np.asarray([s[:4] for s in stems])

    print(f"{len(y)} labelled forks, {y.sum()} positive "
          f"({y.mean():.3%}), {X.shape[1]} features: {names}\n")
    for e in EMBRYOS:
        m = embryo == e
        print(f"  {e}: {m.sum():5d} forks, {y[m].sum():3d} positive ({y[m].mean():.3%})")
    print()

    print(f"{'fold':<8}{'val':<8}{'model':<22}{'auc':>8}{'ap':>8}{'chance':>9}")
    for fold, val in enumerate(EMBRYOS):
        tr, va = embryo != val, embryo == val
        if not tr.any() or not va.any() or y[va].sum() == 0:
            print(f"fold {fold}: unusable split")
            continue
        scaler = StandardScaler().fit(X[tr])
        for name, clf in [
            ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("gradient boosting", GradientBoostingClassifier(random_state=0)),
        ]:
            clf.fit(scaler.transform(X[tr]), y[tr])
            s = clf.predict_proba(scaler.transform(X[va]))[:, 1]
            print(f"{fold:<8}{val:<8}{name:<22}{roc_auc_score(y[va], s):>8.3f}"
                  f"{average_precision_score(y[va], s):>8.3f}{y[va].mean():>9.3f}")
        # Each feature alone, signed so the better direction is reported: with
        # ~20 positives a single-feature AUC is the honest yardstick for whether
        # the multi-feature fit is fitting anything.
        for j, nm in enumerate(names):
            a = roc_auc_score(y[va], X[va, j])
            print(f"{'':<8}{'':<8}{'  ' + nm:<22}{max(a, 1 - a):>8.3f}"
                  f"{'':>8}{'  (flip)' if a < 0.5 else '':>9}")

    if args.score_out:
        # Every fork, scored by the fold that held its embryo out -- the same
        # discipline the CNN path enforces, and necessary here too: the model is
        # fitted on labelled forks but applied to all 107k, and a stack must not
        # be scored by a fit that saw it.
        all_stems, all_table = load(args.candidates, labelled_only=False)
        all_embryo = np.asarray([s[:4] for s in all_stems])
        scores = np.full(len(all_table), np.nan, np.float32)
        for val in EMBRYOS:
            tr = embryo != val
            if not tr.any():
                continue
            scaler = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(scaler.transform(X[tr]), y[tr])
            m = all_embryo == val
            scores[m] = clf.predict_proba(
                scaler.transform(all_table[m, 5:].astype(np.float64)))[:, 1]
        np.savez_compressed(
            args.score_out, stem=all_stems, score=scores,
            label=all_table[:, 4].astype(np.int8),
            coords=all_table[:, :4].astype(np.float32),
            features=all_table[:, 5:].astype(np.float32))
        print(f"\nwrote {args.score_out}: {np.isfinite(scores).sum()}/{len(scores)} scored")


if __name__ == "__main__":
    main()
