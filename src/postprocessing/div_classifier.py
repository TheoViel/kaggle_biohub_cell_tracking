"""Can a classifier beat the hand-set division gates? Leave-one-embryo-out.

    cd src/postprocessing
    python div_dataset.py            # writes the labelled candidates
    python div_classifier.py

The gates in `propose_divisions` are a conjunction of thresholds, i.e. a single
*point* in precision/recall. A classifier over the same candidates is a *curve*.
This asks whether the curve passes above the point, on labels neither has seen.

The split is by embryo, and it has to be: there are two of them, 73 positive
candidates between them, and any random split would put near-duplicate rows from
the same stack on both sides. Training on 6bba and testing on 44b6 leaves 12
positives to be judged on, which is a small number to draw a conclusion from,
and it is the honest one.

The comparison is made at *matched recall*: the gates keep some number of true
divisions, and the question is how many false ones the classifier needs to keep
the same number. Precision at the gates' own operating point is the headline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402  (chdir to src/, so --data resolves as it does
#                                  for div_dataset.py, which writes it)
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# The shipped conjunction, as a mask over the feature matrix.
GATES = dict(min_prob=0.40, max_parent_um=9.0, max_sister_um=14.0,
             symmetry_tau=0.6, diverge_um=3.0)


def gate_mask(X, f):
    """The rows `propose_divisions` would let through, at DEFAULTS."""
    i = {name: k for k, name in enumerate(f)}
    return (
        (X[:, i["prob_q"]] >= GATES["min_prob"])
        & (X[:, i["parent_dist"]] <= GATES["max_parent_um"])
        & (X[:, i["sister_dist"]] <= GATES["max_sister_um"])
        & (X[:, i["symmetry"]] <= GATES["symmetry_tau"])
        & (X[:, i["has_grandkids"]] > 0)
        & (X[:, i["diverge_gap"]] >= GATES["diverge_um"])
        & (X[:, i["is_mutual_nn"]] > 0)
    )


def pr_at_k(scores, y, k):
    """Precision and recall keeping the k highest-scored candidates."""
    if k <= 0:
        return 0.0, 0.0, 0
    idx = np.argsort(-scores)[:k]
    tp = int(y[idx].sum())
    return tp / k, tp / max(int(y.sum()), 1), tp


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../output/pp_experiments/div_dataset.npz")
    p.add_argument("--model", default="hgb", choices=("hgb", "logreg"))
    p.add_argument("--out", default="../output/pp_experiments/div_model.npz")
    args = p.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y, emb = d["X"], d["y"].astype(int), d["embryo"]
    feats = list(d["features"])
    meta, stems = d["meta"], d["stems"]
    print(f"{len(X):,} candidates, {int(y.sum())} positive, "
          f"{len(feats)} features, embryos {sorted(set(emb))}")

    keep_scores = np.full(len(X), np.nan)
    for held in sorted(set(emb)):
        tr, te = emb != held, emb == held
        if y[tr].sum() == 0 or y[te].sum() == 0:
            print(f"\n[{held}] skipped: no positives on one side")
            continue
        print(f"\n{'=' * 74}\nHELD OUT: {held}   "
              f"train {int(y[tr].sum())} pos / {int(tr.sum()):,}   "
              f"test {int(y[te].sum())} pos / {int(te.sum()):,}\n{'=' * 74}")

        if args.model == "hgb":
            # Shallow and short: 61 positives cannot support more.
            clf = HistGradientBoostingClassifier(
                max_depth=3, max_iter=150, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=1.0,
                class_weight={0: 1.0, 1: 200.0}, random_state=0,
            )
            clf.fit(X[tr], y[tr])
            s = clf.predict_proba(X[te])[:, 1]
        else:
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr]), y[tr])
            s = clf.predict_proba(sc.transform(X[te]))[:, 1]
        keep_scores[te] = s

        g = gate_mask(X[te], feats)
        g_tp = int(y[te][g].sum()); g_n = int(g.sum())
        g_prec = g_tp / max(g_n, 1)
        print(f"  hand-set gates : keeps {g_n:5d}  tp {g_tp:3d}  "
              f"precision {g_prec:.3f}  recall {g_tp / y[te].sum():.2f}")

        # the classifier, told to keep exactly as many candidates as the gates
        c_prec, c_rec, c_tp = pr_at_k(s, y[te], g_n)
        print(f"  classifier @{g_n:<5d}: keeps {g_n:5d}  tp {c_tp:3d}  "
              f"precision {c_prec:.3f}  recall {c_rec:.2f}")

        print("  classifier, other budgets:")
        for k in (25, 50, 100, 200, 400, 800):
            pr, rc, tp = pr_at_k(s, y[te], k)
            print(f"      top {k:4d}: tp {tp:3d}  precision {pr:.3f}  recall {rc:.2f}")

        if args.model == "hgb" and hasattr(clf, "feature_importances_"):
            pass  # HistGB has none; permutation importance is too slow here

    ok = ~np.isnan(keep_scores)
    if ok.any():
        np.savez_compressed(args.out, scores=keep_scores, meta=meta,
                            stems=stems, y=y, embryo=emb)
        print(f"\n-> out-of-fold scores for {int(ok.sum()):,} candidates: {args.out}")


if __name__ == "__main__":
    main()
