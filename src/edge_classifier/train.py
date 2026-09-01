"""Fit the link classifier with grouped k-fold, and report whether it is calibrated.

    python edge_classifier/train.py [--folds 5]

Folds are grouped by stack: two links from the same video share a drift, a
detector failure mode and often a cell, so splitting rows at random would let
the model memorise the video rather than the link. The user accepted the small
leak from cross-validating within one embryo instead of holding 6bba out; this
keeps at least the per-stack part of it honest.

What matters here is not accuracy — 90% of scored links are already correct, so
predicting "yes" always scores 90% — but whether the probability the model
emits *means* something, because the gate is a threshold on it. Hence the
reliability table and the Brier score alongside the ranking metrics.
"""

import argparse
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "postprocessing"))
os.chdir(SRC)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = SRC.parent / "output" / "edge_classifier"


def models(seed=0):
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight=None),
        ),
        "gbm": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
        ),
    }


def reliability(y, p, bins=(0.0, 0.5, 0.7, 0.85, 0.93, 0.97, 1.01)):
    """Does a predicted 0.7 actually come true 70% of the time."""
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append((lo, hi, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(OUT / "dataset.npz", allow_pickle=True)
    X, y, groups = data["X"], data["y"].astype(int), data["groups"]
    names = list(data["feature_names"])
    print(f"{len(X):,} labelled links over {len(set(groups))} stacks, "
          f"{100 * y.mean():.1f}% correct\n")

    splitter = GroupKFold(n_splits=args.folds)
    splits = list(splitter.split(X, y, groups))
    stack_fold = {}
    for k, (_, test) in enumerate(splits):
        for name in set(groups[test]):
            stack_fold[name] = k

    results = {}
    for label in models(args.seed):
        oof = np.zeros(len(X))
        fitted = []
        for k, (train, test) in enumerate(splits):
            model = models(args.seed)[label]
            model.fit(X[train], y[train])
            oof[test] = model.predict_proba(X[test])[:, 1]
            fitted.append(model)
        results[label] = (oof, fitted)

        # Average precision on the *negative* class: the wrong links are what
        # the gate has to find, and they are one row in ten.
        ap_wrong = average_precision_score(1 - y, 1 - oof)
        print(f"--- {label} ---")
        print(f"    AUC {roc_auc_score(y, oof):.4f}   "
              f"AP(wrong links) {ap_wrong:.4f} (base rate {1 - y.mean():.4f})   "
              f"Brier {brier_score_loss(y, oof):.4f}")
        print(f"    {'predicted':>18s}  {'n':>7s}  {'mean p':>7s}  {'actual':>7s}")
        for lo, hi, n, mean_p, actual in reliability(y, oof):
            print(f"    {f'[{lo:.2f}, {hi:.2f})':>18s}  {n:>7,d}  {mean_p:>7.3f}  {actual:>7.3f}")
        print()

    best = max(results, key=lambda k: average_precision_score(1 - y, 1 - results[k][0]))
    print(f"-> best by AP(wrong links): {best}\n")

    oof, fitted = results[best]
    final = models(args.seed)[best]
    final.fit(X, y)

    if best == "logistic":
        # The model was fitted on standardised inputs, so coef_ is already
        # "log-odds per standard deviation" and the features compare directly.
        coef = final[-1].coef_[0]
        order = np.argsort(-np.abs(coef))
        print("--- what it leans on (log-odds per standard deviation) ---")
        for i in order[:10]:
            print(f"    {names[i]:<18s} {coef[i]:+.3f}")
    else:
        imp = permutation_importance(final, X, y, n_repeats=5,
                                     random_state=args.seed, scoring="roc_auc")
        order = np.argsort(-imp.importances_mean)
        print("--- what it leans on (permutation importance, AUC) ---")
        for i in order[:10]:
            print(f"    {names[i]:<18s} {imp.importances_mean[i]:+.4f}")

    joblib.dump({"kind": best, "folds": fitted, "stack_fold": stack_fold,
                 "full": final, "feature_names": names}, OUT / "model.joblib")
    np.savez_compressed(OUT / "oof.npz", oof=oof, y=y, groups=groups)
    print(f"\n-> wrote {OUT / 'model.joblib'} and {OUT / 'oof.npz'}")


if __name__ == "__main__":
    main()
