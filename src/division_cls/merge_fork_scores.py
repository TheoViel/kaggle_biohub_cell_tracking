"""Combine several fork-score files into one, for `tune_fork_threshold.py`.

The graph metric is ``div_J = tp / (tp + fp + fn)``, and with ``fn = N_gt - tp``
that is ``tp / (N_gt + fp)`` -- a true division is worth exactly as much as a
false one costs. The wave-1 models sit at very different points on that trade:

    fix_c48nf   48 tp   69 fp   0.2474     recall
    f5          45 tp   52 fp   0.2542     balanced
    fix_c48f5   38 tp   30 fp   0.2452     precision

They are not better and worse versions of each other, they disagree. A model
with the first's recall and the third's precision would score ~45/160 = 0.28,
well past anything measured, which is the case for ensembling rather than
picking a winner.

`rank` averaging by default: the models are calibrated differently (one
saturates at 0/1, another does not), so averaging probabilities lets the
confident-but-wrong model dominate. Ranks only carry the ordering, which is all
the threshold sweep uses. Ranks are computed per stack, since the sweep applies
one threshold everywhere and a stack with many forks would otherwise dominate
the scale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("rank", "prob", "min", "max"), default="rank")
    p.add_argument("--weights", nargs="*", type=float, default=None)
    args = p.parse_args()

    blobs = [np.load(s, allow_pickle=True) for s in args.scores]
    ref = blobs[0]
    stems = np.asarray([str(s) for s in ref["stem"]])
    for b, name in zip(blobs[1:], args.scores[1:]):
        if len(b["score"]) != len(ref["score"]):
            raise SystemExit(f"{name}: {len(b['score'])} rows vs {len(ref['score'])}")
        if not np.array_equal(b["coords"], ref["coords"]):
            raise SystemExit(f"{name}: coords differ -- the files describe different forks")

    w = np.asarray(args.weights if args.weights else [1.0] * len(blobs), float)
    if len(w) != len(blobs):
        raise SystemExit(f"{len(w)} weights for {len(blobs)} score files")
    w = w / w.sum()

    mats = []
    for b in blobs:
        s = np.asarray(b["score"], float)
        if args.mode == "rank":
            r = np.full(len(s), np.nan)
            for stem in set(stems.tolist()):
                m = stems == stem
                v = s[m]
                ok = np.isfinite(v)
                if not ok.any():
                    continue
                # `argsort(argsort(v))` IS the rank vector. Scattering it --
                # `order[argsort(argsort(v))] = arange` -- returns the sorting
                # indices instead, which is a different permutation entirely and
                # silently destroys the ordering the ensemble exists to combine.
                rr = np.full(len(v), np.nan)
                rr[ok] = np.argsort(np.argsort(v[ok])) / max(ok.sum() - 1, 1)
                r[m] = rr
            mats.append(r)
        else:
            mats.append(s)

    stack = np.vstack(mats)
    if args.mode in ("rank", "prob"):
        out = np.nansum(stack * w[:, None], axis=0)
        out[np.all(~np.isfinite(stack), axis=0)] = np.nan
    elif args.mode == "min":
        out = np.nanmin(stack, axis=0)
    else:
        out = np.nanmax(stack, axis=0)

    lab = np.asarray(ref["label"], int)
    for name, b in zip(args.scores, blobs):
        s = np.asarray(b["score"], float)
        m = (lab >= 0) & np.isfinite(s)
        print(f"  {Path(name).stem:<22} pos median {np.median(s[m & (lab == 1)]):.3f}  "
              f"neg median {np.median(s[m & (lab == 0)]):.3f}")
    np.savez_compressed(args.out, stem=ref["stem"], score=out.astype(np.float32),
                        label=ref["label"], coords=ref["coords"],
                        features=ref["features"])
    print(f"\n{args.mode} ensemble of {len(blobs)} -> {args.out} "
          f"({int(np.isfinite(out).sum())}/{len(out)} scored)")


if __name__ == "__main__":
    main()
