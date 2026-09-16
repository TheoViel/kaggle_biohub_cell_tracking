"""Score every fork candidate with the out-of-fold division classifier.

Reads the table `gen_fork_candidates.py` wrote and returns one probability per
row, aligned to it. Unlike training, **every** row is scored including the
unlabelled ones: the threshold has to be applied to every fork the chain
proposes, and only ~0.7% of them sit near an annotated cell.

The fold split is enforced rather than assumed, as in `main_division_predict`:
fold `i` trained without `EMBRYOS[i]`, so a stack is scored by the model that
held its embryo out, and a stack with no such model is skipped rather than
scored by a model that has seen it.

Parents are deduplicated before scoring -- a three-way split contributes two
rows with the same parent -- and the scores are scattered back over the rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from division_cls.infer import load_fold_models, score_stack  # noqa: E402
from params import TRAIN_DIR  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True, help="gen_fork_candidates .npz")
    p.add_argument("--run_dir", required=True,
                   help="a division_cls run holding fold_<n>/ckpt_final.pt")
    p.add_argument("--data_dir", default=TRAIN_DIR)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=512)
    args = p.parse_args()

    blob = np.load(args.candidates, allow_pickle=True)
    stems = np.asarray([str(s) for s in blob["stem"]])
    table = blob["table"]
    models = load_fold_models(args.run_dir, args.device)
    print(f"{len(table)} forks, models for {sorted(models)}", flush=True)

    scores = np.full(len(table), np.nan, np.float32)
    order = sorted(set(stems.tolist()))
    for i, stem in enumerate(order, 1):
        embryo = stem[:4]
        if embryo not in models:
            print(f"  skip {stem}: no out-of-fold model for {embryo}", flush=True)
            continue
        model, cfg = models[embryo]
        rows = np.flatnonzero(stems == stem)
        coords = table[rows, :4].astype(np.float64)
        uniq, inv = np.unique(coords, axis=0, return_inverse=True)
        s = score_stack(uniq, Path(args.data_dir) / f"{stem}.zarr", model, cfg,
                        device=args.device, batch_size=args.batch_size)
        scores[rows] = np.asarray(s, np.float32)[inv]
        if i % 20 == 0 or i == len(order):
            done = int(np.isfinite(scores).sum())
            print(f"  {i}/{len(order)} stacks, {done} forks scored", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, stem=stems, score=scores, label=table[:, 4].astype(np.int8),
                        coords=table[:, :4].astype(np.float32),
                        features=table[:, 5:].astype(np.float32))
    lab = table[:, 4].astype(int)
    ok = np.isfinite(scores)
    print(f"\nwrote {out}  ({ok.sum()}/{len(scores)} scored)")
    for name, m in (("positive", lab == 1), ("negative", lab == 0),
                    ("unlabelled", lab == -1)):
        m = m & ok
        if m.any():
            print(f"  {name:<11} n={m.sum():6d}  score mean {scores[m].mean():.3f} "
                  f"median {np.median(scores[m]):.3f}")


if __name__ == "__main__":
    main()
