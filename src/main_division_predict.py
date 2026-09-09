"""Score every prediction folder's nodes with the division classifier.

    python main_division_predict.py --preds_dir ../logs/edge_run/2026-09-08/13/preds_0.50 \
                                    --run_dir ../logs/division/2026-09-09/2

Writes `<preds_dir>_divprob/<sample>.npz` holding `div_prob`, one row per node
of the prediction npz, `nan` where nothing was scored. The prediction files are
left alone: the probability is an annotation *of* them, and keeping it beside
them means a post-processing sweep can pick it up or ignore it without either
of them being regenerated.

Only the nodes `propose_divisions` could ask about are scored -- a parent with
one committed child and an unclaimed neighbour within `max_parent_um`. That is
a few percent of the graph, and the rest is unscoreable by construction.

**The fold split is enforced, not assumed.** Fold `i` trained without
`EMBRYOS[i]`, so each stack is scored by the model that held its embryo out. A
stack whose embryo has no such model is skipped with a warning rather than
scored by a model that has seen it.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

from division_cls.infer import candidate_parents, load_fold_models, score_stack
from params import LOG_FOLDER, TRAIN_DIR

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", required=True,
                   help="the npz folder main_edge_predict.py wrote")
    p.add_argument("--run_dir", required=True,
                   help="a division_cls run holding fold_<n>/ckpt_final.pt")
    p.add_argument("--out_dir", default=None, help="default <preds_dir>_divprob")
    p.add_argument("--data_dir", default=TRAIN_DIR)
    p.add_argument("--max_parent_um", type=float, default=9.0,
                   help="a parent with no unclaimed node this close is never a "
                        "division candidate, so it is not scored")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    preds_dir = Path(args.preds_dir)
    out_dir = Path(args.out_dir) if args.out_dir else preds_dir.with_name(
        preds_dir.name + "_divprob")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_fold_models(args.run_dir, args.device)
    print(f"-> models   : {args.run_dir}  ({', '.join(sorted(models))})")
    print(f"-> reading  : {preds_dir}")
    print(f"-> writing  : {out_dir}\n")

    paths = sorted(preds_dir.glob("*.npz"))[: args.limit]
    n_scored = n_nodes = 0
    for i, path in enumerate(paths):
        stem = path.stem
        embryo = stem.split("_")[0]
        if embryo not in models:
            print(f"[{i + 1:>3d}/{len(paths)}] {stem}  SKIPPED: no out-of-fold model")
            continue
        model, cfg = models[embryo]

        with np.load(path) as d:
            coords = d["coords"].astype(np.float64)
            edges = d["edges"].astype(np.float64).reshape(-1, 4)
        rows = candidate_parents(coords, edges, max_parent_um=args.max_parent_um)
        prob = score_stack(coords, Path(args.data_dir) / f"{stem}.zarr", model, cfg,
                           device=args.device, batch_size=args.batch_size, rows=rows)

        np.savez_compressed(out_dir / f"{stem}.npz", div_prob=prob.astype(np.float32))
        got = np.isfinite(prob)
        n_scored += int(got.sum())
        n_nodes += len(coords)
        top = np.nanmax(prob) if got.any() else float("nan")
        print(f"[{i + 1:>3d}/{len(paths)}] {stem:<16s} {int(got.sum()):>6d}/{len(coords):<6d} "
              f"candidates scored   max p={top:.3f}", flush=True)

    print(f"\n-> {n_scored:,} of {n_nodes:,} nodes scored "
          f"({n_scored / max(n_nodes, 1):.1%}) into {out_dir}")


if __name__ == "__main__":
    main()
