"""Score pairs of detections, then merge the confident ones, run from src/.

Two steps, deliberately separate, because scoring is the expensive half and the
threshold is the half that has to be swept against the real metric:

    # 1. score every close pair once  (shard over GPUs; same --scores_dir)
    python main_merging_predict.py --preds_dir <preds> --run_dir <merge run> \
        --scores_dir <scores> --gpu 0 --n_shards 4 --shard 0

    # 2. apply a threshold, cheaply, as many times as you like
    python main_merging_predict.py --preds_dir <preds> --scores_dir <scores> \
        --apply --threshold 0.9 --out_dir <preds>_merged
    python main_edge_postprocess.py --preds_dir <preds>_merged

Step 2 writes a prediction folder in the same format as its input, so the
shipped post-processing chain scores it with no changes and the number that
comes out is the competition metric.

The operating point is not a modelling choice. Merging two real cells destroys
a trajectory, missing a duplicate leaves one spurious track, and `README_edge`
already priced that asymmetry from the other side. Sweep `--threshold` and read
`adj_edge_jaccard`, which carries weight 1.0 against `division_jaccard`'s 0.1.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cell_merging_cls.infer import load_fold_models, score_pairs
from cell_merging_cls.merge import apply_merges, choose_merges, connectivity
from params import EMBRYOS, TRAIN_DIR


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", required=True, help="folder of <sample>.npz")
    p.add_argument("--scores_dir", required=True,
                   help="where per-stack pair scores are written / read")
    p.add_argument("--run_dir", default=None,
                   help="training run with fold_*/ckpt_final.pt (scoring only)")
    p.add_argument("--data_dir", default=TRAIN_DIR)
    p.add_argument("--max_pair_um", type=float, default=9.0,
                   help="pairs asked about; measured duplicates top out at 8.1 um")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu", type=int, default=None, help="shorthand for --device cuda:N")
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    # apply
    p.add_argument("--apply", action="store_true", help="step 2: threshold and merge")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--delete", action="store_true",
                   help="drop the duplicate instead of re-pointing its links. The "
                        "oracle measured this at +0.0050 against merge's +0.0129; "
                        "here to reproduce that, not to be used.")
    return p.parse_args()


def embryo_of(stem):
    for e in EMBRYOS:
        if stem.startswith(e):
            return e
    return None


def do_score(args):
    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}" if args.gpu is not None else args.device
    models = load_fold_models(args.run_dir, device)

    stacks = sorted(Path(args.preds_dir).glob("*.npz"))
    if args.limit:
        stacks = stacks[:args.limit]
    stacks = stacks[args.shard::args.n_shards]
    print(f"shard {args.shard}/{args.n_shards}: {len(stacks)} stacks on {device}")

    for i, npz in enumerate(stacks):
        out = scores_dir / f"{npz.stem}.npz"
        if out.exists():
            continue
        embryo = embryo_of(npz.stem)
        if embryo not in models:
            print(f"  skip {npz.stem}: no out-of-fold model")
            continue
        model, cfg = models[embryo]
        with np.load(npz) as d:
            coords = d["coords"]
        pairs, probs = score_pairs(
            coords, Path(args.data_dir) / f"{npz.stem}.zarr", model, cfg,
            device=device, batch_size=args.batch_size, max_pair_um=args.max_pair_um)
        np.savez_compressed(out, pairs=pairs.astype(np.int32),
                            probs=probs.astype(np.float32))
        print(f"  [{i + 1}/{len(stacks)}] {npz.stem}: {len(pairs)} pairs, "
              f"{int((probs >= 0.9).sum())} over 0.9", flush=True)


def do_apply(args):
    out_dir = Path(args.out_dir or f"{args.preds_dir}_merged")
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = Path(args.scores_dir)

    n_merged = n_nodes = n_stacks = 0
    for npz in sorted(Path(args.preds_dir).glob("*.npz")):
        with np.load(npz) as d:
            coords, edges = d["coords"], d["edges"]
            cand = d["candidates"] if "candidates" in d.files else np.empty((0, 5))
        sc = scores_dir / f"{npz.stem}.npz"
        if sc.exists():
            with np.load(sc) as d:
                pairs, probs = d["pairs"], d["probs"]
            drop, onto = choose_merges(
                len(coords), pairs, probs, args.threshold,
                keep_score=connectivity(len(coords), cand))
        else:
            drop, onto = np.zeros(0, int), np.full(len(coords), -1, np.int64)
        c2, e2, k2 = apply_merges(coords, edges, cand, drop,
                                  onto=None if args.delete else onto)
        np.savez_compressed(out_dir / npz.name, coords=c2.astype(np.int16),
                            edges=e2.astype(np.float64),
                            candidates=k2.astype(np.float64))
        n_merged += len(drop)
        n_nodes += len(coords)
        n_stacks += 1

    print(f"{n_stacks} stacks -> {out_dir}")
    print(f"{'deleted' if args.delete else 'merged'} {n_merged} of {n_nodes} "
          f"detections ({100 * n_merged / max(n_nodes, 1):.3f}%) at "
          f"threshold {args.threshold}")
    (out_dir / "merge_info.json").write_text(json.dumps({
        "preds_dir": str(args.preds_dir), "scores_dir": str(args.scores_dir),
        "threshold": args.threshold, "mode": "delete" if args.delete else "merge",
        "n_merged": int(n_merged), "n_nodes": int(n_nodes),
    }, indent=2))


def main():
    args = parse_args()
    if args.apply:
        do_apply(args)
    else:
        if not args.run_dir:
            raise SystemExit("--run_dir is required for scoring")
        do_score(args)


if __name__ == "__main__":
    main()
