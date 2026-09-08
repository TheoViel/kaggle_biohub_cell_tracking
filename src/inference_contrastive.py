"""
Extract NETr embeddings for every detection of a predictions folder.

Takes a training run (``--exp_folder``, holding ``args.json`` and
``fold_<n>/ckpt_final.pt``) and a tracking-pipeline predictions folder
(``--coords_log_folder``, holding ``<sample>.npz`` with a ``coords`` array of
``(t, z, y, x)`` rows), and writes one embedding per detected cell.

The embeddings are written **back into the npz they were read from**, under a
key naming the run that produced them: ``../logs/2026-09-03/13/`` becomes
``embeddings_2026-09-03_13``. One detections file therefore accumulates the
embeddings of every run scored against it, side by side and directly
comparable, instead of each run copying the coords into a folder of its own.

The array is (N, D) and **row-aligned** with that file's own ``coords``: row i
of the embeddings belongs to row i of the coords. That is the form
``embeddings.match.link_sequence`` reads once the rows are grouped by frame.

An npz cannot be appended to, so each write rewrites the whole file, through a
temporary file that is renamed into place -- the detections are the input to
everything downstream and a half-written one would be worse than no embedding.

Which model embeds which sample matters, because a model that trained on a cell
will embed it better than one that did not, and an embedding set mixing the two
cannot be compared against anything. So the default is out-of-fold, exactly as
``main_predict`` does it: fold ``f`` embeds the samples of the embryo it held
out, and the folds together cover every sample once. ``--fold`` overrides that
and runs a single model over everything, which is what you want for a fullfit
model or a quick look, and is leaky if the model trained on those embryos.

Consumers still read a plain ``embeddings`` key (``edge_transformer.data``,
``tracking.dataset``, ``postprocessing.edge_pp``), so they need to be pointed at
the new per-run key before they can use what this writes.

``--gt`` additionally embeds the *annotated* nodes of each sample, into
``gt_coords`` (M, 4) and ``<key>_gt`` (M, D), row-aligned with each other and
kept apart from the detections so that ``coords``/``<key>`` stay exactly what
they were. They are the input to ``main_edge_predict.py --gt``, which appends
the ones no detection reached, lifting node recall to 1 so the edge model can be
read on its linking alone. The annotation is sparse -- tens of cells a video
against tens of thousands of detections -- so this costs almost nothing on top.

Single GPU:
    python inference_contrastive.py --exp_folder ../logs/2026-09-03/0/ \\
        --coords_log_folder ../logs/2026-09-02/9/preds_0.50/

Multi-GPU (one process per GPU, samples sharded across ranks):
    torchrun --standalone --nproc_per_node=4 inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-03/13/ --coords_log_folder ../logs/2026-09-07/4/preds_r6_0.40/
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from embeddings.dataset import find_coord_files
from embeddings.infer import embed_sample, load_netr
from embeddings.links import gt_node_coords
from params import EMBRYOS, LOG_FOLDER, TRAIN_DIR
from training.distributed import cleanup_distributed, init_distributed


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--exp_folder", required=True,
                   help="training run holding args.json and fold_<n>/ckpt_final.pt")
    p.add_argument("--coords_log_folder", required=True,
                   help="predictions folder with <sample>.npz holding `coords`")
    p.add_argument("--data_dir", default=TRAIN_DIR,
                   help="where the matching <sample>.zarr volumes live")
    p.add_argument("--key", default=None,
                   help="npz key to write the embeddings under; default is "
                        "embeddings_<exp_folder>, e.g. embeddings_2026-09-03_13")
    p.add_argument("--fold", default=None,
                   help="run ONE model over every sample: a fold number, or "
                        "'fullfit'. Default is out-of-fold across the run's "
                        "selected_folds.")
    p.add_argument("--ckpt", default=None,
                   help="explicit checkpoint; implies a single model over "
                        "every sample, as --fold does")
    p.add_argument("--samples", default=None,
                   help="comma-separated sample-name prefixes to restrict to")
    p.add_argument("--max_frames", type=int, default=None,
                   help="cap frames per sample (debugging)")
    p.add_argument("--n_samples", type=int, default=None,
                   help="cap the number of samples (debugging)")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--emb_dtype", choices=("float32", "float16"), default="float32",
                   help="stored precision. The embeddings are L2-normalised, so "
                        "float16 loses little and halves the files")
    p.add_argument("--overwrite", action="store_true",
                   help="re-embed samples whose npz already exists")
    p.add_argument("--gt", action="store_true",
                   help="also embed the ground-truth nodes, into `gt_coords` "
                        "and `<key>_gt` beside the detections. Feeds "
                        "`main_edge_predict.py --gt`, which appends the "
                        "unmatched ones to reach a node recall of 1")
    return p.parse_args()


def _ckpt_for(exp_folder: Path, fold) -> Path:
    """``fold_<n>/ckpt_final.pt``, or ``fold_fullfit`` for the fullfit model."""
    name = "fold_fullfit" if str(fold) == "fullfit" else f"fold_{int(fold)}"
    return exp_folder / name / "ckpt_final.pt"


def embeddings_key(exp_folder: str | Path, log_folder: str = LOG_FOLDER) -> str:
    """``../logs/2026-09-03/13/`` -> ``embeddings_2026-09-03_13``.

    The run folder identifies the weights, so the key is the run folder: two
    runs embedding the same detections land in the same file without colliding.
    """
    text = str(exp_folder).rstrip("/")
    prefix = str(log_folder).rstrip("/") + "/"
    if text.startswith(prefix):
        text = text[len(prefix):]
    return "embeddings_" + text.strip("/").replace("/", "_")


def write_embeddings(path: Path, new: dict[str, np.ndarray]) -> None:
    """Add ``new``'s arrays to ``path``'s npz, keeping every array already in it.

    npz has no append, so the file is rebuilt. It is rebuilt beside itself and
    renamed, because ``path`` is a detections file the whole pipeline reads: an
    interrupted in-place write would destroy the coords as well as the
    embeddings, and rename is atomic on POSIX. ``--gt`` adds two arrays at once
    for that reason: one rewrite, and either both land or neither does.
    """
    with np.load(path) as d:
        arrays = {k: d[k] for k in d.files}
    arrays.update(new)

    tmp = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _plan(args, exp_folder: Path, files: list[Path], is_main: bool):
    """Which checkpoint embeds which samples: [(ckpt, [paths]), ...].

    A single ``--fold`` or ``--ckpt`` takes every sample. Otherwise each fold
    takes its own held-out embryo, so the union is every sample exactly once and
    no embedding comes from a model that trained on that cell.
    """
    if args.ckpt or args.fold is not None:
        ckpt = Path(args.ckpt) if args.ckpt else _ckpt_for(exp_folder, args.fold)
        return [(ckpt, files)]

    cfg = json.load(open(exp_folder / "args.json"))
    folds = [int(f) for f in (cfg.get("selected_folds") or [0])]

    plan = []
    for fold in folds:
        embryo = EMBRYOS[fold]
        mine = [f for f in files if f.stem.startswith(embryo)]
        if mine:
            plan.append((_ckpt_for(exp_folder, fold), mine))
        elif is_main:
            print(f"!  fold {fold}: no sample of {embryo} in the coords folder")

    covered = {f for _, group in plan for f in group}
    missed = [f.stem for f in files if f not in covered]
    if missed and is_main:
        # Not fatal, but silence here would look like the samples were embedded.
        print(f"!  {len(missed)} sample(s) belong to no selected fold and are "
              f"skipped: {', '.join(missed[:5])}"
              f"{' ...' if len(missed) > 5 else ''}")
        print("   pass --fold to run one model over every sample")
    return plan


def main():
    args = parse_args()
    ctx = init_distributed(None)
    device = ctx.device

    exp_folder = Path(args.exp_folder.rstrip("/"))
    key = args.key or embeddings_key(args.exp_folder)

    files = find_coord_files(args.coords_log_folder)
    if args.samples:
        wanted = [s.strip() for s in args.samples.split(",") if s.strip()]
        files = [f for f in files if any(f.stem.startswith(s) for s in wanted)]
        if not files:
            raise ValueError(f"no sample of {wanted} in {args.coords_log_folder}")
    if args.n_samples is not None:
        files = files[: args.n_samples]

    plan = _plan(args, exp_folder, files, ctx.is_main)
    if not plan:
        raise FileNotFoundError("nothing to embed")

    missing = [str(c) for c, _ in plan if not c.exists()]
    if missing:
        # Checked up front: finding out on the second checkpoint would leave a
        # half-written embedding folder that looks complete.
        raise FileNotFoundError("missing checkpoint(s): " + ", ".join(missing))

    data_dir = Path(args.data_dir)
    if ctx.is_main:
        print(f"-> exp_folder     : {exp_folder}")
        print(f"-> detections     : {args.coords_log_folder}  ({len(files)} samples)")
        print(f"-> writing key    : {key}  (into each detections npz)")
        if args.gt:
            print(f"-> ground truth   : also embedding the annotated nodes into "
                  f"gt_coords / {key}_gt")
        print(f"-> precision      : {'FP16' if args.fp16 else 'FP32'} compute, "
              f"{args.emb_dtype} stored")
        if ctx.distributed:
            print(f"-> distributed    : {ctx.world_size} processes")
        print()

    n_done = n_nodes = 0
    start = time.time()

    for ckpt, group in plan:
        model, cfg = load_netr(ckpt, device=device, verbose=ctx.is_main)
        patch_zyx = tuple(cfg.get("patch_zyx") or (3, 64, 64))
        n_frames = int(cfg.get("n_frames", 1) or 1)

        # Shard samples across ranks; each writes its own npz files, so there is
        # nothing to gather afterwards.
        gt_key = f"{key}_gt"
        # --gt adds two more arrays to write, so a file already carrying the
        # detections' embeddings is only complete if it carries those too.
        wanted = [key] + ([gt_key, "gt_coords"] if args.gt else [])

        mine = group[ctx.rank :: ctx.world_size]
        for i, path in enumerate(mine):
            with np.load(path) as d:
                have = [k for k in wanted if k in d.files]
                if len(have) == len(wanted) and not args.overwrite:
                    print(f"[rank {ctx.rank}] {i + 1:03d}/{len(mine):03d}  "
                          f"{path.stem}  -  has {', '.join(wanted)}, skipped",
                          flush=True)
                    continue
                coords = np.asarray(d["coords"])
            if coords.size == 0:
                print(f"[rank {ctx.rank}] {i + 1:03d}/{len(mine):03d}  "
                      f"{path.stem}  -  no detections", flush=True)
                continue

            import zarr
            volume = zarr.open_group(str(data_dir / f"{path.stem}.zarr"), mode="r")["0"]

            t0 = time.time()
            kept, emb = embed_sample(
                model, volume, coords, patch_zyx, device,
                n_frames=n_frames, fp16=args.fp16, max_frames=args.max_frames,
            )
            # The key sits beside the file's own coords, so it must have a row
            # for each of them. --max_frames embeds a prefix of the frames and
            # would write a shorter array whose rows no longer line up.
            if len(kept) != len(coords):
                raise ValueError(
                    f"{path.stem}: embedded {len(kept)} of {len(coords)} detections "
                    "-- writing that into the coords npz would break the row "
                    "alignment every consumer assumes. Drop --max_frames."
                )
            new = {key: emb.astype(args.emb_dtype)}
            gt_note = ""
            if args.gt:
                # The annotated cells, embedded by the same model through the
                # same patches -- so a GT node appended to the detections
                # downstream is in the same feature space as its neighbours and
                # not a vector the edge model has never seen the likes of.
                # A sample with no annotation still gets the keys, empty, so
                # "written" and "no ground truth to write" stay distinguishable
                # from "this file predates --gt".
                gt_coords = (
                    gt_node_coords(data_dir / path.stem)
                    if (data_dir / f"{path.stem}.geff").exists()
                    else np.zeros((0, 4), np.float32)
                )
                if len(gt_coords):
                    _, gt_emb = embed_sample(
                        model, volume, gt_coords, patch_zyx, device,
                        n_frames=n_frames, fp16=args.fp16,
                    )
                else:
                    gt_emb = np.zeros((0, emb.shape[1]), np.float32)
                new["gt_coords"] = gt_coords
                new[gt_key] = gt_emb.astype(args.emb_dtype)
                gt_note = f" + {len(gt_coords)} GT"

            write_embeddings(path, new)
            n_done += 1
            n_nodes += len(kept)
            print(f"[rank {ctx.rank}] {i + 1:03d}/{len(mine):03d}  {path.stem}  -  "
                  f"{len(kept)} nodes{gt_note} over "
                  f"{len(np.unique(kept[:, 0].astype(int)))} "
                  f"frames -> {emb.shape[1]}d  ({time.time() - t0:.1f}s)", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if ctx.distributed:
        totals = torch.tensor([n_done, n_nodes], device=device, dtype=torch.float64)
        dist.all_reduce(totals)
        n_done, n_nodes = int(totals[0]), int(totals[1])

    if ctx.is_main:
        print(f"\n-> embedded {n_nodes} nodes over {n_done} samples "
              f"in {time.time() - start:.0f}s")
        print(f"-> wrote {key} into {args.coords_log_folder}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
