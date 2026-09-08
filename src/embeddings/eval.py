"""Validation: does a cell's embedding retrieve its own next frame?

The training objective's top-1 is measured against the *augmentation's* own
pairing -- two views of one frame -- which says the embeddings are stable under
the transforms they were trained on and nothing about whether they follow a
cell through time. This is the number that matters, and it needs ground truth:

    embed frame t and frame t+1, take the cell at t, rank every cell at t+1 by
    cosine similarity, and count a hit when the top-ranked one is the cell the
    ground-truth graph links it to.

Two decisions in that sentence are worth stating, because they set how hard the
task is.

**The candidates are every detection at t+1, not just the annotated ones.** The
real ground truth annotates ~3% of cells, so scoring against only those would
rank a cell against five or six alternatives -- trivial, and nothing like what
the matcher faces. Ranking against the full detection set is the same decision
inference makes.

**The embeddings are computed over the full frame too.** NETr is contextual:
a cell's embedding depends on the cells around it through the transformer.
Embedding only the annotated handful would compute a different function than
the one that runs at inference.

Reported alongside it is a **nearest-neighbour baseline**: link each cell to the
spatially closest detection at t+1 and count hits the same way. No model, no
appearance -- just geometry. It is the number the embeddings have to beat to
have learned anything, and it is not a low bar here: cells move ~1.86 um a frame
against ~8 um of spacing, so position alone is very nearly sufficient. A model
scoring at or below it has found a more expensive way to compute proximity.

Ground-truth nodes are sparse and are not themselves detections, so each one is
attached to the nearest detection within ``match_um``. An annotation with no
detection near it is *not* counted as a miss -- it is a detection failure, and
folding it in here would measure the detector rather than the embeddings. It is
reported separately as coverage, which is the number that bounds what this
metric can ever be worth.
"""

from __future__ import annotations

import numpy as np
import torch

from embeddings.infer import embed_frame
from embeddings.links import MATCH_UM, attach as _attach, gt_frames as _gt_frames
from params import SCALE
from tracking.main import _paths_for_embryos
from training.distributed import unwrap_model


def _embed_frame(model, volume, coords, patch_zyx, device, n_frames=1) -> np.ndarray:
    """Embeddings for every detection in one frame, no augmentation.

    Shares ``embeddings.infer.embed_frame`` with the inference entrypoint, so
    the number reported during training is produced by the same code path that
    writes the embeddings afterwards. fp16 is off: this runs interleaved with
    training, where the autocast context is the caller's business.
    """
    return embed_frame(model, volume, coords, patch_zyx, device,
                       n_frames=n_frames, fp16=False)


@torch.no_grad()
def evaluate_retrieval(
    model,
    args,
    embryos: list[str],
    ctx=None,
    n_val_videos: int | None = None,
    max_pairs_per_video: int | None = None,
    match_um: float = MATCH_UM,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Top-1 link retrieval against ground truth, over ``n_val_videos`` videos.

    Returns ``(summary, per_video)``. ``summary`` holds ``top1``, the
    nearest-neighbour baseline ``top1_nn`` and their ``delta``, the number of
    ground-truth links scored, and ``coverage`` -- the share of links whose both
    endpoints had a detection within ``match_um``, which is the ceiling the
    detector puts on this metric.

    ``delta`` is the one to watch. ``top1`` on its own can look excellent while
    the model has learned nothing an argmin over distances would not have given
    for free.
    """
    import zarr

    from embeddings.dataset import find_coord_files

    device = ctx.device if ctx is not None else next(model.parameters()).device
    core = unwrap_model(model)
    was_training = core.training
    core.eval()

    patch_zyx = tuple(args.patch_zyx)
    n_frames = int(getattr(args, "n_frames", 1) or 1)
    n_val_videos = int(n_val_videos or getattr(args, "n_val_videos", 5) or 5)
    # Defaults to None in DEFAULTS, so argparse gives it no type= and
    # `--max_val_pairs 20` arrives as "20" -- which only shows up as an
    # int >= str TypeError once the cap is actually reached.
    if max_pairs_per_video is not None:
        max_pairs_per_video = (
            None if str(max_pairs_per_video).lower() in ("", "none")
            else int(max_pairs_per_video)
        )
    paths = _paths_for_embryos(args.data_dir, embryos)[:n_val_videos]

    coord_files = {p.stem: p for p in find_coord_files(args.coords_log_folder)}
    rows: list[dict] = []

    try:
        for path in paths:
            # list_datasets yields the stem path -- open_dataset takes it as
            # given, the zarr store needs the suffix put back.
            sample = path.name
            if sample not in coord_files:
                continue
            with np.load(coord_files[sample]) as d:
                det_all = np.asarray(d["coords"])
            gt_nodes, gt_links = _gt_frames(path)
            if not gt_links:
                continue

            volume = zarr.open_group(str(path.parent / f"{sample}.zarr"), mode="r")["0"]
            hits = nn_hits = scored = attempted = 0

            for t in sorted(gt_links):
                pairs = gt_links[t]
                if max_pairs_per_video and attempted >= max_pairs_per_video:
                    break
                det_t = det_all[det_all[:, 0].astype(int) == t][:, 1:4].astype(np.float32)
                det_n = det_all[det_all[:, 0].astype(int) == t + 1][:, 1:4].astype(np.float32)
                if len(det_t) == 0 or len(det_n) == 0:
                    attempted += len(pairs)
                    continue

                src = _attach(gt_nodes[t], det_t, match_um)
                tgt = _attach(gt_nodes[t + 1], det_n, match_um)
                usable = [(i, j) for i, j in pairs if src[i] >= 0 and tgt[j] >= 0]
                attempted += len(pairs)
                if not usable:
                    continue

                emb_t = _embed_frame(core, np.asarray(volume[t]), det_t, patch_zyx,
                                     device, n_frames)
                emb_n = _embed_frame(core, np.asarray(volume[t + 1]), det_n, patch_zyx,
                                     device, n_frames)
                # Both are L2-normalised, so the inner product *is* the cosine.
                nearest = (emb_t @ emb_n.T).argmax(axis=1)

                # The geometric baseline, over the same candidate set and the
                # same links: closest detection in microns, model untouched.
                dist = np.linalg.norm(
                    (det_t[:, None] - det_n[None]) * np.asarray(SCALE, np.float32), axis=2
                )
                closest = dist.argmin(axis=1)

                for i, j in usable:
                    hits += int(nearest[src[i]] == tgt[j])
                    nn_hits += int(closest[src[i]] == tgt[j])
                    scored += 1

            if scored:
                rows.append({
                    "sample": sample,
                    "embryo": sample.split("_")[0],
                    "n_links": scored,
                    "n_gt_links": attempted,
                    "top1": hits / scored,
                    "top1_nn": nn_hits / scored,
                    "delta": (hits - nn_hits) / scored,
                    "coverage": scored / max(attempted, 1),
                })
                if verbose and (ctx is None or ctx.is_main):
                    r = rows[-1]
                    print(f"  {sample}  top1 {r['top1']:.4f} ({hits}/{scored})  "
                          f"nn {r['top1_nn']:.4f}  delta {r['delta']:+.4f}  "
                          f"coverage {r['coverage']:.3f}")
    finally:
        core.train(was_training)

    total = sum(r["n_links"] for r in rows)
    attempted = sum(r["n_gt_links"] for r in rows)
    summary = {
        "top1": sum(r["top1"] * r["n_links"] for r in rows) / total if total else float("nan"),
        "top1_nn": sum(r["top1_nn"] * r["n_links"] for r in rows) / total if total else float("nan"),
        "n_links": total,
        "n_gt_links": attempted,
        "coverage": total / attempted if attempted else float("nan"),
        "n_videos": len(rows),
    }
    summary["delta"] = summary["top1"] - summary["top1_nn"]
    return summary, rows
