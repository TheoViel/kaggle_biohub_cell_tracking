"""Score every candidate link with the edge transformer, on the test split.

`main_edge_predict.py` cannot be used as-is here: it picks its videos with
`load_videos(..., require_geff=True)` and slices them by fold and embryo, and
the test split has neither ground truth nor known embryo names. Everything below
the video list is the same library code that produced the out-of-fold numbers --
`load_video`, `dataset_for`, `score_video`, `link_video` -- so the only thing
this file changes is which stacks are fed in and which checkpoint reads them.

Writes coords + committed edges + the full scored candidate array, which is what
`main_edge_postprocess.py` re-links from. Run from src/, like every entry point.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

import dataclasses

import numpy as np

from edge_transformer.data import load_video
from edge_transformer.infer import dataset_for, load_hoct
from edge_transformer import predict as edge_predict
from edge_transformer.predict import link_video, score_video
from embeddings.dataset import find_coord_files
from tracking_cellmot.io import list_datasets
from training.distributed import cleanup_distributed, init_distributed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log_folder", required=True, help="edge run: args.json + checkpoint")
    p.add_argument("--ckpt", default="fold_fullfit/ckpt_final.pt")
    p.add_argument("--nodes_folder", required=True, help="detections, with embeddings")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--embeddings_key", default=None)
    p.add_argument("--max_nodes_per_frame", type=int, default=None,
                   help="EdgeDataset's own cap, which drops nodes at RANDOM and "
                        "re-draws per window. A training-memory knob; use "
                        "--max_nodes instead")
    p.add_argument("--edge_pair_budget", type=int, default=0,
                   help="cap on B * E_max^2 per forward, 0 = the library default "
                        "(32M). This, not --max_nodes, is what actually governs "
                        "peak GPU memory here -- halving it roughly halves the "
                        "peak, at some wall-clock cost")
    p.add_argument("--max_nodes", type=int, default=0,
                   help="keep only the N most confident detections per frame, "
                        "0 = no cap. A safety net against OOM on an unusually "
                        "crowded stack: the edge model's cost is quadratic in "
                        "the nodes of a frame pair")
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def resolve_ckpt(log_folder: Path, ckpt: str) -> Path:
    """The checkpoint wherever the weights dataset put it -- see infer_detector.py."""
    direct = log_folder / ckpt
    if direct.exists():
        return direct
    found = sorted(log_folder.rglob(Path(ckpt).name))
    if not found:
        available = sorted(str(f.relative_to(log_folder)) for f in log_folder.rglob("*.pt"))
        raise FileNotFoundError(f"{ckpt} not found under {log_folder}. Present: {available}")
    return found[0]


def cap_nodes(video, max_nodes: int):
    """Keep the ``max_nodes`` most confident detections of each frame.

    A bound on the edge model's cost, which is quadratic in the nodes of a frame
    pair. Two things make this different from ``EdgeDataset``'s own
    ``max_nodes_per_frame``, and both matter for inference:

    * **It ranks by confidence.** The dataset's cap drops at random, which is
      right for training -- the survivors stay a fair sample -- and wrong here,
      where a dropped node gets no candidate edges at all and simply cannot be
      linked. If something has to go, it should be the detections the model was
      least sure of.
    * **It is applied once, to the node set.** The dataset's cap is re-drawn per
      window, so a node can be present in one window and absent from the next,
      and the graph a video produces is not a graph of any one node set.

    A cap that never binds costs nothing, so the default is off: the reported
    0.8699 is an uncapped number, and turning this on is a decision to trade
    recall for a memory ceiling.
    """
    if not max_nodes:
        return video
    if video.conf is None:
        raise ValueError(
            "--max_nodes ranks by detection confidence, but this node set has "
            "no 'node_probs' array. Re-run the detection stage, or drop the cap."
        )

    keep_parts = []
    for t in sorted(video.rows_by_frame):
        rows = video.rows_by_frame[t]
        if len(rows) <= max_nodes:
            keep_parts.append(rows)
            continue
        # stable, so ties fall back to file order rather than to whatever the
        # sort happens to do -- and keep ascending, since every consumer reads a
        # frame's rows in file order.
        best = np.argsort(-video.conf[rows], kind="stable")[:max_nodes]
        keep_parts.append(np.sort(rows[best]))
    keep = np.concatenate(keep_parts) if keep_parts else np.empty(0, int)

    if len(keep) == len(video.coords):
        return video

    coords = video.coords[keep]
    frame_of = coords[:, 0].astype(int)

    # Every surviving row moves, so anything indexing into coords has to move
    # with it. Ground-truth links do on the training split -- inference has
    # none, but refusing to handle them would make this untestable against the
    # only data that can say what the cap costs.
    remap = np.full(len(video.coords), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    links = {}
    for t, arr in (video.links or {}).items():
        arr = np.asarray(arr)
        if not len(arr):
            links[t] = arr
            continue
        src, tgt = remap[arr[:, 0].astype(int)], remap[arr[:, 1].astype(int)]
        ok = (src >= 0) & (tgt >= 0)          # a link whose endpoint was cut is gone
        out = arr[ok].copy()
        out[:, 0], out[:, 1] = src[ok], tgt[ok]
        links[t] = out

    return dataclasses.replace(
        video,
        coords=coords,
        emb=None if video.emb is None else video.emb[keep],
        conf=video.conf[keep],
        rows_by_frame={int(t): np.flatnonzero(frame_of == t) for t in np.unique(frame_of)},
        links=links,
    )


def candidate_array(video, scored):
    """(src row, tgt row, prob, distance um, dt) -- a copy of main_edge_predict's.

    Kept identical on purpose: it is the post-processing chain's entire input,
    and the out-of-fold tuning was done against exactly this layout.
    """
    frame_of = np.asarray(video.coords)[:, 0].astype(int)
    if not scored:
        return np.empty((0, 5))
    return np.asarray(
        [(a, b, p, d, frame_of[b] - frame_of[a]) for (a, b), (p, d) in scored.items()],
        dtype=np.float64,
    ).reshape(-1, 5)


def main():
    args = parse_args()
    ctx = init_distributed(None)
    device = ctx.device

    if args.edge_pair_budget:
        # A module constant read inside score_video's batch planner, so it is set
        # rather than passed. Done before any scoring, and echoed below, because
        # a memory knob that is not in the log is a memory knob nobody can
        # reproduce.
        edge_predict.EDGE_PAIR_BUDGET = int(args.edge_pair_budget)

    log_folder = Path(args.log_folder)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = resolve_ckpt(log_folder, args.ckpt)

    model, cfg = load_hoct(str(ckpt_path), device=device, verbose=ctx.is_main)
    emb_key = args.embeddings_key or cfg.get("embeddings_key") or "embeddings"
    use_emb = bool(cfg.get("use_embeddings", True))

    # require_geff=False: the test split has no annotation. load_video already
    # copes -- it attaches links only when a .geff sits beside the stack -- so
    # the videos come back with an empty `links`, which is all inference needs.
    node_files = {p.name: p for p in find_coord_files(args.nodes_folder)}
    paths = sorted(list_datasets(Path(args.data_dir), require_geff=False))
    if args.limit:
        paths = paths[: args.limit]

    videos = []
    for path in paths:
        key = path.name + ".npz" if (path.name + ".npz") in node_files else path.name
        node_path = node_files.get(key)
        if node_path is None:
            if ctx.is_main:
                print(f"!  no detections for {path.name}, skipped")
            continue
        video = load_video(
            path.name, node_path, path,
            use_embeddings=use_emb, embeddings_key=emb_key,
        )
        n_before = len(video.coords)
        video = cap_nodes(video, args.max_nodes)
        if len(video.coords) != n_before and ctx.is_main:
            print(f"   {path.name}: {n_before} -> {len(video.coords)} nodes "
                  f"(cap {args.max_nodes}/frame)")
        videos.append(video)
    if not videos:
        raise FileNotFoundError(
            f"no stack of {args.data_dir} had a matching npz in {args.nodes_folder}"
        )

    dataset = dataset_for(cfg, videos, max_nodes_per_frame=args.max_nodes_per_frame)

    if ctx.is_main:
        print(f"-> checkpoint     : {ckpt_path}")
        print(f"-> nodes_folder   : {args.nodes_folder}")
        print(f"-> embeddings key : {emb_key}  (use_embeddings={use_emb})")
        print(f"-> videos         : {len(videos)}")
        total = sum(len(v.coords) for v in videos)
        per_frame = max((len(r) for v in videos for r in v.rows_by_frame.values()),
                        default=0)
        print(f"-> confidence cap : {args.max_nodes or 'none'} nodes/frame"
              f"   (widest frame after it: {per_frame}, {total} nodes total)")
        print(f"-> dataset cap    : {dataset.max_nodes_per_frame or 'none'}"
              f"   window {dataset.window_size}")
        print(f"-> edge budget    : {edge_predict.EDGE_PAIR_BUDGET:,} pairs/forward")
        print(f"-> threshold      : {args.threshold}")
        print(f"-> writing to     : {out_dir}")
        if ctx.distributed:
            print(f"-> distributed    : {ctx.world_size} processes")
        print()

    mine = list(range(len(videos)))[ctx.rank :: ctx.world_size]
    for i, vi in enumerate(mine):
        video = videos[vi]
        scored = score_video(model, dataset, vi, device, fp16=not args.no_fp16)
        coords, edges = link_video(video, scored, threshold=args.threshold)
        np.savez_compressed(
            out_dir / f"{video.name}.npz",
            coords=np.round(np.asarray(coords, np.float64)).astype(np.int16),
            edges=np.asarray(edges, np.float64).reshape(-1, 4),
            candidates=candidate_array(video, scored),
        )
        print(f"[rank {ctx.rank}] [{i + 1:>3}/{len(mine)}] {video.name}  "
              f"{len(coords)} nodes  {len(edges)} edges", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
