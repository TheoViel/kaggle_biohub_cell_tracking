"""Score every node of a graph, then trim the graph by that score and re-score it.

    python main_annotated_predict.py --run_dir ../logs/annotated_run/<date>/<id> \
        --preds_dir ../logs/edge_run/2026-09-15/0/preds_0.45_pp --sweep

Two halves, deliberately separate. Scoring every detection is the GPU cost and
does not depend on where the threshold goes; the threshold is the part that has
to be swept, and the only honest reading of one is the competition metric out of
the real scorer.

What trimming is supposed to do
-------------------------------
`adj_edge_jaccard = edge_jaccard * (1 - 0.1 * (N_pred - N_total) / N_total)`.
The first factor only sees edges that touch an annotated cell. The second
charges for every node in the file. So deleting a node the annotation never
looks at costs nothing in the first factor and pays in the second -- which is
the whole of this idea, and also the reason to be careful about what it means.

It is worth saying plainly that this is an adjustment-term trade rather than a
tracking improvement. `N_total` is the dataset's estimate of how many cells are
really there, so trimming below it makes the ratio negative and the multiplier
larger than one: the metric pays for under-detection. The test asked of the
model is therefore exactly the right one -- **edge_jaccard must not fall while
adj_edge_jaccard rises**. If edge_jaccard falls, the trim is eating real links
and the gain is bought, not free.

A node is only removable if nothing is attached to it that the metric scores, so
the sweep reports both terms at every threshold and the node count with them.
Edges are dropped with their endpoints, which is what makes `edge_jaccard`
fall if the ranking is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotated_cls.dataset import AnnotatedCellDataset  # noqa: E402
from annotated_cls.labels import ground_truth, load_graph, node_labels  # noqa: E402
from annotated_cls.model import AnnotatedNet  # noqa: E402
from params import EMBRYOS, SCALE, TRAIN_DIR  # noqa: E402
from postprocessing.fast_score import roll_up, row_from_counts  # noqa: E402

SCALE_ARR = np.asarray(SCALE, dtype=np.float64)


def _load_model(run_dir, fold, device):
    args = json.load(open(Path(run_dir) / "args.json"))
    model = AnnotatedNet(
        backbone=args["backbone"], pretrained=False,
        patch_yx=int(args["patch_yx"]), patch_z=int(args["patch_z"]),
        n_frames=int(args["n_frames"]), dropout=0.0,
    )
    ckpt = Path(run_dir) / f"fold_{fold}" / "ckpt_final.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval(), args


@torch.no_grad()
def score_stack(model, args, data_dir, preds_dir, sample, device, batch=512):
    """`(N,)` probability per node of ``sample``, in coords order.

    One zarr read per frame, not per node. The normalisation the dataset uses is
    per *slab* -- the `patch_z` slices centred on the node's z -- so nodes are
    grouped by that band, and each band is normalised once and cut for every
    node in it at once. A stack has ~25k detections over ~100 frames and at most
    64 bands a frame, so this is ~100 zarr reads instead of 25k, which is the
    difference between minutes and hours.

    Sampling matches `AnnotatedCellDataset.__getitem__` unaugmented, so a scored
    crop and a training crop of the same place are the same array.
    """
    import zarr

    from embeddings.patches import extract_patches

    coords, _ = load_graph(preds_dir, sample)
    vol = zarr.open_group(str(Path(data_dir) / f"{sample}.zarr"), mode="r")["0"]
    pz, pyx = int(args["patch_z"]), int(args["patch_yx"])
    n_t, n_z = vol.shape[0], vol.shape[1]
    out = np.zeros(len(coords), np.float32)

    frame_of = coords[:, 0].astype(int)
    z0_of = np.rint(coords[:, 1]).astype(int) - pz // 2
    clips, owners = [], []

    def flush():
        if not clips:
            return
        arr = np.concatenate(clips, 0)[:, None]          # (n, 1, pz, pyx, pyx)
        rows = np.concatenate(owners)
        for s in range(0, len(arr), batch):
            logits = model(torch.from_numpy(arr[s:s + batch]).to(device))
            out[rows[s:s + batch]] = torch.sigmoid(logits.float()).cpu().numpy()
        clips.clear()
        owners.clear()

    for t in range(int(frame_of.min()), int(frame_of.max()) + 1):
        here = np.flatnonzero(frame_of == t)
        if not len(here) or t < 0 or t >= n_t:
            continue
        frame = np.asarray(vol[t], np.float32)           # (Z, Y, X), one read
        for z0 in np.unique(z0_of[here]):
            rows = here[z0_of[here] == z0]
            lo, hi = max(0, int(z0)), min(n_z, int(z0) + pz)
            slab = np.zeros((pz, frame.shape[1], frame.shape[2]), np.float32)
            if lo < hi:
                slab[lo - int(z0): hi - int(z0)] = frame[lo:hi]
            p1, p99 = np.percentile(slab, [1.0, 99.99])
            slab = np.clip((slab - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
            centres = np.stack([
                np.full(len(rows), pz // 2, np.float32),
                coords[rows, 2].astype(np.float32),
                coords[rows, 3].astype(np.float32)], 1)
            clips.append(extract_patches(
                slab, centres, patch_zyx=(pz, pyx, pyx), n_frames=1)[:, 0])
            owners.append(rows)
            if sum(len(c) for c in clips) >= batch:
                flush()
    flush()
    return out


def trim(coords, edges, prob, threshold):
    """Drop nodes below ``threshold`` and every edge that touched one."""
    keep = prob >= threshold
    if keep.all():
        return coords, edges, np.arange(len(coords))
    remap = -np.ones(len(coords), np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    e = np.asarray(edges, np.float64).reshape(-1, 4)
    if len(e):
        ok = keep[e[:, 0].astype(int)] & keep[e[:, 1].astype(int)]
        e = e[ok].copy()
        e[:, 0] = remap[e[:, 0].astype(int)]
        e[:, 1] = remap[e[:, 1].astype(int)]
    return coords[keep], e, np.flatnonzero(keep)


def score_row(coords, edges, gt, n_total):
    """The metric's own counts for a graph, via `labels.node_labels`' matcher."""
    from postprocessing.fast_score import match_nodes

    m = match_nodes(coords.astype(np.int64), gt)
    gid = np.full(len(coords), -1, np.int64)
    for r, n in m.items():
        gid[r] = n
    hc = np.isin(gid, list(gt.has_child))
    hp = np.isin(gid, list(gt.has_parent))
    e = np.asarray(edges).reshape(-1, 4) if len(edges) else np.zeros((0, 4))
    if not len(e):
        return row_from_counts(0, 0, gt.n_edges, len(coords), n_total,
                               len(set(m.values())) / gt.n_nodes)
    a, b = e[:, 0].astype(int), e[:, 1].astype(int)
    valid = hc[a] | hp[b]
    tp = np.fromiter(((int(x), int(y)) in gt.pairs for x, y in zip(gid[a], gid[b])),
                     bool, len(a))
    tp = int((tp & valid).sum())
    return row_from_counts(tp, int(valid.sum()) - tp, gt.n_edges - tp,
                           len(coords), n_total, len(set(m.values())) / gt.n_nodes)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--preds_dir", required=True)
    p.add_argument("--data_dir", default=TRAIN_DIR)
    p.add_argument("--embryo", default="44b6")
    p.add_argument("--fold", type=int, default=None,
                   help="default = the fold whose validation embryo this is, "
                        "which is the only one that is out of sample")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--scores_dir", default=None,
                   help="cache the per-node probabilities here")
    p.add_argument("--quantiles", default="0,0.05,0.1,0.2,0.3,0.4,0.5,0.6",
                   help="fraction of each stack's nodes to trim, lowest score "
                        "first. A quantile rather than a probability because "
                        "the model is calibrated to its training prior, not to "
                        "the real 214:1 one, so the absolute values mean little "
                        "and the question is how many nodes can go.")
    p.add_argument("--out_dir", default=None, help="write the trimmed npz here")
    p.add_argument("--apply", type=float, default=None,
                   help="write one threshold rather than sweeping")
    a = p.parse_args()

    fold = a.fold if a.fold is not None else EMBRYOS.index(a.embryo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = _load_model(a.run_dir, fold, device)
    preds_dir = Path(a.preds_dir)
    samples = sorted(f.stem for f in preds_dir.glob(f"{a.embryo}_*.npz"))[: a.limit]
    print(f"-> {len(samples)} stacks, fold {fold} (val embryo {EMBRYOS[fold]})")
    if EMBRYOS[fold] != a.embryo:
        print("   WARNING: this fold trained on this embryo -- the scores are leaky")

    cache = Path(a.scores_dir) if a.scores_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    ths = [float(x) for x in a.quantiles.split(",")] if a.apply is None else [a.apply]
    rows = {t: [] for t in ths}
    kept_frac = {t: [] for t in ths}
    aucs = []
    for i, s in enumerate(samples):
        coords, edges = load_graph(preds_dir, s)
        f = cache / f"{s}.npy" if cache else None
        if f is not None and f.exists():
            prob = np.load(f)
        else:
            prob = score_stack(model, margs, a.data_dir, preds_dir, s, device)
            if f is not None:
                np.save(f, prob)
        gt = ground_truth(s, a.data_dir)
        n_total = _n_total(s)
        y, _ = node_labels(coords, edges, gt)
        if y.any() and not y.all():
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(y, prob))
        for t in ths:
            # Per stack, not pooled: the score distribution shifts stack to
            # stack, and a global cut would trim one stack to nothing while
            # leaving another untouched.
            cut = -np.inf if t <= 0 else float(np.quantile(prob, t))
            c, e, _ = trim(coords, edges, prob, cut)
            kept_frac[t].append(len(c) / max(len(coords), 1))
            rows[t].append(score_row(c, e, gt, n_total))
            if a.apply is not None and a.out_dir:
                out = Path(a.out_dir)
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / f"{s}.npz",
                                    coords=np.round(c).astype(np.int16),
                                    edges=np.asarray(e, np.float64).reshape(-1, 4))
        print(f"[{i + 1:>3}/{len(samples)}] {s}", flush=True)

    print(f"\nper-node AUC over {len(aucs)} stacks: mean {np.mean(aucs):.4f}  "
          f"median {np.median(aucs):.4f}")
    print(f"\n{'trimmed':>9}{'kept':>7}{'nodes':>10}{'tp':>7}{'fp':>6}{'fn':>6}"
          f"{'edge_J':>9}{'d edge':>9}{'adj_J':>9}{'d adj':>9}")
    base_adj = base_edge = None
    for t in ths:
        ru = roll_up(rows[t])
        n = int(sum(r["n_nodes"] for r in rows[t]))
        if base_adj is None:
            base_adj, base_edge = ru["adj_edge_jaccard"], ru["edge_jaccard"]
        print(f"{t:>9.2f}{np.mean(kept_frac[t]):>7.2f}{n:>10}{ru['edge_tp']:>7}"
              f"{ru['edge_fp']:>6}{ru['edge_fn']:>6}{ru['edge_jaccard']:>9.4f}"
              f"{ru['edge_jaccard'] - base_edge:>+9.4f}"
              f"{ru['adj_edge_jaccard']:>9.4f}{ru['adj_edge_jaccard'] - base_adj:>+9.4f}")
    print("\nedge_jaccard must not fall. A rise in adj_edge_jaccard that comes "
          "with a fall in edge_jaccard is the trim eating real links.")


def _n_total(sample):
    import pandas as pd
    global _META
    try:
        _META
    except NameError:
        _META = pd.read_csv(Path(TRAIN_DIR).parent / "metadata.csv").set_index("sample_name")
    return float(_META.loc[sample, "estimated_number_of_nodes"])


if __name__ == "__main__":
    main()
