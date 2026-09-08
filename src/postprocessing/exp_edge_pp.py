"""Score named post-processing chains over the edge transformer's candidate graph.

    cd src/postprocessing
    python exp_edge_pp.py --preds_dir ../../logs/edge_run/2026-09-04/3/preds_edge \
                          --spread 10 --chains raw,assign@0.5,best
    python exp_edge_pp.py --preds_dir ...        # every sample, every chain

Chains are named in `CHAINS` below and take `(coords, edges, cand)` back to
`(coords, edges)`. Scoring is `fast_score`, which reproduces the competition
metric exactly (`validate_fast.py`) while being fast enough to sweep with; the
reported number is the fold score, i.e. the per-sample adjusted edge Jaccard
weighted by sample size, which is what `summarise` computes for one embryo.

Chains can also be written inline, so sweeping a knob needs no edit here:

    --chains '[recover=1,min_track=4],[recover=1,min_track=4,linefit=0.6]'

`--spread N` takes N samples spanning the difficulty range rather than the
first N, so a chain tuned on the subset is not tuned on the easy end of it.

Two of the knobs bring in the contrastive embeddings from `--embeddings_dir`,
which is how `notebooks/CompareCV.ipynb` links: `ascent=1` swaps the whole
graph builder for that notebook's linker, and `emb_weight` instead blends its
cosine into this one's probabilities. Both are measured in README_edge.md and
neither pays, because this model already reads those embeddings.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402  (chdir to src/ and set the path)
import dedup  # noqa: E402
import edge_pp  # noqa: E402
import fast_score  # noqa: E402
import torch  # noqa: E402

from params import TRAIN_DIR  # noqa: E402

# `recover`'s drift estimate goes through torch, whose intra-op pool sizes
# itself from the core count and ignores OMP_NUM_THREADS -- with one worker per
# core that is a 3x oversubscription, and the sweep runs slower than serial.
torch.set_num_threads(1)

CACHE = Path("../output/pp_experiments/gt_cache")


# --------------------------------------------------------------------------
# chains
# --------------------------------------------------------------------------

def _raw(coords, edges, cand, **kw):
    return coords, edges


def _chain(coords, edges, cand, threshold=0.4, flow_weight=0.0,
           max_residual=np.inf, min_track=0, recover=0, bridge=0.0,
           recover_max_gap=3, recover_max_residual=5.0, recover_min_dup=4.0,
           recover_passes=1, through=0.0, through_max_gap=2, edge_max_um=0.0,
           linefit=0.0, linefit_window=2, pp_public=False,
           emb_weight=0.0, emb_momentum=0.0,
           ascent=0, w_emb=0.5, w_trans=0.5, momentum=0.5, ascent_max_um=0.0,
           ascent_prob='cand',
           velocity=0, vel_momentum=0.5, vel_max_um=7.0,
           vel_prob_weight=0.0, vel_seed_drift=1,
           vel_weight=0.0, vel_max_residual=np.inf, cost='logprob',
           match='hungarian',
           dedup_radius=0.0, dedup_overlap=0.6, dedup_keep_longer=True,
           emb=None):
    """The full chain, every stage switchable.

    Order matters and is the same argument `pipeline.py` makes: recovery before
    pruning, because the fragments a gap-fill joins are exactly the short chains
    pruning would throw away; bridging before pruning for the same reason.

    `ascent=1` swaps the graph builder for `CompareCV.ipynb`'s linker -- dense
    cosine matching with an EMA track state and a per-frame dynamic threshold --
    keeping every stage after it. `emb_weight` instead leaves `assign` in place
    and blends the cosine into its probabilities.

    `dedup_radius` (0 = off) runs `dedup.suppress_parallel` after `recover` and
    before `min_track`: after recovery both halves of a shadowing pair are as
    complete as this chain will ever make them, so the overlap fraction and the
    length comparison `suppress_parallel` uses to pick a winner are measured on
    the real track rather than a fragment recovery has not reached yet; before
    pruning because dropping a whole duplicate track can turn what used to be
    the *other* side of a shadow pair into a short leftover, which `min_track`
    should then get a chance to clean up.
    """
    assign_kw = dict(flow_weight=flow_weight, max_residual=max_residual,
                     emb=emb, emb_weight=emb_weight, emb_momentum=emb_momentum,
                     vel_weight=vel_weight, vel_momentum=vel_momentum,
                     vel_max_residual=vel_max_residual,
                     vel_seed_drift=bool(vel_seed_drift), cost=cost,
                     match=match)
    if velocity:
        edges = edge_pp.velocity_link(
            coords, cand, momentum=vel_momentum, max_um=vel_max_um,
            prob_weight=vel_prob_weight, seed_drift=bool(vel_seed_drift),
            threshold=threshold)
    elif ascent:
        if emb is None:
            raise SystemExit("ascent=1 needs the embeddings; see --embeddings_dir")
        # `CompareCV.ipynb` reads its probabilities from the npz's
        # *committed* edges, not from every scored candidate: the model's own
        # assignment is what the cosine is being asked to arbitrate, and the
        # two give the linker very different matrices.
        prob_src = cand
        if ascent_prob == "edges":
            e = np.asarray(edges, float).reshape(-1, 4)
            prob_src = np.column_stack([e, np.ones(len(e))]) if len(e) \
                else np.empty((0, 5))
        edges = edge_pp.ascent_link(coords, emb, prob_src, w_emb=w_emb,
                                    w_trans=w_trans, momentum=momentum,
                                    max_um=ascent_max_um)
    else:
        edges = edge_pp.assign(coords, cand, threshold=threshold, **assign_kw)
    if edge_max_um:
        edges = edge_pp.drop_long(edges, edge_max_um)
    if through:
        # Before `recover`, which refuses a bridge that lands on an existing
        # detection: those are exactly the gaps this stage closes, and closing
        # one with the detection that is already there beats inventing a node.
        edges = edge_pp.link_through_isolated(
            coords, edges, max_gap=through_max_gap, max_um=through)
    if recover:
        # The re-link after recovery cannot use the embeddings: a recovered
        # node is interpolated between two endpoints and the contrastive model
        # never saw it, so it has no embedding row. It re-links on the model's
        # probabilities alone, whichever rule built the graph.
        recover_kw = {k: v for k, v in assign_kw.items()
                      if k not in ("emb", "emb_weight", "emb_momentum")}
        coords, edges = edge_pp.recover_nodes(
            coords, edges, cand, max_gap=recover_max_gap,
            max_residual=recover_max_residual, min_dup=recover_min_dup,
            passes=recover_passes, threshold=threshold, assign_kw=recover_kw)
    if bridge:
        edges = edge_pp.bridge_tracks(coords, edges, max_um=bridge)
    if dedup_radius:
        keep = dedup.suppress_parallel(coords, edges, radius=dedup_radius,
                                       overlap=dedup_overlap,
                                       keep_longer=dedup_keep_longer)
        coords, edges = dedup.apply_mask(coords, edges, keep)
    if min_track:
        coords, edges = edge_pp.prune_short(coords, edges, min_track)
    if linefit:
        # After pruning, so the fit never runs along a chain that is about to
        # be deleted, and last, because it moves nodes rather than links.
        coords = edge_pp.linefit_smooth(coords, edges, window=linefit_window,
                                        weight=linefit)
    if pp_public:
        from stages import run_pp_public
        coords, edges = run_pp_public(
            coords, edges, filter_short_tracks=1, min_track_len=4,
            relink_tight_um=6.5)
        coords = np.asarray(coords, float)
    return coords, edges


CHAINS = {
    # the model's own output, as main_edge_predict.py wrote it
    "raw": _raw,
    # the same assignment, re-decided offline: should reproduce `raw` exactly
    "assign": partial(_chain),
}

for _th in (0.30, 0.35, 0.45, 0.50, 0.55, 0.60):
    CHAINS[f"th{_th:g}"] = partial(_chain, threshold=_th)
for _mt in (2, 3, 4, 6):
    CHAINS[f"prune{_mt}"] = partial(_chain, min_track=_mt)
CHAINS["recover"] = partial(_chain, recover=1)
CHAINS["bridge6"] = partial(_chain, bridge=6.0)
CHAINS["pp_public"] = partial(_chain, pp_public=True)


def resolve(name):
    """A chain from `CHAINS`, or an inline `[key=value,...]` spec.

    Sweeping a knob should not need an edit here, so a chain can also be
    written out on the command line -- `[threshold=0.45,min_track=3,recover=1]`
    -- and every keyword `_chain` takes is available that way.
    """
    if not name.startswith("["):
        if name not in CHAINS:
            raise SystemExit(f"unknown chain '{name}'; known: {', '.join(CHAINS)}")
        return CHAINS[name]
    kw = {}
    for part in name.strip("[]").split(","):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        kw[key.strip()] = _literal(value.strip())
    bad = set(kw) - set(_chain.__code__.co_varnames[: _chain.__code__.co_argcount])
    if bad:
        raise SystemExit(f"unknown chain setting(s): {', '.join(sorted(bad))}")
    return partial(_chain, **kw)


def _literal(text):
    if text in ("inf", "np.inf"):
        return np.inf
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {"True": True, "False": False}.get(text, text)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _gt(sample):
    """`fast_score.GroundTruth` for a sample, cached: loading a geff is slow."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{sample}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    import tracksdata as td
    result = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{sample}.geff")
    gt = fast_score.GroundTruth(result[0] if isinstance(result, tuple) else result)
    path.write_bytes(pickle.dumps(gt))
    return gt


_META = None


def _n_total(sample):
    global _META
    if _META is None:
        import pandas as pd
        _META = pd.read_csv("../input/metadata.csv").set_index("sample_name")
    return float(_META.loc[sample, "estimated_number_of_nodes"])


def _run_one(job):
    path, names, embeddings_dir = job
    sample = Path(path).stem
    coords0, edges0, cand = edge_pp.load_pred(path)
    gt, n_total = _gt(sample), _n_total(sample)

    emb = None
    if embeddings_dir:
        loaded = edge_pp.load_embeddings(Path(embeddings_dir) / f"{sample}.npz")
        if loaded is not None:
            emb, emb_coords = loaded
            # Row i must be the same node in both files, or every cosine is
            # between two unrelated cells and the failure is silent.
            if not np.array_equal(emb_coords.astype(np.int64),
                                  coords0.astype(np.int64)):
                raise SystemExit(
                    f"{sample}: the embeddings npz and the prediction npz "
                    "disagree on the detections; they must come from the same "
                    "nodes folder"
                )

    rows = {}
    for name in names:
        t0 = time.time()
        coords, edges = resolve(name)(coords0.copy(), edges0.copy(), cand, emb=emb)
        # Round before scoring, because `submission.py` writes int(round(z)):
        # a stage that leaves sub-voxel positions -- `linefit_smooth` and
        # `recover` both do -- must be judged on what actually gets submitted.
        # The z voxel is 1.625 um, so this is not a rounding detail, it is up
        # to 0.81 um of the error the smoothing exists to remove.
        row = fast_score.metrics(np.round(np.asarray(coords, float)), edges,
                                 gt, n_total)
        row["seconds"] = time.time() - t0
        row["n_edges"] = len(edges)
        rows[name] = row
    return sample, rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", required=True)
    p.add_argument("--chains", default=None,
                   help=f"comma-separated; default all of {', '.join(CHAINS)}")
    p.add_argument("--embeddings_dir", default="../logs/2026-09-03/13/embeddings/",
                   help="the contrastive embeddings the `ascent` and `emb_weight` "
                        "chains link on; must be the nodes folder the predictions "
                        "were made from. '' to skip loading them")
    p.add_argument("--spread", type=int, default=None,
                   help="N samples spanning the difficulty range, not the first N")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--out", default=None, help="write the per-sample rows here")
    args = p.parse_args()

    paths = sorted(Path(args.preds_dir).glob("*.npz"))
    if args.spread:
        paths = spread(paths, args.spread)
    paths = paths[: args.limit]
    names = _split_chains(args.chains) if args.chains else list(CHAINS)
    for n in names:
        resolve(n)  # fail on a bad spec before spending the workers on it

    print(f"-> {len(paths)} samples x {len(names)} chains, {args.jobs} workers\n")
    per_chain = {n: [] for n in names}
    with Pool(args.jobs) as pool:
        for i, (sample, rows) in enumerate(
            pool.imap_unordered(
                _run_one, [(str(q), names, args.embeddings_dir) for q in paths])
        ):
            for n, row in rows.items():
                row["sample_name"] = sample
                per_chain[n].append(row)
            print(f"[{i + 1:>3}/{len(paths)}] {sample}", flush=True)

    width = max(14, max(len(n) for n in names) + 2)
    print(f"\n{'chain':<{width}}{'score':>9}{'delta':>9}{'J':>9}{'tp':>8}{'fp':>7}"
          f"{'fn':>7}{'recall':>8}{'nodes':>9}{'ratio':>7}{'s/stk':>7}")
    base = None
    summary = {}
    for n in names:
        s = fast_score.roll_up(per_chain[n])
        base = s["adj_edge_jaccard"] if base is None else base
        nodes = float(np.mean([r["n_nodes"] for r in per_chain[n]]))
        ratio = float(np.mean([r["total_node_ratio"] for r in per_chain[n]])) + 1
        secs = float(np.mean([r["seconds"] for r in per_chain[n]]))
        summary[n] = dict(s, n_nodes=nodes, node_ratio=ratio, seconds=secs)
        print(f"{n:<{width}}{s['adj_edge_jaccard']:>9.4f}"
              f"{s['adj_edge_jaccard'] - base:>+9.4f}{s['edge_jaccard']:>9.4f}"
              f"{s['edge_tp']:>8.0f}{s['edge_fp']:>7.0f}{s['edge_fn']:>7.0f}"
              f"{s['node_recall']:>8.3f}{nodes:>9.0f}{ratio:>7.3f}{secs:>7.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"preds_dir": args.preds_dir, "n_samples": len(paths),
             "summary": summary, "per_sample": per_chain}, indent=2, default=float))
        print(f"\n-> wrote {args.out}")


def _split_chains(text):
    """Split on commas, except the ones inside an inline `[...]` spec."""
    out, depth, current = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            out.append(current)
            current = ""
            continue
        depth += (ch == "[") - (ch == "]")
        current += ch
    if current:
        out.append(current)
    return [c.strip() for c in out if c.strip()]


def spread(paths, n):
    """N samples spanning the difficulty range, ordered by node count.

    Difficulty here is over-detection: the stacks that score worst are the ones
    with the most detections per annotated cell (README, "Why the worst stacks
    are the worst"), and that needs no ground truth to measure.
    """
    sizes = []
    for q in paths:
        with np.load(q) as d:
            sizes.append((len(d["coords"]) / max(_n_total(q.stem), 1), q))
    sizes.sort()
    idx = np.linspace(0, len(sizes) - 1, min(n, len(sizes))).round().astype(int)
    return [sizes[i][1] for i in sorted(set(idx.tolist()))]


if __name__ == "__main__":
    main()
