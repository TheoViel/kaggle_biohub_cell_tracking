"""Tune `edge_pp.propose_divisions` against the real competition metric.

    cd src/postprocessing
    python exp_divisions.py --preds_dir <abs>/logs/edge_run/2026-09-08/13/preds_0.50
    python exp_divisions.py --preds_dir ... --configs '[min_prob=0.15],[min_prob=0.3]'

Why this exists rather than another chain in `exp_edge_pp.py`: `fast_score`
computes the edge Jaccard and nothing else, so it is blind to
`0.1 * division_jaccard` -- the whole term this stage exists to move. Every
number here comes from `tracking_cellmot.metrics.evaluate`, the same scorer
`main_edge_postprocess.py` reports.

Two things make that affordable.

**The base chain is cached.** `assign` and `recover` are the expensive stages
and no division knob touches them, so they run once per stack into
`--cache_dir` and every config after that starts from the saved graph.

**A stack with no proposal is not re-scored.** The stage only ever *adds*
edges, so when it adds none the metrics are bit-identical to the baseline's,
which is already on disk. Divisions are rare -- 26 in 71 stacks for 44b6, 125
in 128 for 6bba -- so most stacks fall through this path and a sweep costs a
fraction of a full re-score.

The reported number is `cv_score`: the mean over embryos of
`adj_edge_jaccard + 0.1 * division_jaccard`, which is what
`main_edge_postprocess.py` prints and what a fold is judged on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402  (chdir to src/ and set the path)
import edge_pp  # noqa: E402
import torch  # noqa: E402
import tracksdata as td  # noqa: E402

from params import EMBRYOS, SCALE, TRAIN_DIR  # noqa: E402
from tracking.post_process import build_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate,
    macro_summarise,
    node_recall,
    per_sample_metrics,
    summarise,
)

torch.set_num_threads(1)

CLF_PATH = None

# Out-of-fold classifier scores, when a run is judging one: {stem: {(parent, q): score}}.
# Loaded lazily per worker so the Pool does not pickle it per task.
_CLF_PATH = None
_CLF = None


def _clf_for(stem, path=None):
    """The classifier's scores for one stack, or None.

    The path travels in the config rather than through a module global: the
    global plus a Pool initialiser worked when this module was imported and
    silently produced no scores when it was run as __main__, which is the one
    way it is actually used.
    """
    global _CLF, _CLF_PATH
    path = path or _CLF_PATH
    if path is None:
        return None
    if _CLF is None or _CLF_PATH != path:
        _CLF_PATH = path
        d = np.load(path, allow_pickle=True)
        by = {}
        for s, m, sc in zip(d["stems"], d["meta"], d["scores"]):
            if sc == sc:  # not NaN: scored out of fold
                by.setdefault(str(s), {})[(int(m[1]), int(m[3]))] = float(sc)
        _CLF = by
    return _CLF.get(stem)


def _init_worker(clf_path):
    global _CLF_PATH
    _CLF_PATH = clf_path


def apply_classifier(coords, edges, scores, top_k=None, min_score=0.0):
    """Commit the highest-scored division candidates the classifier ranked.

    Same structural rules as `propose_divisions` -- a child takes one parent, a
    parent forks once -- but the ordering is the model's, not
    `parent_dist + 0.15 * sister_dist`.
    """
    edges = np.asarray(edges, float).reshape(-1, 4)
    if not scores:
        return edges, 0
    tgt = edges[:, 1].astype(int)
    has_parent = np.zeros(len(coords), bool)
    has_parent[tgt] = True
    src = edges[:, 0].astype(int)
    n_children = np.bincount(src, minlength=len(coords))
    p_um = edge_pp.pos_um(coords)

    ranked = sorted(((v, k) for k, v in scores.items() if v >= min_score),
                    reverse=True)
    if top_k is not None:
        ranked = ranked[:int(top_k)]
    added, used = [], set()
    for _, (parent, q) in ranked:
        if has_parent[q] or parent in used or n_children[parent] != 1:
            continue
        added.append((parent, q, 1.0, float(np.linalg.norm(p_um[parent] - p_um[q]))))
        has_parent[q] = True
        used.add(parent)
    if added:
        edges = np.vstack([edges, np.asarray(added, float)])
    return edges, len(added)

# The chain the divisions are added to, from main_edge_postprocess.DEFAULTS.
BASE = dict(
    threshold=0.4, recover=1, recover_max_gap=5, recover_max_residual=5.0,
    recover_min_dup=3.0, min_track=5, linefit=0.6, linefit_window=3,
    vel_weight=0.02, vel_momentum=0.7,
)

DIV_DEFAULTS = dict(
    min_prob=0.2, max_parent_um=9.0, max_existing_child_um=10.0,
    max_sister_um=14.0, symmetry_tau=0.6, diverge_um=2.25,
    require_divergence=1, require_mutual_nn=1,
    frame_frac_cap=0.0076, global_frac_cap=0.00375,
    # merge_duplicate_parents, which runs first. 0 turns it off.
    merge_dup_um=0.0, merge_require_mutual=1,
    # classifier selection instead of the gates
    clf=0, clf_top_k=0, clf_min_score=0.0, clf_path=None,
    # merge_shadow_forks: the temporal duplicate test. 0 turns it off.
    shadow_radius=0.0, shadow_min=3, shadow_frac=0.6,
    # assign(fork_penalty=...): a priced fork in the assignment itself. It
    # changes `assign`, so a config using it cannot reuse the cached base chain
    # and is rebuilt from the raw npz instead.
    fork_threshold=None,
)

MERGE_KEYS = ("max_parent_um", "max_sister_um", "symmetry_tau", "diverge_um",
              "require_divergence", "global_frac_cap")


def _literal(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_config(spec):
    """`[min_prob=0.15,diverge_um=3]` -> a dict over DIV_DEFAULTS."""
    cfg = dict(DIV_DEFAULTS)
    if spec.strip() in ("base", "off"):
        return None if spec.strip() == "off" else cfg
    for part in spec.strip("[]").split(","):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in DIV_DEFAULTS or key == "clf_path":
            raise SystemExit(f"unknown division setting {key!r}; "
                             f"known: {', '.join(sorted(DIV_DEFAULTS))}")
        cfg[key] = _literal(value.strip())
    return cfg


# --------------------------------------------------------------------------
# the cached base chain
# --------------------------------------------------------------------------

def build_cache(job):
    """Run assign + recover once and save the graph the sweep starts from."""
    path, cache_dir = job
    stem = Path(path).stem
    out = Path(cache_dir) / f"{stem}.npz"
    if out.exists():
        return stem, "cached"
    coords, _, cand = edge_pp.load_pred(path)
    akw = dict(vel_weight=BASE["vel_weight"], vel_momentum=BASE["vel_momentum"])
    edges = edge_pp.assign(coords, cand, threshold=BASE["threshold"], **akw)
    coords, edges = edge_pp.recover_nodes(
        coords, edges, cand, max_gap=BASE["recover_max_gap"],
        max_residual=BASE["recover_max_residual"], min_dup=BASE["recover_min_dup"],
        threshold=BASE["threshold"], assign_kw=akw,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, coords=coords, edges=edges, candidates=cand)
    return stem, "built"


def _finish(coords, edges):
    """prune -> linefit, the tail of the chain that follows the divisions."""
    coords, edges = edge_pp.prune_short(coords, edges, BASE["min_track"])
    coords = edge_pp.linefit_smooth(
        coords, edges, window=BASE["linefit_window"], weight=BASE["linefit"])
    return coords, edges


def _score(stem, coords, edges):
    graph = build_graph(np.round(np.asarray(coords)).astype(np.int64), [
        (int(s), int(t), float(p), float(d))
        for s, t, p, d in np.asarray(edges, float).reshape(-1, 4)
    ])
    gt = td.graph.IndexedRXGraph.from_geff(Path(TRAIN_DIR) / f"{stem}.geff")
    gt = gt[0] if isinstance(gt, tuple) else gt
    row = per_sample_metrics(evaluate(graph, gt, scale=SCALE),
                             _n_total(stem), node_recall(graph, gt))
    row["sample_name"] = stem
    row["embryo"] = stem.split("_")[0]
    return row


_META = None


def _n_total(sample):
    global _META
    import pandas as pd
    if _META is None:
        meta = Path(TRAIN_DIR).parent / "metadata.csv"
        _META = pd.read_csv(meta).set_index("sample_name") if meta.exists() \
            else pd.DataFrame()
    if sample in getattr(_META, "index", []):
        return float(_META.loc[sample, "estimated_number_of_nodes"])
    return float("nan")


def run_stack(job):
    """One stack under one division config; `None` cfg is the baseline."""
    cache_path, cfg = job
    stem = Path(cache_path).stem
    with np.load(cache_path) as d:
        coords, edges, cand = d["coords"], d["edges"], d["candidates"]

    # `fork_penalty` lives inside `assign`, so the cached graph is the wrong
    # starting point: redo assign + recover with it on.
    if cfg is not None and cfg.get("fork_threshold") is not None:
        akw = dict(vel_weight=BASE["vel_weight"], vel_momentum=BASE["vel_momentum"],
                   fork_threshold=float(cfg["fork_threshold"]))
        edges = edge_pp.assign(coords, cand, threshold=BASE["threshold"], **akw)
        coords, edges = edge_pp.recover_nodes(
            coords, edges, cand, max_gap=BASE["recover_max_gap"],
            max_residual=BASE["recover_max_residual"],
            min_dup=BASE["recover_min_dup"], threshold=BASE["threshold"],
            assign_kw=akw)

    stats = {}
    if cfg is not None and cfg.get("clf"):
        scores = _clf_for(stem, cfg.get("clf_path"))
        edges, n = apply_classifier(
            coords, edges, scores,
            top_k=(cfg.get("clf_top_k") or None),
            min_score=cfg.get("clf_min_score", 0.0))
        stats = {"added": n, "proposals": len(scores or {})}
        if n == 0:
            return stem, None, stats
        coords, edges = _finish(coords, edges)
        return stem, _score(stem, coords, edges), stats
    if cfg is not None:
        kw = dict(cfg)
        # Selection-rule keys; `propose_divisions` does not take them.
        for k in ("clf", "clf_top_k", "clf_min_score", "clf_path"):
            kw.pop(k, None)
        kw["require_divergence"] = bool(kw["require_divergence"])
        kw["require_mutual_nn"] = bool(kw["require_mutual_nn"])
        shadow_r = float(kw.pop("shadow_radius", 0.0))
        shadow_min = int(kw.pop("shadow_min", 3))
        shadow_frac = float(kw.pop("shadow_frac", 0.6))
        kw.pop("fork_threshold", None)
        if shadow_r > 0:
            edges, sstats = edge_pp.merge_shadow_forks(
                coords, edges, radius=shadow_r, min_shadow=shadow_min,
                shadow_frac=shadow_frac,
                max_sister_um=kw["max_sister_um"],
                global_frac_cap=kw["global_frac_cap"], return_stats=True)
            stats.update({f"shadow_{k}": v for k, v in sstats.items()})
        merge_um = float(kw.pop("merge_dup_um", 0.0))
        merge_mutual = bool(kw.pop("merge_require_mutual", 1))
        if merge_um > 0:
            # First: a division already in the graph but split over two
            # duplicate detections of the parent needs no new link at all.
            merge_kw = {k: kw[k] for k in MERGE_KEYS}
            edges, mstats = edge_pp.merge_duplicate_parents(
                coords, edges, max_dup_um=merge_um, require_mutual=merge_mutual,
                return_stats=True, **merge_kw)
            stats.update({f"merge_{k}": v for k, v in mstats.items()})
        edges, dstats = edge_pp.propose_divisions(
            coords, edges, cand, return_stats=True, **kw)
        stats.update(dstats)
        # Nothing added: the graph is the baseline's, so the caller reuses the
        # baseline row rather than paying for an identical evaluation.
        if (stats["added"] == 0 and stats.get("merge_merged", 0) == 0
                and stats.get("shadow_merged", 0) == 0
                and cfg.get("fork_threshold") is None):
            return stem, None, stats

    coords, edges = _finish(coords, edges)
    return stem, _score(stem, coords, edges), stats


def evaluate_config(cache_paths, cfg, baseline_rows, jobs, label):
    t0 = time.time()
    rows, stats_total, n_eval = [], {}, 0
    with Pool(jobs, initializer=_init_worker, initargs=(CLF_PATH,)) as pool:
        for stem, row, stats in pool.imap_unordered(
            run_stack, [(str(p), cfg) for p in cache_paths], chunksize=1
        ):
            if row is None:
                rows.append(baseline_rows[stem])
            else:
                rows.append(row)
                n_eval += 1
            for k, v in (stats or {}).items():
                stats_total[k] = stats_total.get(k, 0) + v

    groups, macro = macro_summarise(rows)
    out = {
        "label": label, "cv_score": macro["score"], "rows": rows,
        "per_embryo": {e: summarise([r for r in rows if r["embryo"] == e])
                       for e in sorted({r["embryo"] for r in rows})},
        "stats": stats_total, "seconds": time.time() - t0, "n_rescored": n_eval,
    }
    return out


def report(res, base=None):
    cv = res["cv_score"]
    delta = f"  ({cv - base['cv_score']:+.4f})" if base else ""
    print(f"\n{res['label']:<44s} cv_score={cv:.4f}{delta}"
          f"   [{res['seconds']:.0f}s, {res['n_rescored']} rescored]")
    for emb, s in res["per_embryo"].items():
        b = base["per_embryo"][emb] if base else None
        d = f" ({s['score'] - b['score']:+.4f})" if b else ""
        print(f"    {emb}: score={s['score']:.4f}{d}  adj={s['adj_edge_jaccard']:.4f}"
              f"  divJ={s['division_jaccard']:.3f}"
              f"  div tp/fp/fn={s['division_tp']}/{s['division_fp']}/{s['division_fn']}"
              f"  edge fp={s['edge_fp']:.0f}")
    st = res["stats"]
    if st:
        print(f"    added={st.get('added', 0)}  proposals={st.get('proposals', 0)}"
              f"  rejected: prob={st.get('prob_rejected', 0)}"
              f" dist={st.get('distance_rejected', 0)}"
              f" mutualNN={st.get('mutual_nn_rejected', 0)}"
              f" diverge={st.get('divergence_rejected', 0)}"
              f" sym={st.get('symmetry_rejected', 0)}"
              f" [noGC={st.get('divergence_no_grandchild', 0)}"
              f" tooClose={st.get('divergence_too_close', 0)}]")
        if st.get("shadow_merged"):
            print(f"    shadow forks merged={st['shadow_merged']}  "
                  f"(pairs={st.get('shadow_shadow_pairs', 0)}, "
                  f"too_short={st.get('shadow_too_short', 0)}, "
                  f"no_split={st.get('shadow_no_split', 0)}, "
                  f"sister={st.get('shadow_sister_rejected', 0)})")
        if st.get("merge_merged"):
            print(f"    merged duplicate parents={st['merge_merged']}  "
                  f"(pairs={st.get('merge_pairs_considered', 0)}, "
                  f"rejected: sister={st.get('merge_sister_rejected', 0)} "
                  f"parent={st.get('merge_parent_rejected', 0)} "
                  f"diverge={st.get('merge_divergence_rejected', 0)} "
                  f"sym={st.get('merge_symmetry_rejected', 0)})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", required=True)
    p.add_argument("--cache_dir", default="../output/pp_experiments/div_cache")
    p.add_argument("--configs", default="base",
                   help="comma-joined [key=value,...] specs, or 'base'")
    p.add_argument("--embryos", default=None, help="comma-separated subset")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--out", default=None, help="write results json here")
    p.add_argument("--clf_scores", default=None,
                   help="div_classifier.py's out-of-fold scores npz")
    args = p.parse_args()

    global CLF_PATH
    CLF_PATH = args.clf_scores
    paths = sorted(Path(args.preds_dir).glob("*.npz"))
    if args.embryos:
        keep = tuple(args.embryos.split(","))
        paths = [q for q in paths if q.name.startswith(keep)]
    paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no npz under {args.preds_dir}")

    cache_dir = Path(args.cache_dir) / Path(args.preds_dir).name
    print(f"-> {len(paths)} stacks from {args.preds_dir}")
    print(f"-> cache {cache_dir}")
    t0 = time.time()
    with Pool(args.jobs) as pool:
        built = [s for _, s in pool.imap_unordered(
            build_cache, [(str(q), str(cache_dir)) for q in paths], chunksize=1)]
    print(f"-> base chain: {built.count('built')} built, "
          f"{built.count('cached')} reused ({time.time() - t0:.0f}s)")

    cache_paths = [cache_dir / f"{q.stem}.npz" for q in paths]

    baseline = evaluate_config(cache_paths, None, {}, args.jobs, "baseline (no divisions)")
    baseline_rows = {r["sample_name"]: r for r in baseline["rows"]}
    report(baseline)

    results = [baseline]
    specs = [s for s in _split_specs(args.configs) if s]
    for spec in specs:
        cfg = parse_config(spec)
        if cfg is not None:
            cfg["clf_path"] = CLF_PATH
        res = evaluate_config(cache_paths, cfg, baseline_rows, args.jobs, spec)
        report(res, baseline)
        results.append(res)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "rows"} for r in results],
            indent=2, default=float))
        print(f"\n-> {args.out}")

    if len(results) > 1:
        print(f"\n{'=' * 78}\nRANKED\n{'=' * 78}")
        for r in sorted(results, key=lambda r: -r["cv_score"]):
            print(f"  {r['label']:<50s} {r['cv_score']:.4f} "
                  f"({r['cv_score'] - baseline['cv_score']:+.4f})")


def _split_specs(text):
    """Split on commas that are outside brackets."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return out


if __name__ == "__main__":
    main()
