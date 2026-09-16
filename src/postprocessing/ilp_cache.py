"""Freeze the ILP's node filter to disk, so the rest of the chain can be swept.

    cd src/postprocessing
    python ilp_cache.py --preds_dir ../logs/edge_run/2026-09-15/0/preds_0.45 \
                        --out ../logs/edge_run/2026-09-15/0/ilp_d0.75_t0.4 \
                        --ilp_disappearance 0.75 --ilp_threshold 0.4

Paths are relative to ``src/``: ``common`` chdirs there, as the other scripts
here do.

`main_edge_postprocess.py --ilp` runs the solver and then the whole usual chain
over what survives -- `assign -> recover -> divisions -> prune -> linefit`. The
solve is 9-16 s a stack against about 1 s for everything after it, so sweeping
any of those knobs with `--ilp` on pays for the solver again on every arm, for
an answer that never changes: the ILP is upstream of all of them and does not
depend on one.

So run it once and write what it produced. `ilp_filter_nodes` returns
``(coords, edges, candidates)`` renumbered onto the nodes it kept -- exactly the
three arrays a predictions npz holds -- so the frozen output *is* a predictions
folder, and every tool that reads one works on it unchanged:

    python tune_edge_pp.py --preds_dir <the frozen folder> --knobs min_track,...
    python main_edge_postprocess.py --preds_dir <the frozen folder>

with `ilp` left off, because it has already happened. A sweep then costs the
1 s a stack it should.

The knobs worth turning afterwards are the ones whose tuning assumed a graph the
solver had not touched. `recover_*` and `min_track` both exist to delete or
repair what the detector got wrong, which is the same job the node filter has
just done -- `main_edge_postprocess.py`'s own DEFAULTS block records the two
being substitutes, the filter being worth +0.0091 before a `recover` retune and
+0.0002 after it. And the division gates are fitted to a node set the filter has
now thinned, which is why the comment there insists on refitting divisions to
the ILP graph before comparing: the filter drops dividing nodes along with
spurious ones, so a fork the gates were tuned to catch may no longer have two
children to catch it by.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: F401,E402  (chdir to src/ and set the path)

import edge_pp  # noqa: E402
import main_edge_postprocess as MEP  # noqa: E402

warnings.filterwarnings("ignore")

ILP_KEYS = ("ilp_threshold", "ilp_edge_weight", "ilp_appearance",
            "ilp_disappearance", "ilp_division", "ilp_keep_edges")


def _job(task):
    path, out_dir, cfg = task
    path = Path(path)
    coords, edges, cand = edge_pp.load_pred(path)
    if not len(cand):
        raise SystemExit(f"{path} carries no candidates")
    n_before = len(coords)
    coords, edges, cand = MEP.ilp_filter_nodes(coords, edges, cand, cfg)
    np.savez_compressed(
        Path(out_dir) / path.name,
        # Same dtypes a predictions npz uses, so the frozen folder is
        # indistinguishable from one main_edge_predict.py wrote.
        coords=np.round(np.asarray(coords, np.float64)).astype(np.int16),
        edges=np.asarray(edges, np.float64).reshape(-1, 4),
        candidates=np.asarray(cand, np.float64).reshape(-1, 5),
    )
    return path.stem, n_before, len(coords), len(cand)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds_dir", required=True,
                   help="the npz main_edge_predict.py wrote, with candidates")
    p.add_argument("--out", required=True, help="where to write the frozen folder")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    for k in ILP_KEYS:
        p.add_argument(f"--{k}", type=type(MEP.DEFAULTS[k]), default=MEP.DEFAULTS[k])
    a = p.parse_args()

    cfg = dict(MEP.DEFAULTS)
    cfg.update({k: getattr(a, k) for k in ILP_KEYS})
    cfg["ilp"] = 1

    d, out = Path(a.preds_dir), Path(a.out)
    paths = sorted(d.glob("*.npz"))[: a.limit]
    if not paths:
        raise SystemExit(f"no npz under {d}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"-> {len(paths)} stacks from {d}")
    print("   " + "  ".join(f"{k}={cfg[k]}" for k in ILP_KEYS))
    print(f"-> writing {out}")

    with Pool(a.jobs) as pool:
        res = list(pool.imap_unordered(
            _job, [(str(x), str(out), cfg) for x in paths], chunksize=1))

    before = sum(r[1] for r in res)
    after = sum(r[2] for r in res)
    # Recorded beside the folder so a sweep over it can say what it was swept on.
    (out / "ilp_config.json").write_text(json.dumps(
        {**{k: cfg[k] for k in ILP_KEYS}, "preds_dir": str(d),
         "n_stacks": len(res), "nodes_before": before, "nodes_after": after},
        indent=2))
    print(f"-> nodes {before} -> {after}  ({after / before:.1%} kept)")
    print(f"\nnow sweep the rest of the chain over it, with the ILP already done:\n"
          f"  python tune_edge_pp.py --preds_dir {out} --knobs "
          f"min_track,recover_max_gap,recover_max_residual,recover_min_dup")


if __name__ == "__main__":
    main()
