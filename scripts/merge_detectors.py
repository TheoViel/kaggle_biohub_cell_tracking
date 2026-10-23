#!/usr/bin/env python
"""Ensemble two detectors by appending the detections the base one missed.

Node set, per frame:

    N = BASE(th_base)  U  { a in ADD(th_add) : dist(a, BASE) > d_add }

Distances are physical (``params.SCALE``). ``d_add`` should not go below the
NMS radius the predictions were made at: appending a node closer than that
recreates exactly the pair the base model's own NMS removed.

The node set the shipped edge model was trained on (v4):

    python scripts/merge_detectors.py \
        --base logs/2026-09-13/1/preds_r6_0.30 \
        --add logs/2026-09-24/5/preds_r6_0.30 \
        --cfg 0.45 0.40 7 \
        --out ensemble/v4_2026-09-13_1_2026-09-24_5_append_b0.45_a0.40_d7

Every per-node array present in both sources (node_probs, annot_probs,
coords_f, ...) is carried over from whichever model contributed the node.
``edges`` is written empty: main_edge.py rebuilds the candidates.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from params import SCALE  # noqa: E402

SC = np.array(SCALE)


def merge_idx(ca, pa, cb, pb, th_a, th_b, d_add):
    """Rows of base and add that survive, and the permutation sorting them by frame."""
    b_idx = np.nonzero(pb >= th_b)[0]
    a_idx = np.nonzero(pa >= th_a)[0]

    if len(a_idx) and len(b_idx) and d_add > 0:
        b = cb[b_idx]
        t_b = np.round(b[:, 0]).astype(int)
        t_a = np.round(ca[a_idx][:, 0]).astype(int)
        keep = np.zeros(len(a_idx), bool)
        for f in np.unique(t_a):
            sel = t_a == f
            bb = b[t_b == f]
            if not len(bb):
                keep[sel] = True
                continue
            d, _ = cKDTree(bb[:, 1:4] * SC).query(ca[a_idx[sel]][:, 1:4] * SC, k=1)
            keep[sel] = d > d_add
        a_idx = a_idx[keep]

    t = np.concatenate([cb[b_idx][:, 0], ca[a_idx][:, 0]]) if len(b_idx) or len(a_idx) \
        else np.zeros(0)
    return b_idx, a_idx, np.argsort(t, kind="stable")


def node_arrays(db, da) -> list[str]:
    """Per-node array names present in both sources."""
    def per_node(d):
        n = len(d["coords"])
        return {k for k in d.files
                if k != "edges" and np.asarray(d[k]).shape[:1] == (n,)}
    return sorted(per_node(db) & per_node(da))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="preds folder of the base detector")
    p.add_argument("--add", required=True, help="preds folder whose extras get appended")
    p.add_argument("--cfg", nargs=3, type=float, required=True,
                   metavar=("TH_BASE", "TH_ADD", "D_ADD"))
    p.add_argument("--out", required=True, help="output folder")
    args = p.parse_args()
    th_base, th_add, d_add = args.cfg

    base, add, out_dir = Path(args.base), Path(args.add), Path(args.out)
    names = sorted(q.stem for q in base.glob("*.npz"))
    if not names:
        raise SystemExit(f"no predictions under {base}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_base = n_add = 0
    for n in names:
        db, da = np.load(base / f"{n}.npz"), np.load(add / f"{n}.npz")
        b_idx, a_idx, order = merge_idx(
            da["coords"].astype(float), da["node_probs"].astype(float),
            db["coords"].astype(float), db["node_probs"].astype(float),
            th_add, th_base, d_add,
        )
        out = {k: np.concatenate([db[k][b_idx], da[k][a_idx]])[order]
               for k in node_arrays(db, da)}
        out["edges"] = np.zeros((0, 4), dtype=db["edges"].dtype)
        np.savez_compressed(out_dir / f"{n}.npz", **out)
        n_base += len(b_idx)
        n_add += len(a_idx)

    settings = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": "python " + " ".join(sys.argv),
        # The run folders, which scripts/copy_to_dataset.py packages.
        "base_run": str(base.parent), "add_run": str(add.parent),
        "source_preds": base.name, "source_preds_add": add.name,
        "th_base": th_base, "th_add": th_add, "d_add_um": d_add,
        "n_samples": len(names), "n_from_base": n_base, "n_from_add": n_add,
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    total = n_base + n_add
    print(f"-> {len(names)} stacks written to {out_dir}")
    print(f"   {total} nodes: {n_base} from base, {n_add} from add "
          f"({n_add / max(total, 1):.1%})")


if __name__ == "__main__":
    main()
