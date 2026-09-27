#!/usr/bin/env python
"""Score a folder of post-processed npz against the annotation.

    python scripts/score_gated_folder.py /tmp/fk_44b6_gated /tmp/fk_6bba_gated

`main_edge_postprocess.py` scores what it produces, but
`division_cls/apply_fork_gate.py` writes a folder and stops -- so the gated
output has no number attached to it. This reads any number of folders as one
combined set and reports the same macro `multi_hypothesis.score_one` /
`macro_summarise` pair every other arm in this repo is reported with, so the
result is comparable to `compare_division_arms.py`'s table rather than merely
similar to it.

Several folders are accepted because an out-of-fold gate has to be run once
per embryo -- `apply_fork_gate.py --fold` applies to a whole folder -- and the
macro must still be taken over both at once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "postprocessing"))

import multi_hypothesis as M  # noqa: E402
from tracking_cellmot.metrics import macro_summarise, summarise  # noqa: E402


def main(dirs):
    rows = []
    for d in dirs:
        for q in sorted(Path(d).glob("*.npz")):
            z = np.load(q)
            rows.append(M.score_one(q.stem, z["coords"], z["edges"]))
    if not rows:
        raise SystemExit(f"no npz under {dirs}")

    _, macro = macro_summarise(rows)
    print(f"{len(rows)} stacks from {len(dirs)} folder(s)\n")
    print(f"  cv_score         : {macro['score']:.4f}")
    print(f"  adj_edge_jaccard : {macro['adj_edge_jaccard']:.4f}")
    print(f"  edge_jaccard     : {macro['edge_jaccard']:.4f}")
    print(f"  division_jaccard : {macro['division_jaccard']:.4f}")
    print(f"  node_recall      : {macro['node_recall']:.4f}\n")
    for emb in sorted({r["embryo"] for r in rows}):
        s = summarise([r for r in rows if r["embryo"] == emb])
        print(f"  {emb}: score={s['score']:.4f}  adj={s['adj_edge_jaccard']:.4f}"
              f"  divJ={s['division_jaccard']:.4f}"
              f"  tp/fp/fn={s['division_tp']}/{s['division_fp']}/{s['division_fn']}")


if __name__ == "__main__":
    main(sys.argv[1:] or sys.exit(__doc__))
