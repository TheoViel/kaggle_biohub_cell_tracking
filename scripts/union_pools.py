#!/usr/bin/env python
"""Concatenate fork-candidate pools into one training set.

    python scripts/union_pools.py --out logs/fork_cls/pool28_union.npz \
        logs/fork_cls/pool28_ours.npz logs/fork_cls/pool28_edu.npz

The pools come from *different chains* over the same movies -- one harvests the
ILP's forks, the other runs the geometric proposer with the ILP off -- so a row
in one is not a row in the other even when both describe the same division. The
union is therefore a concatenation, not a merge: the classifier reads pixels at
the parent, and seeing the same parent twice under two proposal regimes is the
point, because at inference it will face whichever regime the chain is running.

Exact duplicates *are* dropped: identical `(stem, t, z, y, x, child_prob)` rows
are the same fork proposed identically by both chains, and keeping both would
weight it twice for no new information. Near-duplicates are kept -- two chains
placing a parent half a voxel apart is exactly the variation worth training on.

The label column is taken from whichever copy carries one: a fork labelled in
one pool and unlabelled in the other is the same event, and `-1` means "the
annotation does not say", not "no".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pools", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    stems, tables, cols, srcs = [], [], None, []
    for path in a.pools:
        z = np.load(path, allow_pickle=True)
        if cols is None:
            cols = z["columns"]
        elif list(z["columns"]) != list(cols):
            raise SystemExit(f"{path}: column layout differs from {a.pools[0]}")
        st = np.asarray([str(s) for s in z["stem"]])
        stems.append(st)
        tables.append(z["table"])
        srcs.append(np.full(len(st), Path(path).stem))
        lab = z["table"][:, 4].astype(int)
        print(f"  {Path(path).stem:<18} {len(st):>7,} forks   "
              f"{int((lab == 1).sum()):>4} pos  {int((lab == 0).sum()):>6,} neg  "
              f"{int((lab < 0).sum()):>7,} unlabelled")

    stem = np.concatenate(stems)
    table = np.concatenate(tables)
    src = np.concatenate(srcs)

    # Identity of a fork: stack, parent voxel, and the child edge probability
    # -- the same key `apply_fork_gate` and the sweeps match on.
    key = [(s, round(float(r[0])), round(float(r[1]), 3), round(float(r[2]), 3),
            round(float(r[3]), 3), round(float(r[7]), 6))
           for s, r in zip(stem, table)]
    seen: dict[tuple, int] = {}
    keep = np.zeros(len(table), bool)
    for i, k in enumerate(key):
        j = seen.get(k)
        if j is None:
            seen[k] = i
            keep[i] = True
        elif table[j, 4] < 0 <= table[i, 4]:
            # The duplicate carries a label the first copy lacked; take it.
            table[j, 4] = table[i, 4]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, stem=stem[keep], table=table[keep],
                        columns=cols, sources=src[keep])
    lab = table[keep][:, 4].astype(int)
    print(f"\n{keep.sum():,} forks -> {out}  "
          f"({len(table) - keep.sum():,} exact duplicates dropped)")
    print(f"  labelled positive : {(lab == 1).sum()}")
    print(f"  labelled negative : {(lab == 0).sum():,}")
    print(f"  unlabelled        : {(lab < 0).sum():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
