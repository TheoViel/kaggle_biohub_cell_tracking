#!/usr/bin/env python3
"""Collect the trained models the kernel loads into dataset/, for Kaggle upload.

Given an edge run, its args.json names the node set (and through the merge's
settings.json, the detector runs behind it) and the NETr run behind its
embeddings. Each run's JSON files, logs.txt and fold_*/ckpt_final.pt are copied
to

    dataset/tracking_<date>_<exp>/
    dataset/contrastive_<date>_<exp>/
    dataset/edge_<date>_<exp>/
    dataset/division_<name>/          (one per --division)

Usage:
    python scripts/copy_to_dataset.py logs/edge_run/2026-09-28/0 \
        --division pool28_ours=logs/division/2026-09-28/0 \
        --division pool28_edu=logs/division/2026-09-28/1 \
        --division pool28_union=logs/division/2026-09-28/2
    python scripts/copy_to_dataset.py --dry-run logs/edge_run/2026-09-28/0

A detector without an args.json (the teammate's predictions) is reported and
skipped: its weights ship as their own Kaggle dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
# The same log root as the training code (src/params.py).
LOGS = (SRC / os.environ.get("BIOHUB_LOGS", "../logs/")).resolve()


def from_src(path: str) -> Path:
    """A path as written in an args.json, which is relative to src/."""
    p = Path(path)
    return p if p.is_absolute() else (SRC / p).resolve()


def run_tag(run: Path) -> str:
    """logs/<...>/2026-09-26/1 -> 2026-09-26_1."""
    return f"{run.parent.name}_{run.name}"


def detector_runs(nodes_folder: Path) -> list[Path]:
    """The detector runs behind a node set: a merge folder or a preds_* folder."""
    settings = nodes_folder / "settings.json"
    if not settings.exists():
        return [nodes_folder.parent]
    cfg = json.loads(settings.read_text())
    # merge_detectors.py records the run folders as given on its command line,
    # from the repository root.
    return [p if p.is_absolute() else (ROOT / p).resolve()
            for p in (Path(cfg[k]) for k in ("base_run", "add_run"))]


def contrastive_run(embeddings_key: str) -> Path:
    """embeddings_contrastive_2026-09-26_0 -> <logs>/contrastive/2026-09-26/0."""
    m = re.fullmatch(r"embeddings_contrastive_(\d{4}-\d{2}-\d{2})_(\d+)", embeddings_key)
    if not m:
        raise ValueError(f"cannot parse a contrastive run from {embeddings_key!r}")
    return LOGS / "contrastive" / m.group(1) / m.group(2)


def copy_run(src: Path, dst: Path, dry_run: bool) -> None:
    """Root JSON files, logs.txt and every fold_*/ckpt_final.pt."""
    if not (src / "args.json").is_file():
        raise FileNotFoundError(f"no args.json in {src}")
    files = sorted(src.glob("*.json")) + [src / "logs.txt"]
    files += sorted(src.glob("fold_*/ckpt_final.pt"))
    if not (src / "fold_fullfit" / "ckpt_final.pt").is_file():
        print(f"  !! {src} has no fold_fullfit/ckpt_final.pt -- the kernel cannot load it")
    print(f"  {src} -> {dst}")
    for f in files:
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        print(f"    {rel}")
        if not dry_run:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / rel)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("edge_run", help="edge run folder, e.g. logs/edge_run/2026-09-28/0")
    p.add_argument("--division", action="append", default=[], metavar="NAME=RUN",
                   help="a division classifier run, copied to division_<NAME>")
    p.add_argument("--dest", default=str(ROOT / "dataset"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    dest = Path(args.dest).resolve()
    edge = Path(args.edge_run).resolve()
    cfg = json.loads((edge / "args.json").read_text())

    for run in detector_runs(from_src(cfg["nodes_folder"])):
        if (run / "args.json").is_file():
            copy_run(run, dest / f"tracking_{run_tag(run)}", args.dry_run)
        else:
            print(f"  {run}: external detector, not copied")
    ctr = contrastive_run(cfg["embeddings_key"])
    copy_run(ctr, dest / f"contrastive_{run_tag(ctr)}", args.dry_run)
    copy_run(edge, dest / f"edge_{run_tag(edge)}", args.dry_run)
    for spec in args.division:
        name, run = spec.split("=", 1)
        copy_run(Path(run).resolve(), dest / f"division_{name}", args.dry_run)

    print("\nDry run, nothing written." if args.dry_run else f"\nCopied to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
