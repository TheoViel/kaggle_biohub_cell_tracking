#!/usr/bin/env python3
"""Copy a full pipeline (tracking, contrastive, edge) to dataset/ for Kaggle upload.

Given an edge run id, read its args.json to find the linked tracking and
contrastive runs, then copy each run's config and fold_*/ckpt_final.pt
weights into dataset/ using the naming convention:

    tracking_<date>_<exp>/
    contrastive_<date>_<exp>/
    edge_<date>_<exp>/

where <date> and <exp> match the logs/ directory layout.

Usage:
    python scripts/copy_to_dataset.py 2026-09-09/0
    python scripts/copy_to_dataset.py edge_run/2026-09-09/0
    python scripts/copy_to_dataset.py --dry-run 2026-09-09/0
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
DEFAULT_DEST = ROOT / "dataset"


def resolve_edge_run(name: str) -> Path:
    """Resolve an edge run folder from several common name formats."""
    raw = name.strip().rstrip("/")
    path = Path(raw)
    if path.is_absolute() and path.is_dir():
        return path.resolve()

    candidates = [
        LOGS / "edge_run" / raw,
        LOGS / raw,
        ROOT / raw,
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "args.json").is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Edge run not found for {name!r}. "
        "Expected something like 2026-09-09/0 under logs/edge_run/."
    )


def parse_tracking_from_nodes_folder(nodes_folder: str) -> tuple[str, str, Path]:
    """../logs/2026-09-09/4/preds_r6_0.40/ -> date, exp, run path."""
    text = nodes_folder.replace("\\", "/").rstrip("/")
    match = re.search(r"logs/(\d{4}-\d{2}-\d{2})/(\d+)/", text)
    if not match:
        raise ValueError(f"Cannot parse tracking run from nodes_folder: {nodes_folder!r}")
    date, exp = match.group(1), match.group(2)
    return date, exp, LOGS / date / exp


def parse_contrastive_from_key(key: str) -> tuple[str, str, Path]:
    """embeddings_contrastive_2026-09-09_1 -> date, exp, run path."""
    match = re.fullmatch(r"embeddings_contrastive_(\d{4}-\d{2}-\d{2})_(\d+)", key)
    if not match:
        raise ValueError(f"Cannot parse contrastive run from embeddings_key: {key!r}")
    date, exp = match.group(1), match.group(2)
    return date, exp, LOGS / "contrastive" / date / exp


def parse_edge_run_id(edge_run: Path) -> tuple[str, str]:
    """logs/edge_run/2026-09-09/0 -> date, exp."""
    parts = edge_run.parts
    try:
        idx = parts.index("edge_run")
        return parts[idx + 1], parts[idx + 2]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot parse date/exp from edge run path: {edge_run}") from exc


def fold_dirs(run_dir: Path) -> list[Path]:
    return sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("fold_")
    )


def copy_run(src: Path, dst_name: str, dest_root: Path, dry_run: bool) -> Path:
    """Copy args.json and all fold_*/ checkpoints from a training run."""
    dst = dest_root / dst_name
    if not src.is_dir():
        raise FileNotFoundError(f"Source run not found: {src}")
    if not (src / "args.json").is_file():
        raise FileNotFoundError(f"No args.json in {src}")

    folds = fold_dirs(src)
    if not folds:
        raise FileNotFoundError(f"No fold_* directories in {src}")

    print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    if dry_run:
        for fold in folds:
            ckpt = fold / "ckpt_final.pt"
            status = "ckpt_final.pt" if ckpt.is_file() else "missing ckpt_final.pt"
            print(f"    {fold.name}: {status}")
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "args.json", dst / "args.json")

    for fold in folds:
        ckpt = fold / "ckpt_final.pt"
        if not ckpt.is_file():
            print(f"    ! no ckpt_final.pt in {fold.name}, skipping")
            continue
        out_fold = dst / fold.name
        out_fold.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, out_fold / ckpt.name)
        print(f"    copied {fold.name}/ckpt_final.pt")

    return dst


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "edge_run",
        help="Edge run id or path (e.g. 2026-09-09/0, edge_run/2026-09-09/0)",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=f"Destination root (default: {DEFAULT_DEST.relative_to(ROOT)}/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files",
    )
    args = parser.parse_args()

    dest_root = Path(args.dest).resolve()
    edge_src = resolve_edge_run(args.edge_run)
    cfg = json.loads((edge_src / "args.json").read_text())

    nodes_folder = cfg.get("nodes_folder")
    embeddings_key = cfg.get("embeddings_key")
    if embeddings_key is None:
        embeddings_key = (cfg.get("model_config") or {}).get("embeddings_key")

    if not nodes_folder:
        print("error: edge args.json has no nodes_folder", file=sys.stderr)
        return 1
    if not embeddings_key:
        print("error: edge args.json has no embeddings_key", file=sys.stderr)
        return 1

    tr_date, tr_exp, tracking_src = parse_tracking_from_nodes_folder(nodes_folder)
    ct_date, ct_exp, contrastive_src = parse_contrastive_from_key(embeddings_key)
    edge_date, edge_exp = parse_edge_run_id(edge_src)

    print("Pipeline models:")
    print(f"  tracking:    tracking_{tr_date}_{tr_exp}")
    print(f"  contrastive: contrastive_{ct_date}_{ct_exp}")
    print(f"  edge:        edge_{edge_date}_{edge_exp}")
    print()

    copy_run(tracking_src, f"tracking_{tr_date}_{tr_exp}", dest_root, args.dry_run)
    copy_run(contrastive_src, f"contrastive_{ct_date}_{ct_exp}", dest_root, args.dry_run)
    copy_run(edge_src, f"edge_{edge_date}_{edge_exp}", dest_root, args.dry_run)

    print()
    if args.dry_run:
        print("Dry run complete — no files written.")
    else:
        print(f"Copied to {dest_root.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
