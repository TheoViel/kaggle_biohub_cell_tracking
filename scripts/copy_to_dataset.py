#!/usr/bin/env python3
"""Copy a full pipeline (tracking, contrastive, edge) to dataset/ for Kaggle upload.

Given an edge run id, read its args.json to find the linked tracking and
contrastive runs, then copy each run's JSON files, logs.txt, and
fold_*/ckpt_final.pt weights into dataset/ using the naming convention:

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


def parse_tracking_runs(nodes_folder: str) -> list[tuple[str, str, Path]]:
    """Every tracking run behind a nodes_folder, as (date, exp, run path).

    A single-detector folder is ``../logs/<date>/<exp>/preds_*`` and yields one
    run. An ensemble folder is ``../ensemble/<tag>/`` and yields *two* -- the
    blend's settings.json records which detectors were fused, and both have to
    reach Kaggle or the notebook cannot rebuild the node set at test time.
    """
    text = nodes_folder.replace("\\", "/").rstrip("/")
    match = re.search(r"logs/(\d{4}-\d{2}-\d{2})/(\d+)/", text)
    if match:
        date, exp = match.group(1), match.group(2)
        return [(date, exp, LOGS / date / exp)]

    if "/ensemble/" in text or text.startswith("ensemble/"):
        tag = text.split("ensemble/")[-1]
        settings = ROOT / "ensemble" / tag / "settings.json"
        if not settings.exists():
            raise ValueError(
                f"{nodes_folder!r} is an ensemble folder but {settings} is missing, "
                "so the detectors behind it cannot be identified"
            )
        cfg = json.loads(settings.read_text())
        # append blends name base_run/add_run; consensus blends model_a/model_b.
        runs = [cfg.get(k) for k in ("base_run", "add_run", "model_a", "model_b")]
        out = []
        for r in runs:
            if not r:
                continue
            m = re.search(r"logs/(\d{4}-\d{2}-\d{2})/(\d+)", r.replace("\\", "/"))
            if not m:
                # A partner from outside this repo -- `logs/eduardo_current_mean3`
                # is a shim over output/eduardo_predictions, not a training run,
                # so there are no fold_*/ckpt_final.pt here to copy. Its weights
                # ship as their own Kaggle dataset and the notebook mounts them
                # directly; copying anything for it would be inventing a run
                # folder that does not exist. Reported, not raised: an ensemble
                # with an external member is a valid pipeline, and refusing to
                # package the rest of it because one member lives elsewhere was
                # the bug.
                out.append((None, r.rstrip("/").split("/")[-1], None))
                continue
            key = (m.group(1), m.group(2), LOGS / m.group(1) / m.group(2))
            if key not in out:
                out.append(key)
        if not any(d for d, _, _ in out):
            raise ValueError(f"{settings} names no detector runs in this repo")
        return out

    raise ValueError(f"Cannot parse tracking run from nodes_folder: {nodes_folder!r}")


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
    """Copy root JSON/log files and all fold_*/ checkpoints from a training run."""
    dst = dest_root / dst_name
    if not src.is_dir():
        raise FileNotFoundError(f"Source run not found: {src}")
    if not (src / "args.json").is_file():
        raise FileNotFoundError(f"No args.json in {src}")

    folds = fold_dirs(src)
    if not folds:
        raise FileNotFoundError(f"No fold_* directories in {src}")

    print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    metadata_files = sorted(src.glob("*.json"))
    log_file = src / "logs.txt"
    if dry_run:
        for path in metadata_files:
            print(f"    {path.name}")
        print(f"    logs.txt" if log_file.is_file() else "    missing logs.txt")
        for fold in folds:
            ckpt = fold / "ckpt_final.pt"
            status = "ckpt_final.pt" if ckpt.is_file() else "missing ckpt_final.pt"
            print(f"    {fold.name}: {status}")
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    for path in metadata_files:
        shutil.copy2(path, dst / path.name)
        print(f"    copied {path.name}")
    if log_file.is_file():
        shutil.copy2(log_file, dst / log_file.name)
        print("    copied logs.txt")
    else:
        print("    ! no logs.txt, skipping")

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

    tracking_runs = parse_tracking_runs(nodes_folder)
    ct_date, ct_exp, contrastive_src = parse_contrastive_from_key(embeddings_key)
    edge_date, edge_exp = parse_edge_run_id(edge_src)

    external = [name for d, name, _ in tracking_runs if d is None]
    tracking_runs = [t for t in tracking_runs if t[0] is not None]

    print("Pipeline models:")
    for tr_date, tr_exp, src in tracking_runs:
        missing = "" if (src / "fold_fullfit" / "ckpt_final.pt").exists() else \
            "   !! no fold_fullfit/ckpt_final.pt -- inference cannot load it"
        print(f"  tracking:    tracking_{tr_date}_{tr_exp}{missing}")
    for name in external:
        print(f"  tracking:    {name}   (external partner -- NOT copied; its "
              f"weights ship as their own dataset)")
    print(f"  contrastive: contrastive_{ct_date}_{ct_exp}")
    print(f"  edge:        edge_{edge_date}_{edge_exp}")
    n_det = len(tracking_runs) + len(external)
    if n_det > 1:
        print(f"  (ensemble node set -- all {n_det} detectors are needed)")
    print()

    for tr_date, tr_exp, src in tracking_runs:
        copy_run(src, f"tracking_{tr_date}_{tr_exp}", dest_root, args.dry_run)
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
