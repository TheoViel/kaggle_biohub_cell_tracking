#!/usr/bin/env python
"""Download the Zebrahub single-objective release into input/ext/.

The host throttles hard per connection (~15 kB/s measured), so everything here
is parallel: large files are split into byte ranges fetched concurrently, and
zarr trees are walked with a thread pool over their chunk files. Aggregate
throughput scales with --workers until the local link saturates.

Every transfer is resumable. A file is skipped when its size already matches
the server's Content-Length, so re-running after an interruption costs one HEAD
per file and nothing else.

    # metadata + track tables only (~2.4 GB)
    python scripts/fetch_zebrahub.py --what tracks

    # one imaging dataset at a pyramid level (see --list for sizes)
    python scripts/fetch_zebrahub.py --what images --datasets ZSNS003 --level 2

    # what is on the server, and what it would cost
    python scripts/fetch_zebrahub.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective"
DEST = Path(__file__).resolve().parent.parent / "input" / "ext"

DATASETS = ["ZSNS001", "ZSNS001_tail", "ZSNS002", "ZSNS003", "ZSNS004", "ZSNS005"]

_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


def _open(url: str, headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)


def listing(url: str) -> list[dict]:
    """Caddy's JSON directory listing."""
    with _open(url.rstrip("/") + "/", {"Accept": "application/json"}) as r:
        return json.load(r)


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n is not None else None
    except urllib.error.HTTPError:
        return None


def _fetch_range(url: str, start: int, end: int, path: Path, attempts: int = 5) -> int:
    """Write [start, end] of url at the same offset in path. Returns bytes written."""
    for attempt in range(attempts):
        try:
            with _open(url, {"Range": f"bytes={start}-{end}"}, timeout=120) as r:
                data = r.read()
            with open(path, "r+b") as f:
                f.seek(start)
                f.write(data)
            return len(data)
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
            log(f"    retry {attempt + 1}/{attempts} on {path.name} [{start}-{end}]: {exc}")
    return 0


def fetch_file(url: str, path: Path, workers: int = 16, chunk: int = 8 << 20) -> int:
    """Download one file, splitting it across ``workers`` range requests.

    Returns bytes transferred; 0 means it was already complete.
    """
    size = remote_size(url)
    if size is not None and path.exists() and path.stat().st_size == size:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)

    # Small or unsized files are not worth splitting.
    if size is None or size <= chunk:
        with _open(url, timeout=120) as r:
            data = r.read()
        path.write_bytes(data)
        return len(data)

    with open(path, "wb") as f:  # preallocate so ranges can seek
        f.truncate(size)

    spans = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_range, url, s, e, path) for s, e in spans]
        for fut in as_completed(futures):
            total += fut.result()
    return total


def walk(url: str, rel: str = "") -> list[tuple[str, str, int]]:
    """Recursively enumerate (url, relative path, size) for every file under url."""
    out: list[tuple[str, str, int]] = []
    stack = [(url, rel)]
    while stack:
        u, r = stack.pop()
        try:
            entries = listing(u)
        except Exception as exc:  # noqa: BLE001
            log(f"  ! cannot list {u}: {exc}")
            continue
        for e in entries:
            name = e["name"].rstrip("/")
            child_rel = f"{r}/{name}" if r else name
            child_url = u.rstrip("/") + "/" + name
            if e["is_dir"]:
                stack.append((child_url, child_rel))
            else:
                out.append((child_url, child_rel, e["size"]))
    return out


def fetch_many(items: list[tuple[str, str, int]], dest: Path, workers: int) -> None:
    """Download a list of (url, relative path, size).

    The connection budget goes wherever the parallelism is. A zarr level is
    thousands of small chunk files, so it is spent across files; the track
    tables are a handful of large ones, so it is spent splitting each into byte
    ranges instead. Either way ``workers`` connections stay busy — which is the
    only thing that matters against a per-connection throttle.
    """
    moved = 0
    total = len(items)
    started = time.time()

    def rate():
        dt = max(time.time() - started, 1e-6)
        return f"{moved / 1e6:.1f} MB, {moved / dt / 1e6:.3f} MB/s"

    if total <= workers // 2:
        for i, (url, rel, _) in enumerate(items, 1):
            try:
                moved += fetch_file(url, dest / rel, workers=workers)
            except Exception as exc:  # noqa: BLE001
                log(f"  ! {rel}: {exc}")
            log(f"  {i}/{total} {rel} — {rate()}")
        return

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_file, u, dest / r, 1): r for u, r, _ in items}
        for fut in as_completed(futures):
            rel = futures[fut]
            done += 1
            try:
                moved += fut.result()
            except Exception as exc:  # noqa: BLE001
                log(f"  ! {rel}: {exc}")
            if done % 100 == 0 or done == total:
                log(f"  {done}/{total} files — {rate()}")


def cmd_list() -> None:
    import numpy as np

    log(f"{BASE}\n")
    entries = listing(BASE)
    files = [e for e in entries if not e["is_dir"]]
    log("Files at the root:")
    for e in sorted(files, key=lambda x: -x["size"]):
        log(f"  {e['name']:<28} {e['size'] / 1e6:10.1f} MB")
    log(f"  {'TOTAL':<28} {sum(e['size'] for e in files) / 1e9:10.2f} GB\n")

    log("Imaging pyramids (raw uint16 volume; on-disk is compressed):")
    for d in DATASETS:
        for lvl in range(6):
            try:
                with _open(f"{BASE}/{d}.ome.zarr/{lvl}/.zarray") as r:
                    a = json.load(r)
            except Exception:  # noqa: BLE001
                continue
            shape, chunks = a["shape"], a["chunks"]
            itemsize = np.dtype(a["dtype"]).itemsize
            raw = float(np.prod(shape, dtype=float)) * itemsize
            nchunks = float(np.prod([-(-s // c) for s, c in zip(shape, chunks)], dtype=float))
            log(f"  {d:<14} L{lvl}  {str(shape):<32} {raw / 1e12:7.2f} TB  {nchunks:>9,.0f} files")
    log("\nAt the measured ~0.3-2 MB/s aggregate, 1 TB is on the order of a week.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--what", choices=("tracks", "images", "benchmark"), default="tracks",
                   help="tracks = json + *_tracks.csv (~2.4 GB); images = ome.zarr "
                        "pyramids; benchmark = tracks_benchmark/")
    p.add_argument("--datasets", default=",".join(DATASETS),
                   help="comma-separated subset for --what images")
    p.add_argument("--level", type=int, default=2,
                   help="pyramid level for --what images; 0 is full resolution "
                        "(0.439 um XY, closest to the competition's 0.40625)")
    p.add_argument("--workers", type=int, default=32,
                   help="concurrent connections; the host throttles per connection, "
                        "so this is the throughput knob")
    p.add_argument("--dest", default=str(DEST))
    p.add_argument("--list", action="store_true", help="show what is on the server and exit")
    p.add_argument("--dry-run", action="store_true", help="enumerate but do not download")
    args = p.parse_args()

    if args.list:
        cmd_list()
        return

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.what == "tracks":
        items = [
            (f"{BASE}/{e['name']}", e["name"], e["size"])
            for e in listing(BASE)
            if not e["is_dir"]
        ]
    elif args.what == "benchmark":
        items = [
            (u, f"tracks_benchmark/{r}", s)
            for u, r, s in walk(f"{BASE}/tracks_benchmark")
        ]
    else:
        items = []
        for d in args.datasets.split(","):
            d = d.strip()
            if not d:
                continue
            root = f"{BASE}/{d}.ome.zarr"
            # Group metadata, then just the requested pyramid level.
            for meta in (".zattrs", ".zgroup"):
                items.append((f"{root}/{meta}", f"{d}.ome.zarr/{meta}", 0))
            items += [
                (u, f"{d}.ome.zarr/{args.level}/{r}", s)
                for u, r, s in walk(f"{root}/{args.level}")
            ]

    known = sum(s for _, _, s in items)
    log(f"-> {len(items)} files, {known / 1e9:.2f} GB known size, into {dest}")
    if args.dry_run:
        for u, r, s in items[:20]:
            log(f"   {r}  {s / 1e6:.2f} MB")
        log("   ...")
        return

    fetch_many(items, dest, workers=args.workers)
    log("done")


if __name__ == "__main__":
    sys.exit(main())
