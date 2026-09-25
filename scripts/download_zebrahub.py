"""Download one embryo (ome-zarr) + associated tracks from the public zebrahub bucket."""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def fetch_to(url: str, dest: str, retries: int = 5) -> int:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    last_err = None
    for attempt in range(retries):
        try:
            data = fetch(url)
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return len(data)
        except urllib.error.HTTPError as e:
            # A chunk that is all fill_value is simply not written, so a 404 is
            # the array being sparse, not an error: zarr reads the gap back as
            # fill_value whether or not the file is on disk. Retrying one costs
            # ~20s of backoff for nothing.
            if e.code == 404:
                return 0
            last_err = e
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"FAILED {url}: {last_err}", file=sys.stderr)
    return -1


def chunk_counts(shape, chunks):
    return [math.ceil(s / c) for s, c in zip(shape, chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embryo", default="ZSNS001_tail")
    ap.add_argument("--out", default="/data/kaggle_biohub_cell_tracking/zebrahub_raw")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--levels", default="0", help="comma-separated pyramid levels to fetch, e.g. '0' or '0,1,2'")
    args = ap.parse_args()
    levels = set(args.levels.split(","))

    name = args.embryo
    out_dir = os.path.join(args.out, name)
    zarr_dir = os.path.join(out_dir, f"{name}.ome.zarr")
    os.makedirs(out_dir, exist_ok=True)

    # metadata + tracks
    for fname in [f"{name}.json", f"{name}_tracks.csv"]:
        url = f"{BASE}/{fname}"
        dest = os.path.join(out_dir, fname)
        print(f"downloading {fname} ...")
        n = fetch_to(url, dest)
        print(f"  {fname}: {n} bytes" if n >= 0 else f"  {fname}: FAILED")

    # zarr group-level metadata
    url = f"{BASE}/{name}.ome.zarr/.zgroup"
    fetch_to(url, os.path.join(zarr_dir, ".zgroup"))

    zattrs = json.loads(fetch(f"{BASE}/{name}.ome.zarr/.zattrs"))
    datasets = zattrs["multiscales"][0]["datasets"]
    # only keep metadata for the levels actually fetched, so the store is self-consistent
    zattrs["multiscales"][0]["datasets"] = [d for d in datasets if d["path"] in levels]
    os.makedirs(zarr_dir, exist_ok=True)
    with open(os.path.join(zarr_dir, ".zattrs"), "w") as f:
        json.dump(zattrs, f)

    tasks = []
    for ds in datasets:
        level = ds["path"]
        if level not in levels:
            continue
        zarray_url = f"{BASE}/{name}.ome.zarr/{level}/.zarray"
        zarray = json.loads(fetch(zarray_url))
        dest = os.path.join(zarr_dir, level, ".zarray")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            json.dump(zarray, f)

        shape = zarray["shape"]
        chunks = zarray["chunks"]
        counts = chunk_counts(shape, chunks)
        total = 1
        for c in counts:
            total *= c
        print(f"level {level}: shape={shape} chunks={chunks} -> {total} chunk files")

        def gen_indices(counts):
            idx = [0] * len(counts)
            while True:
                yield tuple(idx)
                for d in range(len(counts) - 1, -1, -1):
                    idx[d] += 1
                    if idx[d] < counts[d]:
                        break
                    idx[d] = 0
                else:
                    return

        for idx in gen_indices(counts):
            key = "/".join(str(i) for i in idx)
            url = f"{BASE}/{name}.ome.zarr/{level}/{key}"
            dest = os.path.join(zarr_dir, level, *[str(i) for i in idx])
            tasks.append((url, dest))

    print(f"total chunk files to fetch: {len(tasks)}")
    done = 0
    failed = 0
    total_bytes = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_to, u, d): (u, d) for u, d in tasks}
        for fut in cf.as_completed(futs):
            n = fut.result()
            done += 1
            if n < 0:
                failed += 1
            else:
                total_bytes += n
            if done % 500 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                print(f"[{done}/{len(tasks)}] failed={failed} bytes={total_bytes/1e9:.2f}GB elapsed={elapsed:.0f}s", flush=True)

    print(f"DONE embryo={name} failed={failed} total_bytes={total_bytes}")


if __name__ == "__main__":
    main()
