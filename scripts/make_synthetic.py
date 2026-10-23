#!/usr/bin/env python
"""Generate the synthetic sequences the detector is pretrained on.

    python scripts/make_synthetic.py --version 5

writes <input>/synthetic/v<version>/sequences/seq_*.npz (~50 MB each, ~48 GB for
the default 1000) plus metadata.json. The generator (src/data_gen) is calibrated
on real volumes of the first embryo: cell counts and tissue shell from a blob
detector, motion from the real lineages. Sequence ``i`` is seeded with
``500000 + i``, so a run is reproducible.

To compare a generated set with the real data, run from src/:

    python -m data_gen.calibrate --real 40 --synth ../input/synthetic/v5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data_gen import VOXEL_NATIVE, generate_sequence, real_stats  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", type=int, default=5)
    p.add_argument("--n_sequences", type=int, default=1000)
    p.add_argument("--seq_len", type=int, default=6, help="frames per sequence")
    p.add_argument("--division_rate", type=float, default=0.01,
                   help="per cell per frame; the real rate is ~0.0026, "
                        "over-sampled on purpose")
    p.add_argument("--n_calib_videos", type=int, default=10,
                   help="real videos measured for cell counts and shell geometry")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--input", default=str(REPO / "input"),
                   help="data root: reads <input>/train, writes <input>/synthetic")
    args = p.parse_args()

    root = Path(args.input).resolve()
    out = root / "synthetic" / f"v{args.version}"
    (out / "sequences").mkdir(parents=True, exist_ok=True)

    train_dir = root / "train"
    assert any(train_dir.glob("*.zarr")), f"no *.zarr under {train_dir}"
    by_embryo: dict[str, list[str]] = {}
    for v in sorted(q.stem for q in train_dir.glob("*.zarr")):
        by_embryo.setdefault(v.split("_")[0], []).append(v)
    embryo = sorted(by_embryo)[0]
    calib = by_embryo[embryo][: args.n_calib_videos]
    counts, shell = real_stats(train_dir, calib)
    print(f"calibrated on {len(calib)} videos of {embryo}: "
          f"{counts.min():.0f}-{counts.max():.0f} cells/frame, "
          f"shell {(shell or {}).get('kind')}")

    sequences = Parallel(n_jobs=args.jobs)(
        delayed(generate_sequence)(
            i, counts, shell, out, n_frames=args.seq_len,
            division_rate=args.division_rate,
        )
        for i in tqdm(range(args.n_sequences), desc="sequences")
    )
    sequences = sorted((s for s in sequences if s is not None), key=lambda r: r["index"])

    n_nodes = sum(s["n_nodes"] for s in sequences)
    n_div = sum(s["n_divisions"] for s in sequences)
    metadata = {
        "version": args.version,
        "n_sequences": len(sequences),
        "seq_len": args.seq_len,
        "total_gb": sum(s["bytes"] for s in sequences) / 1024**3,
        "total_nodes": n_nodes,
        "total_divisions": n_div,
        "division_rate": n_div / max(n_nodes, 1),
        "voxel_um": VOXEL_NATIVE.tolist(),
        "calibration_videos": calib,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    print(json.dumps(metadata, indent=2, default=float))


if __name__ == "__main__":
    main()
