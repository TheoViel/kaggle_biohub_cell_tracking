"""Build a division-classifier crop cache straight from the generator.

`synthetic.build_cache` reads `input/synthetic/v5` sequences off disk, which is
what it has to do -- those are 50 MB npz files rendered by a different pipeline
for a different purpose. `data_gen.division_samples` renders a
`(3, 32, 128, 128)` clip in about a second, so here the volumes never touch
disk: a worker generates a clip, cuts its crops, and throws the volume away.

The output is the same `{clips, labels}` npz `SyntheticClipDataset` already
reads, so nothing downstream changes.

```bash
python -m division_cls.gen_divisions --n 4000 --out ../input/synthetic/div_v1.npz
python -m division_cls.gen_divisions --verify 96      # measure, do not write
```
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data_gen.division_samples import generate_division_clip
from embeddings.patches import extract_patches


def _normalise(vols):
    """Per frame, as the real loader does.

    A per-crop range would erase how bright a cell is next to its neighbours,
    which is part of what tells a daughter from a small ordinary cell.
    """
    out = []
    for t in range(len(vols)):
        v = vols[t].astype(np.float32)
        lo, hi = np.percentile(v, [1.0, 99.99])
        out.append(np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0))
    return out


def _cut(norm, t, zyx, patch_z, patch_yx, n_frames):
    """One `(n_frames, patch_z, patch_yx, patch_yx)` clip centred on a cell."""
    half = n_frames // 2
    zc, y, x = (float(q) for q in zyx)
    clip = np.zeros((n_frames, patch_z, patch_yx, patch_yx), np.float32)
    for k, dt in enumerate(range(-half, half + 1)):
        tt = t + dt
        if tt < 0 or tt >= len(norm):
            continue
        v = norm[tt]
        z0 = int(round(zc)) - patch_z // 2
        lo, hi = max(0, z0), min(v.shape[0], z0 + patch_z)
        slab = np.zeros((patch_z, v.shape[1], v.shape[2]), np.float32)
        if lo < hi:
            slab[lo - z0:hi - z0] = v[lo:hi]
        clip[k] = extract_patches(
            slab, np.array([[patch_z // 2, y, x]], np.float32),
            patch_zyx=(patch_z, patch_yx, patch_yx), n_frames=1)[0, 0]
    return clip


def clip_samples(index, patch_yx=32, patch_z=3, n_frames=3, neg_per_pos=12,
                 seed_base=7_000_000):
    """`(clips, labels)` for one generated volume: the divider and its neighbours.

    The negatives are the *background cells of the same clip*, at the same
    frame. Same volume, same illumination, same regime, same depth -- so the
    only thing that distinguishes them is the answer. That is the same argument
    `dataset.py` makes for drawing real negatives from the dividing cells' own
    lineages, and it is why the generator returns a scene rather than one crop.
    """
    vols, nodes, edges, div = generate_division_clip(index, n_frames=n_frames,
                                                     seed_base=seed_base)
    if not len(div):
        return np.zeros((0, n_frames, patch_z, patch_yx, patch_yx), np.float32), \
            np.zeros(0, np.int8)
    norm = _normalise(vols)
    par = int(div[0])
    t = int(nodes[par, 0])

    rng = np.random.default_rng(seed_base + index)
    others = np.flatnonzero((nodes[:, 0] == t) & (np.arange(len(nodes)) != par))
    if len(others) > neg_per_pos:
        others = rng.choice(others, neg_per_pos, replace=False)

    rows = np.concatenate([[par], others]).astype(int)
    clips = np.stack([_cut(norm, t, nodes[r, 1:4], patch_z, patch_yx, n_frames)
                      for r in rows])
    labels = np.zeros(len(rows), np.int8)
    labels[0] = 1
    return clips, labels


def _worker(a):
    i, kw = a
    return clip_samples(i, **kw)


def build(out_path, n=4000, workers=32, seed_base=7_000_000, verbose=True, **kw):
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_worker, [(i, dict(kw, seed_base=seed_base))
                                    for i in range(n)], chunksize=4))
    X = np.concatenate([c for c, _ in out if len(c)])
    y = np.concatenate([l for _, l in out if len(l)])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, clips=X, labels=y)
    if verbose:
        pos = int(y.sum())
        print(f"-> {len(X)} crops ({pos} divisions, 1:{(len(y) - pos) / max(pos, 1):.0f}) "
              f"to {out_path}", flush=True)
    return out_path


# --------------------------------------------------------------------------
# Verification: does the generator reproduce the measured real statistics?
# --------------------------------------------------------------------------
# Real, on the 151 annotated divisions and 1 192 matched controls; see the
# `division_samples` docstring for how these were measured.
REAL = {"same cell @ t-1": (1.52, 0.188, 256), "parent @ t": (1.51, 0.244, 285),
        "daughter @ t+1": (1.54, 0.504, 213), "ordinary cell": (1.51, 0.253, 329)}


def shape_stats(c):
    """Elongation, bimodality and area of the cell in one crop.

    The crop is background-subtracted at its own 25th percentile and squared, so
    the moments follow the cell rather than its neighbours' halos. Kept
    byte-for-byte equivalent to the script that measured the real data --
    a verification that used a different estimator would verify nothing.
    """
    if c is None or c.size == 0:
        return None
    img = c.max(axis=0)
    img = np.clip(img - np.percentile(img, 25), 0, None)
    if img.max() <= 0:
        return None
    w = (img / img.max()) ** 2
    yy, xx = np.indices(img.shape)
    m = w.sum()
    cy, cx = (w * yy).sum() / m, (w * xx).sum() / m
    vyy = (w * (yy - cy) ** 2).sum() / m
    vxx = (w * (xx - cx) ** 2).sum() / m
    vxy = (w * (yy - cy) * (xx - cx)).sum() / m
    ev, evec = np.linalg.eigh(np.array([[vyy, vxy], [vxy, vxx]]))
    ev = np.maximum(ev, 1e-6)
    major = evec[:, 1]

    # Profile along the major axis: two kernels show a dip between two maxima.
    ts = np.linspace(-2.5 * np.sqrt(ev[1]), 2.5 * np.sqrt(ev[1]), 41)
    py = np.clip(np.round(cy + major[0] * ts).astype(int), 0, img.shape[0] - 1)
    px = np.clip(np.round(cx + major[1] * ts).astype(int), 0, img.shape[1] - 1)
    prof = img[py, px].astype(float)
    prof = prof / (prof.max() + 1e-9)
    mid = len(prof) // 2
    left, right = prof[:mid].max(), prof[mid:].max()
    dip = prof[max(0, mid - 6):mid + 7].min()
    return dict(elong=float(np.sqrt(ev[1] / ev[0])),
                bimod=float(max(0.0, (min(left, right) - dip) / (min(left, right) + 1e-9))),
                area=float((img > 0.35 * img.max()).sum()))


def _measure(i):
    from data_gen.constants import VOXEL_NATIVE
    vols, nodes, edges, div = generate_division_clip(i)
    norm = _normalise(vols)
    par = int(div[0])
    t = int(nodes[par, 0])
    H, PZ = 16, 1

    def crop(tt, zyx):
        if not (0 <= tt < len(norm)):
            return None
        z, y, x = [int(round(q)) for q in zyx]
        c = norm[tt][max(0, z - PZ):z + PZ + 1, max(0, y - H):y + H, max(0, x - H):x + H]
        return np.asarray(c, np.float32) if c.size else None

    rows = []

    def add(role, s):
        if s:
            s["role"] = role
            rows.append(s)

    add("parent @ t", shape_stats(crop(t, nodes[par, 1:4])))
    prev = edges[edges[:, 1] == par][:, 0]
    if len(prev):
        add("same cell @ t-1", shape_stats(crop(t - 1, nodes[prev[0], 1:4])))
    kids = edges[edges[:, 0] == par][:, 1]
    for c in kids[:2]:
        add("daughter @ t+1", shape_stats(crop(t + 1, nodes[c, 1:4])))
    rng = np.random.default_rng(i)
    for n in rng.permutation(np.flatnonzero(nodes[:, 0] == t)[1:])[:8]:
        add("ordinary cell", shape_stats(crop(t, nodes[n, 1:4])))

    sep = (float(np.linalg.norm((nodes[kids[0], 1:4] - nodes[kids[1], 1:4]) * VOXEL_NATIVE))
           if len(kids) == 2 else np.nan)
    return rows, sep, int((nodes[:, 0] == t).sum())


def verify(n=96, workers=32):
    """Re-measure the generator against the real table. Prints, returns nothing."""
    import pandas as pd
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_measure, range(n)))
    df = pd.DataFrame([r for o in out for r in o[0]])
    seps = np.array([o[1] for o in out], float)

    print(f"{'role':<18}{'n':>5}{'elong':>8}{'real':>7}{'bimod':>8}{'real':>7}"
          f"{'area':>7}{'real':>7}")
    for role, (e, b, a) in REAL.items():
        d = df[df.role == role]
        if not len(d):
            continue
        print(f"{role:<18}{len(d):>5}{d.elong.median():>8.2f}{e:>7.2f}"
              f"{d.bimod.median():>8.3f}{b:>7.3f}{d.area.median():>7.0f}{a:>7.0f}")
    print(f"\nsister separation  {np.nanmedian(seps):.1f} um   (real 10.6)")
    print(f"cells per frame    {np.median([o[2] for o in out]):.0f}   "
          f"(real 17-52 for this volume)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=4000, help="clips, one division each")
    p.add_argument("--out", default="../input/synthetic/div_v1.npz")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--patch_yx", type=int, default=32)
    p.add_argument("--patch_z", type=int, default=3)
    p.add_argument("--n_frames", type=int, default=3)
    p.add_argument("--neg_per_pos", type=int, default=12)
    p.add_argument("--seed_base", type=int, default=7_000_000)
    p.add_argument("--verify", type=int, default=0,
                   help="measure this many clips against the real table and exit")
    a = p.parse_args()
    if a.verify:
        verify(a.verify, a.workers)
        return
    build(a.out, n=a.n, workers=a.workers, seed_base=a.seed_base,
          patch_yx=a.patch_yx, patch_z=a.patch_z, n_frames=a.n_frames,
          neg_per_pos=a.neg_per_pos)


if __name__ == "__main__":
    main()
