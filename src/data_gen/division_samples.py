"""Small volumes built for one question: did this cell divide?

Separate from `sequences.py` on purpose. That generator makes long lineages in
full-size volumes, and a division is an incidental event inside one; this makes
a `(3, 32, 128, 128)` clip whose only content of interest is a single division
at the centre frame -- which is exactly the unit `division_cls` trains on. Small
volumes buy thousands of divisions for the cost of a handful of sequences.

**The appearance is measured, not invented.** The first attempt at synthetic
divisions (`sequences._lobe`) rendered the parent as two visible lobes at the
frame the label names, on the assumption that mitosis is already apparent there.
Measured over the 151 annotated divisions, against 1 192 ordinary cells from the
same stacks and frames, on the max-projected crop the classifier actually sees:

| role | n | elongation | bimodality | area |
|---|---|---|---|---|
| same cell at t-1 | 145 | 1.52 | 0.188 | 256 |
| **parent at t** | 151 | **1.51** | **0.244** | 285 |
| daughter at t+1 | 302 | 1.54 | **0.504** | 213 |
| **ordinary cell** | 1192 | **1.51** | **0.253** | 329 |

A dividing cell at its own division frame is indistinguishable from any other
cell on both statistics (elongation quartiles 1.26/1.51/1.77 against
1.30/1.51/1.80; bimodality over 0.15 for 73% of parents and 75% of ordinary
cells). What *is* distinctive is the frame after: bimodality doubles, because a
crop centred where the cell was now holds two resolved objects.

So a division here is a **temporal** event and the parent is rendered as an
ordinary cell. Teaching a bilobed parent taught a rule that is false on real
images, and mixing that into training cost the classifier ~0.2 AP at every
proportion tried. Two caveats on the measurement: it is two hand-built 2D
features on a max projection, so it can only say these two statistics do not
separate -- not that nothing does -- and it contradicts the premise the
generator was previously built on, which is the reason it is stated at length.

What comes out, measured the same way (96 clips, `sweep` below):

| role | elongation | bimodality | area |
|---|---|---|---|
| same cell at t-1 | 1.39 (1.52) | 0.226 (0.188) | 332 (256) |
| parent at t | 1.34 (1.51) | 0.228 (0.244) | 336 (285) |
| daughter at t+1 | 1.58 (1.54) | **0.423** (0.504) | 254 (213) |
| ordinary cell | 1.46 (1.51) | 0.228 (0.253) | 316 (329) |

Real in brackets. The property that matters holds: the parent at t sits on top
of the ordinary cell on both statistics, and the daughter at t+1 is the only row
that separates. Absolute bimodality at t+1 falls short of the real 0.504 and
does not move when the daughters are shrunk further (0.68 / 0.58 / 0.50 all give
0.42-0.45), so what limits it is crowding and the PSF rather than cell size --
left as is rather than chased with a knob that does not control it.

The geometry checks out too, on statistics nothing here was fitted to:

| | generated | real |
|---|---|---|
| \|sep\| p25/50/75 | 8.6/10.6/12.9 | 8.0/10.6/12.7 |
| \|dz\| p25/50/75 | 2.0/4.9/7.6 | 1.6/4.9/6.5 |
| \|dxy\| p25/50/75 | 5.7/8.4/10.5 | 6.0/8.4/10.8 |
| near/sep, far/sep | 0.45, 0.59 | 0.42, 0.67 |
| daughters inside the 3-slice crop | 50% | 52% |

Only `SISTER_UM` was fitted; the rest follow from splitting along a uniformly
random 3D axis, which the `dxy/|sep|` ratio (0.87 against 0.90) says is close
enough to what real divisions do. The last row is worth keeping because it looks
alarming when plotted -- in most clips *both* daughters sit outside the three z
slices, so the t+1 crop shows little -- and that is simply what real divisions
do at this z resolution. Note it must be measured on rounded z: real annotation
coordinates are integer voxels, and comparing them against continuous synthetic
ones makes the generator look wrong (36%) when it is not (50%).

Sister separation comes out at 10.5 um against the measured 10.6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_gen.acquisition import Palette, Regime, concat, reserved_um
from data_gen.constants import VOXEL_NATIVE
from data_gen.render import gen_volume_native
from data_gen.sequences import to_uint16

# The clip. 128 voxels of y/x is 52 um and 32 z slices likewise, so the box
# holds a division and enough neighbourhood to make it ambiguous.
SHAPE = (32, 128, 128)
N_FRAMES = 3

# Fitted to the 151 annotated divisions; same numbers as `sequences.SISTER_UM`.
SISTER_UM = 10.6
SISTER_SD = 3.1
SISTER_MIN_UM = 4.0

# Cells besides the dividing one. The real data runs 140-417 cells in a
# (64, 256, 256) frame and this box is an eighth of that volume, so 17-52.
# Packing more would score better on the table below -- crowding raises
# elongation and bimodality across every row -- and that is exactly why it is
# not done: the density is measured, and buying an appearance match with a
# density the data contradicts is the mistake this whole file exists to undo.
N_BACKGROUND = (20, 48)

# The knobs below were set by matching the measured table, not by taste;
# `scripts/verify` in `division_cls/gen_divisions.py` re-measures them.
#
# `PACK_FACTOR` scales the room `reserved_um` claims. Below 1 because
# `reserved_um` reserves the *enclosing sphere*, so an elongated nucleus is
# packed as if it were a ball of its longest axis and the scene comes out
# sparser and cleaner than the real one -- which showed as every synthetic row
# being too round and too unimodal, the controls included.
#
# `ELONGATE` / `STRETCH` set how many nuclei have their long axis in y/x and by
# how much. The stock `sample_aniso` elongates half of them along a *random*
# axis, so two thirds of those point mostly along z and vanish under the max
# projection; that alone left synthetic elongation at 1.27 against the real 1.51.
#
# `DAUGHTER_SCALE` shrinks the daughters past what `Palette.divide` draws. Real
# daughters threshold to 213 px against an ordinary cell's 329 and separate
# cleanly, and separation is what the classifier reads.
#
# `PARENT_ROUND` makes the dividing cell isotropic at t-1 and t. Real cells
# about to divide are *less* bimodal than ordinary ones (0.188 against 0.253),
# the direction mitotic rounding predicts. It is a weak effect on 145 cells and
# it does not show in elongation at all, so this is deliberately partial.
PACK_FACTOR = 0.45
RADIUS_SCALE = 0.75
ELONGATE = 0.90
STRETCH = (1.6, 2.4)
DAUGHTER_SCALE = 0.58
PARENT_ROUND = 0.6

STEP_UM = 1.86          # median real per-frame displacement
_LO = np.array([3.0, 8.0, 8.0])
_HI = np.array(SHAPE, float) - 1.0 - _LO


def _elongate(palette, rng, p=ELONGATE, stretch_range=STRETCH):
    """Turn a share of the nuclei's long axes into the y/x plane."""
    n = len(palette)
    pick = rng.random(n) < p
    ax = rng.integers(1, 3, n)                     # 1 or 2, never z
    stretch = rng.uniform(*stretch_range, n)
    aniso = palette.aniso.copy()
    aniso[pick] = np.clip(aniso[pick] / aniso[pick].max(-1, keepdims=True), 0.82, 1.2)
    aniso[np.flatnonzero(pick), ax[pick]] *= stretch[pick]
    palette.aniso[:] = aniso.astype(np.float32)
    return palette


def _place(n_want, rng, regime, palette, pack=PACK_FACTOR):
    """Positions for as many of `palette`'s cells as fit, cell 0 at the centre.

    The collision test uses each cell's *own* radius and anisotropy out of the
    palette, so the room reserved at placement is the room the renderer will
    use -- `gen_volume_native` makes the same argument for its own branches.
    """
    reserved = pack * np.array([float(reserved_um(palette.radius_um[k], palette.aniso[k]))
                                for k in range(len(palette))])
    centre = np.array(SHAPE, float) / 2 + rng.normal(0, 2.0, 3)
    keep, pos = [0], [np.clip(centre, _LO, _HI)]
    for k in range(1, len(palette)):
        if len(keep) >= n_want:
            break
        for _ in range(24):
            p = rng.uniform(_LO, _HI)
            d = np.linalg.norm((np.array(pos) - p) * VOXEL_NATIVE, axis=1)
            if np.all(d >= regime.collision_factor * (reserved[keep] + reserved[k])):
                keep.append(k)
                pos.append(p)
                break
    return np.array(pos, float), np.array(keep, int)


def generate_division_clip(index, n_frames=N_FRAMES, seed_base=7_000_000, divide=True,
                           pack=PACK_FACTOR, radius_scale=RADIUS_SCALE,
                           elongate=ELONGATE, daughter_scale=DAUGHTER_SCALE,
                           parent_round=PARENT_ROUND, n_background=N_BACKGROUND,
                           stretch=STRETCH):
    """One clip. Returns `(volumes, nodes, edges, division_rows)`.

    `nodes` is (N, 5) as t/z/y/x/track_id and `edges` (M, 2) indexes into it, so
    the arrays match what `sequences.py` writes and the same readers work.
    `division_rows` holds the row of the dividing parent -- the centre cell at
    frame `n_frames // 2`, and nothing else, by construction.

    Negatives come from the *background* cells of the same clip: same volume,
    same illumination, same regime, different answer. That is the matched
    control, and it is why this returns a whole small scene rather than one crop.
    """
    rng = np.random.default_rng(seed_base + index)
    regime = Regime.sample(rng)
    half = n_frames // 2

    n_bg = int(rng.integers(*n_background))
    palette = _elongate(Palette.sample(n_bg + 1, rng, regime), rng, elongate, stretch)
    palette.radius_um[:] *= radius_scale
    pos, keep = _place(n_bg + 1, rng, regime, palette, pack)
    palette = palette.take(keep)
    if divide and parent_round:
        # Mitotic rounding, applied to the cell that will divide. Packing is
        # already done, and this only ever makes the nucleus smaller along its
        # long axis, so it cannot create an overlap placement did not allow.
        a = palette.aniso[0]
        palette.aniso[0] = a * (1 - parent_round) + a.mean() * parent_round

    # Per-cell drift, in voxels: `STEP_UM` is the 3D displacement, so each axis
    # carries 1/sqrt(3) of it.
    vel = rng.normal(0, STEP_UM / np.sqrt(3), (len(pos), 3)) / VOXEL_NATIVE
    tracks = [dict(pos=pos[k].copy(), app=palette.take([k]), tid=k, vel=vel[k],
                   parent_slot=None) for k in range(len(pos))]

    div_axis = rng.normal(0, 1, 3)
    div_axis /= np.linalg.norm(div_axis) + 1e-9
    sep = max(SISTER_MIN_UM, rng.normal(SISTER_UM, SISTER_SD))

    nodes, edges, div_rows = [], [], []
    vols = np.zeros((n_frames,) + SHAPE, np.uint16)
    prev_row, next_row = {}, 0

    for t in range(n_frames):
        cens = np.stack([tr["pos"] for tr in tracks]).astype(np.float32)
        vol, _, _, _, _ = gen_volume_native(
            zshape=SHAPE[0], xyshape=SHAPE[1], cens_native=cens,
            rng=np.random.default_rng(seed_base + index * 131 + t + 1),
            place_mode="uniform", regime=regime,
            palette=concat([tr["app"] for tr in tracks]),
        )
        vols[t] = to_uint16(vol)

        cur = {}
        for k, tr in enumerate(tracks):
            nodes.append([t, *tr["pos"], tr["tid"]])
            cur[k] = next_row
            if divide and t == half and k == 0:
                div_rows.append(next_row)
            next_row += 1
        for k, tr in enumerate(tracks):
            if tr["parent_slot"] is not None and tr["parent_slot"] in prev_row:
                edges.append([prev_row[tr["parent_slot"]], cur[k]])
        prev_row = cur
        if t == n_frames - 1:
            break

        nxt = []
        for k, tr in enumerate(tracks):
            step = tr["vel"] + rng.normal(0, 0.3, 3)
            # The centre cell divides on the step out of the centre frame, so
            # the daughters exist at t+1 and the label names frame `half`. The
            # parent itself was rendered above as an ordinary cell.
            if divide and k == 0 and t == half:
                d0, d1 = tr["app"].divide(0, rng, regime)
                for d in (d0, d1):
                    d.radius_um[:] *= daughter_scale
                offset = 0.5 * sep * div_axis / VOXEL_NATIVE
                for app, sgn in ((d0, +1), (d1, -1)):
                    nxt.append(dict(pos=np.clip(tr["pos"] + step + sgn * offset, _LO, _HI),
                                    app=app, tid=tr["tid"], vel=tr["vel"],
                                    parent_slot=k))
                continue
            nxt.append(dict(pos=np.clip(tr["pos"] + step, _LO, _HI),
                            app=tr["app"].drift(rng, regime), tid=tr["tid"],
                            vel=tr["vel"], parent_slot=k))
        tracks = nxt

    return (vols, np.asarray(nodes, np.float32),
            np.asarray(edges, np.int32).reshape(-1, 2),
            np.asarray(div_rows, np.int32))


def write_division_clip(index, out_dir, **kw):
    """`generate_division_clip` to an npz, for inspection and for QC notebooks."""
    vols, nodes, edges, div_rows = generate_division_clip(index, **kw)
    path = Path(out_dir) / f"div_{index:06d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, volumes=vols, nodes=nodes, edges=edges,
                        divisions=div_rows,
                        voxel_um=VOXEL_NATIVE.astype(np.float32))
    return str(path)
