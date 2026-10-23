"""Generating one synthetic sample: a static volume, or a time sequence.

Motion is calibrated on the real lineage edges of the .geff files: median step
1.86 um/frame, lag-1 directional persistence +0.30, sister separation 7.24 um.
Divisions are deliberately over-sampled -- they are the supervision the real
data is short of (~0.26% of nodes) and the main reason this generator exists.

A sequence is one *acquisition*: a single ``Regime`` covers all its frames, and
each cell keeps its ``Palette`` row as it moves. Both used to be redrawn per
frame, which is worse than it sounds -- the illumination field, the shape of
every nucleus and its brightness all changed completely between t and t+1, so
the only thing linking a cell to its own next frame was position. A tracker
trained on that cannot learn to match on appearance, because in the data it was
shown appearance carries no information across time.

Motion is decomposed into two parts, which is what lets the overlap guardrail
below stay cheap:

* ``GLOBAL_IMAGE_SHIFT`` -- one translation applied to every cell alike (stage
  drift, the embryo moving as a whole). Being rigid it cannot change any pair's
  separation, so it never needs checking. It may be zero.
* ``CELL_MOVEMENT`` -- each cell's own jitter about that shift. This is the
  only part that can bring two nuclei together, so this is the only part the
  guardrail scales back.

The guardrail exists because placement used to be the only thing enforcing
spacing: ``sample_shell_positions`` applies a collision check at t=0 and the
propagation that followed applied none, so cells drifted straight through each
other over the remaining frames. Measured over 40 sequences, 18% of cells ended
up within 5.2 um of a neighbour and 1% within 2.0 um -- two 8.5 um nuclei at 2
um of separation, each carrying its own node. That is a pair of labels on what
renders as one blob, which is precisely the "split one cell into two detections"
behaviour the detector must not be taught.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_gen.acquisition import Palette, Regime, concat, reserved_um
from data_gen.constants import VOXEL_NATIVE
from data_gen.geometry import sample_shell_positions
from data_gen.render import gen_volume_native

# Measured on the real lineages; see the module docstring.
STEP_MED_UM = 1.86
PERSISTENCE = 0.30
# Sister separation one frame after the division, measured on the annotation:
# 151 real divisions give a median of 10.57 um, p10 6.36 and p90 14.36. The
# value here was 7.24 and produced a median of 6.82, with 54% of pairs closer
# than the detector's 7 um NMS radius against 15% in the real data -- so over
# half of the generated divisions were teaching a configuration that cannot
# survive its own detector.
# 12.4, not the 10.57 measured: the collision check rejects the widest draws
# (a daughter placed further out is likelier to overlap a neighbour), so the
# realised median comes out ~15% below the drawn one. At 10.6 the generated
# median was 8.95; this pre-compensates for that bias rather than leaving the
# fit check to quietly shift the distribution.
SISTER_UM = 12.4
SISTER_SD = 3.1
SISTER_MIN_UM = 4.0

# A dividing cell does not go from one sphere to two in a single step: at the
# frame the annotation calls the division, the nucleus has already resolved
# into two kernels inside one envelope. Rendering it as an ordinary sphere left
# the labelled frame with no visible cue at all, so the classifier's only
# evidence was the frame after. The two lobes are drawn at this fraction of the
# separation the daughters will take, along the same axis.
MITOTIC_FRAC = 0.5
MITOTIC_LOBE_SCALE = 0.78
# Median length of a 3D isotropic gaussian step, in units of its per-axis
# sigma (the chi distribution with three degrees of freedom). Dividing the
# measured 1.86 um step by it gives the per-axis sigma to draw.
#
# The trailing factor is calibration slack, set so that the generated edges
# reproduce the measured 1.86 um median once persistence, the collective drift
# and the overlap guardrail have all had their say. It was 0.93 while cells
# could pass through each other: the tissue is packed close to tangent, so most
# of a drawn step is spent on room the cell does not have, and the realised
# median came out at 0.96 um against the 1.86 target until the drawn step was
# raised to match.
#
# It depends on how tightly the field is packed, so it has to be refitted
# whenever the packing rule changes. Measured over 40 sequences at the current
# packing: 2.40 -> 1.77, 2.60 -> 1.85, 2.90 -> 2.03. Fit on at least 20
# sequences -- 6 gives 2.15 against the same 1.86 and would send you chasing a
# bias that is not there.
_CHI3_MEDIAN = 1.5382
_STEP_UM = (STEP_MED_UM / _CHI3_MEDIAN) * 2.60
# Per axis, in voxels. The conversion has to happen per axis: a voxel is 1.625
# um in z and 0.40625 in y/x, so a step drawn isotropically in *voxels* is four
# times longer in z than in y/x once measured in microns. It was, and it showed
# -- generated cells moved 0.41 um per frame laterally against the real 0.81,
# while the 3D step matched (1.84 against 1.86) because the inflated z
# displacement made up the difference and hid the error.
_STEP_VOX = _STEP_UM / VOXEL_NATIVE
# Size of GLOBAL_IMAGE_SHIFT, as a share of the drawn step. Set it to 0.0 for a
# field that only jitters, with no collective drift at all. It is part of what
# the 0.93 slack above is calibrated against, so changing it means re-running
# ``calibrate.py``.
GLOBAL_SHIFT_FRAC = 0.35
# How far CELL_MOVEMENT is scaled back when a cell would otherwise overlap a
# neighbour. 0.0 is always available -- it leaves the cell at its globally
# shifted position, which is a rigid image of a frame that was already clear --
# so the search cannot fail.
MOVE_SCALES = (1.0, 0.6, 0.3, 0.0)

SHAPE = (64, 256, 256)
# Keep cells clear of the border, where a half-rendered cell is a bad label.
_LO, _HI = [4, 12, 12], [59, 243, 243]


def to_uint16(vol):
    """float [0, 1] -> uint16, which is what the real data is stored as."""
    return np.clip(vol * 65535.0, 0, 65535).astype(np.uint16)


def generate_static(index, counts, shell, out_dir, seed_base=100_000):
    """One static volume with its cell centroids."""
    rng = np.random.default_rng(seed_base + index)
    regime = Regime.sample(rng)
    n_cells = int(max(20, rng.choice(counts) * 1.25))
    vol, _, cens, _, mask = gen_volume_native(
        zshape=SHAPE[0], xyshape=SHAPE[1], n_cells=n_cells, rng=rng,
        shell=shell, regime=regime,
    )
    rel = f"static/vol_{index:05d}.npz"
    path = Path(out_dir) / rel
    np.savez(
        path,
        volume=to_uint16(vol),
        mask=mask,
        centroids=cens.astype(np.float32),
        voxel_um=VOXEL_NATIVE.astype(np.float32),
    )
    return {
        "index": index,
        "file": rel,
        "n_cells": int(len(cens)),
        "shape": list(vol.shape),
        "bytes": int(path.stat().st_size),
    }


def _global_shift(flow):
    """GLOBAL_IMAGE_SHIFT: one translation for the whole field, in voxels.

    ``flow`` is a unit direction in microns, so it converts per axis too. The
    same vector goes to every cell, so on its own it cannot change any pair's
    separation. Set ``GLOBAL_SHIFT_FRAC`` to 0.0 for no collective drift.
    """
    return GLOBAL_SHIFT_FRAC * _STEP_UM * flow / VOXEL_NATIVE


def _cell_movement(track, rng):
    """CELL_MOVEMENT: one cell's own jitter, in voxels.

    An Ornstein-Uhlenbeck step -- persistent velocity plus fresh noise. The
    global drift is deliberately not folded in here: it used to be, and storing
    the sum back into ``vel`` let persistence compound the drift frame after
    frame, so the collective motion grew instead of staying constant.
    """
    return PERSISTENCE * track["vel"] + np.sqrt(1 - PERSISTENCE**2) * rng.normal(
        0, _STEP_VOX * track["sc"], 3
    )


def _clears(point, radius, obs_pos, obs_r, csf):
    """Whether a nucleus at ``point`` keeps its distance from every obstacle.

    The rule is placement's own (``geometry.sample_shell_positions``): centres
    stay at least ``csf * (r_i + r_j)`` apart, measured in microns so the 4x
    anisotropy between z and y/x is respected.
    """
    if len(obs_pos) == 0:
        return True
    d = np.linalg.norm((np.asarray(obs_pos) - point) * VOXEL_NATIVE, axis=1)
    return bool((d >= csf * (radius + np.asarray(obs_r))).all())


def _resolve_moves(pos, shift, offset, radii, csf, rng):
    """Take as much of each cell's move as leaves the frame clear.

    Cells are visited in a random order -- a fixed one would hand the low
    indices their full step every frame and make the high ones always give way
    -- and each takes the largest scale in ``MOVE_SCALES`` that clears every
    other cell's *currently chosen* position.

    A cell that cannot place even its unjittered shift stays where it was.
    That is what makes the search total: ``pos`` is the previous frame, which
    was already clear, and every cell that moved was checked against the cells
    still sitting on it, so a cell's own previous position is always free for
    it. The cost is that a field drifting into the edge of the volume piles up
    against it rather than leaving -- these sequences keep every cell for their
    whole length by construction, and giving them real exits is a separate job.
    """
    out = pos.copy()
    for i in rng.permutation(len(pos)):
        others = np.delete(np.arange(len(pos)), i)
        for s in MOVE_SCALES:
            cand = np.clip(pos[i] + shift + s * offset[i], _LO, _HI)
            if _clears(cand, radii[i], out[others], radii[others], csf):
                out[i] = cand
                break
    return out


def _lobe(app):
    """One kernel of a dividing nucleus: the cell's own appearance, shrunk.

    Not `Palette.divide`, which draws two *daughters* with fresh texture and
    independent brightness. These are the two halves of a nucleus that has not
    finished separating, so they keep the parent's appearance and only lose
    size -- what distinguishes the mitotic frame from the frame after it is
    that the lobes still belong to one cell.
    """
    out = app.take([0])
    out.radius_um[:] = out.radius_um * MITOTIC_LOBE_SCALE
    return out


def _plan_transition(tracks, cens, rng, regime, flow, division_rate):
    """Move every cell, then decide which divide. Returns `(nxt, mitotic)`.

    Separated from the render loop so the frame can be drawn knowing which of
    its cells are mid-division; `mitotic` maps a slot to the lobe offset (in
    voxels) and the parent radius, and is empty when nothing divides.
    """
    # Slow appearance change: bleaching, growth, a little rotation.
    apps = [tr["app"].drift(rng, regime) for tr in tracks]
    radii = np.array([tr["r"] for tr in tracks], np.float32)
    shift = _global_shift(flow)
    offsets = np.stack([_cell_movement(tr, rng) for tr in tracks])
    moved = _resolve_moves(cens, shift, offsets, radii, regime.collision_factor, rng)

    # Divisions are tested against the *settled* frame: positions are final
    # before any daughter is placed, so a division is checked against where the
    # neighbours actually ended up. `occupied` then tracks the frame as it is
    # built -- a dividing cell is replaced by its first daughter and the second
    # is appended -- so two neighbours dividing at once still cannot be placed
    # on top of each other.
    occupied, occ_r = list(moved), list(radii)
    nxt, mitotic = [], {}
    for k, tr in enumerate(tracks):
        # The jitter the guardrail allowed. Feeding back what was taken -- not
        # what was drawn -- keeps a blocked cell from building up a velocity it
        # was never able to use.
        v = moved[k] - (cens[k] + shift)
        app, daughters = apps[k], None
        if rng.random() < division_rate:
            d = rng.normal(0, 1, 3)
            d /= np.linalg.norm(d) + 1e-9
            sep = max(SISTER_MIN_UM, rng.normal(SISTER_UM, SISTER_SD))
            half = 0.5 * sep * d / VOXEL_NATIVE
            pair = app.divide(0, rng, regime)
            spots = np.stack([moved[k] + half, moved[k] - half])
            r_pair = np.array([float(reserved_um(q.radius_um[0], q.aniso[0]))
                               for q in pair], np.float32)
            # Everything except the cell being replaced. The two sisters are not
            # checked against each other: SISTER_UM is measured on the real
            # lineages and is the one separation not up for negotiation.
            rest = [i for i in range(len(occupied)) if i != k]
            obs_p = [occupied[i] for i in rest]
            obs_r = [occ_r[i] for i in rest]
            fits = (
                (spots >= _LO).all() and (spots <= _HI).all()
                and all(_clears(s, r, obs_p, obs_r, regime.collision_factor)
                        for s, r in zip(spots, r_pair))
            )
            # A cell with no room to divide simply does not divide this frame.
            # Forcing it would put two labels inside one nucleus, which is the
            # failure this whole guardrail is here to avoid.
            if fits:
                daughters = (pair, spots, r_pair)
                # What this cell shows *now*: two kernels, half a separation
                # apart on the axis the daughters will take.
                mitotic[k] = (d * (MITOTIC_FRAC * sep) / VOXEL_NATIVE,
                              float(app.radius_um[0]))

        if daughters is None:
            nxt.append(dict(
                pos=moved[k], vel=v, sc=tr["sc"], tid=tr["tid"], r=tr["r"],
                parent_slot=k, is_div=False, app=app,
            ))
            continue

        pair, spots, r_pair = daughters
        occupied[k], occ_r[k] = spots[0], r_pair[0]
        occupied.append(spots[1])
        occ_r.append(r_pair[1])
        for spot, daughter, r_d in zip(spots, pair, r_pair):
            nxt.append(dict(
                pos=spot, vel=v * 0.5, sc=tr["sc"], tid=tr["tid"],
                r=float(r_d), parent_slot=k, is_div=True, app=daughter,
            ))
    return nxt, mitotic


def generate_sequence(index, counts, shell, out_dir, n_frames=6,
                      division_rate=0.05, seed_base=500_000):
    """One time sequence with a lineage: volumes, nodes, edges and divisions.

    ``nodes`` is (N, 5) [t, z, y, x, track_id]; ``edges`` indexes into it as
    (parent, child) pairs; ``divisions`` lists the node indices that divided.
    """
    rng = np.random.default_rng(seed_base + index)
    # One acquisition for the whole sequence: same optics, same lighting, same
    # noise level at every frame, as a microscope would give.
    regime = Regime.sample(rng)
    n0 = int(max(20, rng.choice(counts) * 1.25 * regime.density_factor))
    pos, radii, aniso = sample_shell_positions(
        n0, SHAPE, rng, shell=shell,
        r_lo=regime.radius_um[0], r_hi=regime.radius_um[1],
        csf=regime.collision_factor,
    )
    if len(pos) < 10:
        return None
    # Appearance is drawn once per cell here, then carried along the track. The
    # radii are the ones placement already used for its collision check, so the
    # rendered cells are the cells that were packed.
    palette = Palette.sample(len(pos), rng, regime)
    palette.radius_um[:] = radii
    palette.aniso[:] = aniso
    reserved = reserved_um(radii, aniso)

    # Per-cell speed scale, log-normal, which reproduces the heavy tail of the
    # real step distribution -- most cells barely move, a few move a lot.
    sc = np.exp(rng.normal(0, 0.55, len(pos)))
    vel = rng.normal(0, _STEP_VOX, (len(pos), 3)) * sc[:, None]
    flow = rng.normal(0, 1, 3)
    flow /= np.linalg.norm(flow) + 1e-9

    # ``r`` is the radius the spacing rule uses, and it is deliberately *not*
    # the palette's: appearance drifts a couple of percent a frame, and letting
    # that feed back into the geometry means a cell that grows slightly is
    # retroactively overlapping a neighbour it was legally placed against. The
    # rendered size still drifts; only the room a cell reserves is held fixed.
    tracks = [
        dict(pos=p.copy(), vel=vel[k], sc=sc[k], tid=k, r=float(reserved[k]),
             app=palette.take([k]))
        for k, p in enumerate(pos)
    ]
    nodes, edges, divisions = [], [], []
    vols = np.zeros((n_frames,) + SHAPE, np.uint16)
    next_id, prev_ids = 0, {}

    for t in range(n_frames):
        cens = np.array([tr["pos"] for tr in tracks], np.float32)

        # The transition out of this frame is planned *before* the frame is
        # rendered, because a cell that is about to divide has to be drawn
        # differently in it: at the frame the annotation calls the division the
        # nucleus has already split into two kernels. Planning first is the only
        # way the label and the picture can agree -- otherwise the labelled
        # frame shows an ordinary sphere and the evidence exists only afterwards.
        nxt, mitotic = (None, {}) if t == n_frames - 1 else _plan_transition(
            tracks, cens, rng, regime, flow, division_rate
        )

        # A dividing cell renders as two lobes on the axis its daughters will
        # take. They are extra *render* entries only: the cell keeps one node,
        # at the centroid, which is the midpoint of the two lobes.
        # `.copy()` matters: writing the first lobe back into a view of `cens`
        # would move the centre the second lobe is then measured from, and the
        # pair would come out as {centre, centre + offset/2} rather than
        # straddling it.
        lobe_cens = cens.copy()
        render_cens = [lobe_cens]
        render_apps = [tr["app"] for tr in tracks]
        for k, (offset, r_um) in mitotic.items():
            render_apps[k] = _lobe(render_apps[k])
            lobe_cens[k] = cens[k] + offset / 2.0
            render_cens.append((cens[k] - offset / 2.0)[None])
            render_apps.append(_lobe(tracks[k]["app"]))

        frame_palette = concat(render_apps)
        vol, _, _, _, _ = gen_volume_native(
            zshape=SHAPE[0], xyshape=SHAPE[1],
            cens_native=np.concatenate(render_cens).astype(np.float32),
            # A fresh rng per frame, for the shot noise and the medium: those
            # are genuinely independent between exposures. Everything that is
            # not -- optics, lighting, cell appearance -- comes in through
            # ``regime`` and ``frame_palette`` instead.
            rng=np.random.default_rng(900_000 + index * 97 + t),
            shell=shell, regime=regime, palette=frame_palette,
        )
        vols[t] = to_uint16(vol)

        cur_ids = {}
        for k, tr in enumerate(tracks):
            nodes.append([t, tr["pos"][0], tr["pos"][1], tr["pos"][2], tr["tid"]])
            cur_ids[k] = next_id
            next_id += 1
        for k, tr in enumerate(tracks):
            parent = tr.get("parent_slot")
            if parent is not None and parent in prev_ids:
                edges.append([prev_ids[parent], cur_ids[k]])
                if tr.get("is_div"):
                    divisions.append(prev_ids[parent])
        prev_ids = cur_ids
        if t == n_frames - 1:
            break
        tracks = nxt

    rel = f"sequences/seq_{index:04d}.npz"
    path = Path(out_dir) / rel
    np.savez(
        path,
        volumes=vols,
        nodes=np.array(nodes, np.float32),
        edges=np.array(edges, np.int32),
        divisions=np.array(sorted(set(divisions)), np.int32),
        # The volumes are rendered at the native voxel size. The previous
        # generator wrote (1.625, 1.625, 1.625) here, which is wrong and was
        # measurably so: under it a cell reads 4.1 um in z against 14.6 um in
        # y/x, a 3.6:1 disc rather than a sphere.
        voxel_um=VOXEL_NATIVE.astype(np.float32),
    )
    return {
        "index": index,
        "file": rel,
        "T": int(n_frames),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_divisions": len(set(divisions)),
        "bytes": int(path.stat().st_size),
    }
