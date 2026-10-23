"""Weak positives: image peaks that are plausibly unannotated cells.

The GT annotation covers only a small fraction of the cells actually present,
so unlabelled cells would otherwise be trained as background. These peaks are
fed to the detection loss at a reduced weight instead.

Kept out of the loss so they can be derived from the raw normalised image,
before augmentation, and precomputed once per video (see
main_precompute_weak.py) rather than recomputed for every sampled window.
"""

from __future__ import annotations

import torch



def greedy_nms(
    points: torch.Tensor,      # (n, 3) voxel coords
    scores: torch.Tensor,      # (n,) intensities
    voxel_size: torch.Tensor,  # (3,) um per voxel
    radius_um: float,
    seeds: torch.Tensor | None = None,  # (m, 3) coords that suppress but are
                                        # never emitted (the GT cells)
) -> torch.Tensor:             # indices into points, kept
    """Keep the brightest point in every ``radius_um`` ball.

    Pure torch so it runs wherever its inputs live. Converges in one round per
    *output* peak rather than per input point: candidates are sorted by
    intensity once, so the head of the surviving list is always the next
    winner and each round deletes its whole neighbourhood. That is ~10^2
    rounds for ~10^5 candidates.
    """
    n = points.shape[0]
    dev = points.device
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=dev)

    order = torch.argsort(scores, descending=True, stable=True)
    P = points[order].float() * voxel_size
    r2 = float(radius_um) ** 2
    idx = torch.arange(n, device=dev)

    # GT cells suppress their own neighbourhood up front: a peak there is that
    # same cell, already a true positive.
    if seeds is not None and seeds.shape[0]:
        S = seeds.float() * voxel_size
        keep = torch.ones(n, dtype=torch.bool, device=dev)
        for i in range(0, S.shape[0], 64):              # bound the n x m temp
            d2 = ((P[:, None, :] - S[None, i:i + 64, :]) ** 2).sum(-1)
            keep &= (d2 >= r2).all(dim=1)
        idx = idx[keep]

    # Every survivor with no higher-scoring survivor inside the radius wins
    # *this* round, not just the brightest one. That is the same answer the
    # one-at-a-time loop gives -- a point the loop would emit later is exactly
    # one that nothing above it suppresses, and a point it would suppress has,
    # by definition, a higher survivor within r -- and it converges in rounds
    # proportional to the longest suppression chain rather than to the number
    # of peaks. On this data that is 2-4 rounds against ~930.
    #
    # Round count is what matters because each one costs a device-to-host sync
    # (`.numel()` on a CUDA tensor), and at ~20 us a sync the old loop spent
    # more time waiting than computing: 23 ms a frame for work the GPU does in
    # a fraction of that.
    #
    # The n x n neighbour test is chunked over rows so the temporary stays
    # bounded. `CHUNK` is a memory budget, not a speed knob: at `max_candidates`
    # = 50 000 survivors a chunk holds CHUNK x 50 000 floats plus the same again
    # as a bool, so 4096 rows peaks at 2.6 GiB -- fine on a development card and
    # not on a 16 GB T4 that also holds the detector. 512 rows is ~130 MiB and
    # costs a few more iterations of a kernel that was never the bottleneck.
    CHUNK = 512

    def _sq_dist(a, b):
        """(c, m) squared distances, per axis, no `cdist`.

        Neither `cdist` mode is usable here. The default expands ||a-b||^2 as
        |a|^2 + |b|^2 - 2a.b, which cancels: coordinates reach ~10^2 um so the
        squares reach ~10^4 and float32 leaves ~1e-3 of slop on a test whose
        threshold is 36. That is not a rounding curiosity -- on a lattice spaced
        at exactly the radius, which a voxel grid produces constantly, it
        suppressed 469 of 6 069 peaks that the one-at-a-time loop keeps.
        `donot_use_mm_for_euclid_dist` is exact but 64x slower, worse than the
        loop it replaces.

        Subtract, square, sum, one axis at a time: the same arithmetic the
        original `((P[idx] - P[i0]) ** 2).sum(dim=1)` did, so it agrees to the
        bit, and only (c, m) temporaries rather than the (c, m, 3) a broadcast
        difference would materialise.
        """
        d = a[:, 0, None] - b[None, :, 0]
        out = d * d
        for k in (1, 2):
            d = a[:, k, None] - b[None, :, k]
            out += d * d
        return out

    out: list[torch.Tensor] = []
    while idx.numel():
        Pi = P[idx]
        m = int(idx.numel())
        dominated = torch.zeros(m, dtype=torch.bool, device=dev)
        for s in range(0, m, CHUNK):
            c = min(CHUNK, m - s)
            near = _sq_dist(Pi[s:s + c], Pi) < r2
            # Strictly higher score = strictly earlier in the score-sorted
            # order, so "is there a better one within r" is a lower column
            # index. Columns before this chunk are all earlier; inside it the
            # strict lower triangle is (which also excludes the diagonal, so a
            # point cannot dominate itself); after it, none are.
            near[:, s:s + c] &= torch.ones(c, c, dtype=torch.bool,
                                           device=dev).tril(-1)
            near[:, s + c:] = False
            dominated[s:s + c] = near.any(dim=1)
        winners = idx[~dominated]
        out.append(winners)
        if not winners.numel():
            break                        # cannot happen: the top point always wins
        # Everything inside r of a winner is now suppressed.
        keep = torch.ones(m, dtype=torch.bool, device=dev)
        Pw = P[winners]
        for s in range(0, m, CHUNK):
            c = min(CHUNK, m - s)
            keep[s:s + c] = (_sq_dist(Pi[s:s + c], Pw) >= r2).all(dim=1)
        idx = idx[keep]
    if not out:
        return torch.zeros(0, dtype=torch.long, device=dev)
    kept = torch.cat(out)
    # Score order, so the result is the sequence the one-at-a-time loop emits.
    return order[kept.sort().values]


def _is_local_max_at(
    vol: torch.Tensor,        # (Z, Y, X)
    pts: torch.Tensor,        # (n, 3) long
) -> torch.Tensor:            # (n,) bool
    """Exact 3x3x3 local-maximum test, evaluated only at ``pts``.

    A stride-1 max-pool answers this for every voxel in the volume, but the
    question is only ever asked about the few percent that clear the intensity
    threshold. Gathering the 26 neighbours of those points is bandwidth-bound
    on n, not on the volume, which is an order of magnitude less work.
    """
    Z, Y, X = vol.shape
    z, y, x = pts[:, 0], pts[:, 1], pts[:, 2]
    v = vol[z, y, x]
    keep = torch.ones(pts.shape[0], dtype=torch.bool, device=vol.device)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                keep &= v >= vol[
                    (z + dz).clamp(0, Z - 1),
                    (y + dy).clamp(0, Y - 1),
                    (x + dx).clamp(0, X - 1),
                ]
    return keep


@torch.no_grad()
def compute_weak_positives(
    imgs: torch.Tensor,        # (W, Z, Y, X) normalised, pre-augmentation
    coords: torch.Tensor,      # (W, M, 3) GT coords in imgs voxel space
    masks: torch.Tensor,       # (W, M) bool
    voxel_size: tuple[float, float, float],
    nms_um: float = 7.0 * 3,
    device: str | torch.device | None = None,
    downscale: tuple[int, int, int] = (2, 8, 8),
    threshold: float = 0.5,
) -> torch.Tensor:             # (W, Z, Y, X) uint8
    """Image local maxima that are plausibly unannotated cells.

    A peak counts as a weak positive when it is a local maximum over an
    ``nms_um``-wide neighbourhood, is brighter than half the mean intensity at
    the GT cells of its frame, and does not sit inside that neighbourhood of a
    GT cell (those voxels are already true positives).

    Computed here rather than in the loss so it sees the raw normalised image:
    brightness augmentation would otherwise shift the intensity threshold, and
    cropping would change which cells define it. The mask is then carried
    through the same spatial augmentations as the image.

    The suppression radius is physical (microns), so it is independent of the
    UNet's output stride.

    ``downscale`` strides the volume before searching and scales the peaks
    back afterwards. The search is memory-bound, so (2, 8, 8) cuts the work
    128x. It also makes the working grid isotropic (1.625*2 == 0.40625*8 ==
    3.25 um), so the NMS ball is not skewed by voxel anisotropy. The cost is
    localisation: a peak is resolved only to its block, i.e. +-3.25 um.

    ``device`` runs the work elsewhere (e.g. 'cuda') and returns the result
    on the input's device. CUDA cannot be used from fork-based DataLoader
    workers (that needs num_workers=0 or the spawn start method), which is
    why the dataset stays on CPU and the precompute script is the natural
    place to ask for a GPU.
    """
    src_device = imgs.device
    dev = torch.device(device) if device is not None else src_device

    imgs = imgs.to(dev, dtype=torch.float32)
    coords = coords.to(dev)
    masks = masks.to(dev)

    W = imgs.shape[0]
    full_spatial = tuple(imgs.shape[1:])

    # Search on a strided view: 128x less to scan at (2, 8, 8), and the
    # resulting grid is isotropic so the NMS ball stays a ball.
    dz, dy, dx = (int(v) for v in downscale)
    small = imgs[:, ::dz, ::dy, ::dx]
    spatial = tuple(small.shape[1:])
    step = torch.as_tensor((dz, dy, dx), dtype=torch.float32, device=dev)
    vs = torch.as_tensor(voxel_size, dtype=torch.float32, device=dev) * step

    weak = torch.zeros(W, *full_spatial, dtype=torch.uint8, device=dev)
    nt = masks.sum(dim=1).long()
    for f in range(W):
        n = int(nt[f].item())
        if n == 0:
            continue
        # GT coords live in full-resolution voxels; bring them onto the
        # strided grid for the intensity threshold and the NMS seeds.
        gt = (coords[f, :n] / step).long()
        zi = gt[:, 0].clamp(0, spatial[0] - 1)
        yi = gt[:, 1].clamp(0, spatial[1] - 1)
        xi = gt[:, 2].clamp(0, spatial[2] - 1)
        threshold = small[f, zi, yi, xi].mean() * threshold

        # Threshold first: only a few percent of voxels clear it, and the
        # local-max test below then only has to look at those.
        cand = torch.nonzero(small[f] > threshold)
        if cand.numel() == 0:
            continue
        # Restrict to genuine local maxima. Feeding the whole thresholded set
        # to the NMS is NOT equivalent: the greedy pass is order-dependent, so
        # it would also keep "shoulder" voxels sitting one radius from a real
        # peak on a bright ridge (279 peaks vs 100 on a test frame).
        cand = cand[_is_local_max_at(small[f], cand)]
        if cand.numel() == 0:
            continue

        scores = small[f][cand[:, 0], cand[:, 1], cand[:, 2]]
        seeds = torch.stack([zi, yi, xi], dim=1)
        keep = greedy_nms(cand, scores, vs, nms_um, seeds=seeds)
        if keep.numel():
            k = (cand[keep].float() * step).long()   # back to full resolution
            weak[
                f,
                k[:, 0].clamp(0, full_spatial[0] - 1),
                k[:, 1].clamp(0, full_spatial[1] - 1),
                k[:, 2].clamp(0, full_spatial[2] - 1),
            ] = 1

    return weak.to(src_device)
