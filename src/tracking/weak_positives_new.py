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
import torch.nn.functional as F


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

    out: list[torch.Tensor] = []
    while idx.numel():
        i0 = idx[0]                      # brightest survivor: P is score-sorted
        out.append(i0)
        d2 = ((P[idx] - P[i0]) ** 2).sum(dim=1)
        idx = idx[d2 >= r2]
    if not out:
        return torch.zeros(0, dtype=torch.long, device=dev)
    return order[torch.stack(out)]


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


def _peak_near(
    vol: torch.Tensor,        # (Z, Y, X)
    pts: torch.Tensor,        # (n, 3) long
    r: int = 1,
) -> torch.Tensor:            # (n,) float
    """Brightest value in a (2r+1)^3 box around each point.

    Mapping a full-resolution GT coordinate onto the search grid floors it to
    a block, and a cell centred near a block boundary spills its peak into the
    neighbour. Reading the block alone therefore samples the cell's skirt and
    underestimates its brightness, which drags the whole intensity threshold
    down with it. Taking the max over the surrounding blocks measures the cell
    the same way a candidate peak is measured: as the brightest sample it
    produces on this grid.
    """
    Z, Y, X = vol.shape
    z, y, x = pts[:, 0], pts[:, 1], pts[:, 2]
    best = vol[z.clamp(0, Z - 1), y.clamp(0, Y - 1), x.clamp(0, X - 1)]
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                best = torch.maximum(best, vol[
                    (z + dz).clamp(0, Z - 1),
                    (y + dy).clamp(0, Y - 1),
                    (x + dx).clamp(0, X - 1),
                ])
    return best


def _nms_from_matrix(close: torch.Tensor) -> torch.Tensor:  # (n,) bool keep
    """Greedy NMS over a precomputed "within radius" adjacency matrix.

    ``close`` is (n, n) symmetric, with rows ordered by *decreasing* score, so
    the first surviving row is always the next winner. Each accepted point
    clears its whole neighbourhood in one vectorised row-AND, which turns the
    suppression into n cheap bitmask ops with no distance recomputation.

    The loop is run on CPU regardless of where ``close`` lives: it is n scalar
    reads of ``keep``, and on CUDA each one is a device sync that costs far
    more than the arithmetic it guards.
    """
    n = close.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)

    close = close.cpu()
    keep = torch.ones(n, dtype=torch.bool)
    for i in range(n):
        if keep[i]:
            keep &= ~close[i]
            keep[i] = True     # a point never suppresses itself
    return keep


def _drop_near(
    points_um: torch.Tensor,   # (n, 3) in microns
    others_um: torch.Tensor,   # (m, 3) in microns
    radius_um: float,
) -> torch.Tensor:             # (n,) bool, True = keep
    """Keep points at least ``radius_um`` from every point in ``others_um``."""
    if points_um.shape[0] == 0 or others_um.shape[0] == 0:
        return torch.ones(points_um.shape[0], dtype=torch.bool, device=points_um.device)
    return (torch.cdist(points_um, others_um) >= radius_um).all(dim=1)


@torch.no_grad()
def compute_weak_positives(
    imgs: torch.Tensor,        # (W, Z, Y, X) normalised, pre-augmentation
    coords: torch.Tensor,      # (W, M, 3) GT coords in imgs voxel space
    masks: torch.Tensor,       # (W, M) bool
    voxel_size: tuple[float, float, float],
    nms_um: float = 5.0,
    device: str | torch.device | None = None,
    downscale: tuple[int, int, int] = (1, 4, 4),
    threshold: float = 0.5,
    ignore_threshold: float = 0.0,
    ignore_nms_um: float | None = None,
    ignore_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:  # (W, Z, Y, X) uint8 each: weak, ignore
    """Image local maxima that are plausibly unannotated cells.

    A peak counts as a weak positive when it is a local maximum over an
    ``nms_um``-wide neighbourhood, is brighter than ``threshold`` times the
    mean peak intensity of its frame's GT cells, and does not sit inside that
    neighbourhood of a GT cell (those voxels are already true positives). Two
    weak peaks within ``nms_um`` of each other are a single cell, not two
    crowded ones — the NMS radius is a physical exclusion distance, so the
    loser stays a hard negative rather than becoming ambiguous.

    ``threshold`` is a fraction of the *mean* GT peak, so 1.0 rejects every
    cell dimmer than the average annotated one — including roughly half of a
    population identical to the GT. Values below 1 are the useful range.

    Ignore set
    ----------
    ``threshold`` only catches cells bright enough to clear it. A dim,
    unannotated cell produces no candidate at all, and every voxel it
    occupies is silently trained as background — confidently wrong, since we
    never actually looked. ``ignore_threshold`` (as a fraction of
    ``threshold``, 0 disables) reruns the same local-max search at that lower
    bar: local maxima that clear it but not the main threshold are real
    intensity bumps too faint to trust as a cell, so they go to the ignore
    set instead of background. Bumps already covered by a GT cell or a kept
    weak positive are true positives already known, not ambiguous, so they
    are dropped rather than added to ignore.

    ``ignore_nms_um`` (default: ``nms_um``) is the exclusion radius for that
    second pass — both ignore-vs-positive and ignore-vs-ignore. It is worth
    setting smaller than ``nms_um``, because the two radii answer different
    questions. Between positives, ``nms_um`` asserts "no two cells are this
    close", so it must be generous: a false weak positive is supervised as a
    cell and actively teaches a wrong peak. An ignore point asserts only
    "say nothing here", so an over-wide radius does not add a wrong gradient,
    it *deletes* correct background gradient in a ring around every cell —
    and at ``nms_um`` the ignore points are pushed a full exclusion distance
    away from the cells whose dim neighbours they are meant to cover.

    Its useful range is bounded below by the search grid. Two local maxima
    need a strictly lower sample between them, so no two points of the same
    pass can land closer than *two* grid steps — 3.25 um at the default
    ``downscale``, whatever ``ignore_nms_um`` says. Below that the only thing
    still shrinking is the distance to a GT cell, which is not on the grid.
    Going finer means a finer ``downscale``, not a smaller radius.

    ``ignore_floor`` clamps the ignore bar to an absolute normalised
    intensity, as a backstop for frames whose GT cells are unusually dim. Use
    it sparingly: the background pedestal differs ~3x between embryos, so a
    fixed floor is not fold-symmetric and will mean different things on each
    side of a cross-validation split. 0 disables it.

    Computed here rather than in the loss so it sees the raw normalised image:
    brightness augmentation would otherwise shift the intensity threshold, and
    cropping would change which cells define it. The mask is then carried
    through the same spatial augmentations as the image.

    The suppression radius is physical (microns), so it is independent of the
    UNet's output stride.

    ``downscale`` max-pools the volume before searching and scales the peaks
    back afterwards. The search is memory-bound, so (1, 4, 4) cuts it 16x. It
    also makes the working grid isotropic (1.625 == 0.40625*4 um), so the NMS
    ball is not skewed by voxel anisotropy. The cost is localisation: a peak
    is resolved only to its block, i.e. +-1.625 um.

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

    # Max-pool rather than stride the volume: a subsample reads whatever voxel
    # the grid happens to land on, so a cell between samples is measured on
    # its skirt. Pooling keeps each block's brightest voxel, so a peak
    # survives wherever inside its block it sits.
    # Never pool wider than the volume: max_pool3d floors the output size, so
    # a kernel larger than an axis yields a 0-length one and raises. Crops
    # smaller than the kernel are legitimate, as is the (W, 1, 1, 1)
    # placeholder a loader with image reading disabled hands over.
    dz, dy, dx = (min(int(v), s) for v, s in zip(downscale, full_spatial))
    # return_indices keeps, for every block, where inside it the max actually
    # sat. Without it a peak is reported at its block *corner* — up to
    # (dz, dy, dx) - 1 voxels off, always biased toward lower coordinates.
    small, argmax = F.max_pool3d(
        imgs.unsqueeze(0), kernel_size=(dz, dy, dx), stride=(dz, dy, dx),
        return_indices=True,
    )
    small, argmax = small.squeeze(0), argmax.squeeze(0)

    spatial = tuple(small.shape[1:])
    step = torch.as_tensor((dz, dy, dx), dtype=torch.float32, device=dev)
    vs = torch.as_tensor(voxel_size, dtype=torch.float32, device=dev)

    ig_nms_um = nms_um if ignore_nms_um is None else ignore_nms_um
    use_ignore = ignore_threshold > 0

    weak = torch.zeros(W, *full_spatial, dtype=torch.uint8, device=dev)
    ignore = torch.zeros(W, *full_spatial, dtype=torch.uint8, device=dev)
    nt = masks.sum(dim=1).long()

    fz, fy, fx = full_spatial

    def _to_full(f, blocks):
        """Block coords -> the full-resolution voxel the block's max sat on."""
        flat = argmax[f][blocks[:, 0], blocks[:, 1], blocks[:, 2]]
        return torch.stack(
            [flat // (fy * fx), (flat // fx) % fy, flat % fx], dim=1
        )

    def _mark(out, f, blocks):
        if not blocks.shape[0]:
            return
        k = _to_full(f, blocks)
        out[f, k[:, 0], k[:, 1], k[:, 2]] = 1

    for f in range(W):
        n = int(nt[f].item())
        if n == 0:
            continue
        # GT coords are full-resolution. Only the intensity lookup needs them
        # on the block grid; distances stay in microns at full precision.
        gt_blocks = torch.stack([
            (coords[f, :n, i] / step[i]).long().clamp(0, spatial[i] - 1)
            for i in range(3)
        ], dim=1)
        gt_um = coords[f, :n].float() * vs
        abs_threshold = _peak_near(small[f], gt_blocks).mean() * threshold
        low_bar = abs_threshold * ignore_threshold if use_ignore else abs_threshold
        if ignore_floor > 0:
            low_bar = torch.clamp(low_bar, min=ignore_floor)

        # One extraction at the *lowest* bar, shared by both passes: the weak
        # candidates are simply the bright end of the same list. The local-max
        # filter is what makes the pairwise matrix below affordable — the raw
        # thresholded set runs to ~2e5 points (a 160 GB distance matrix),
        # while its local maxima number ~1e3 (a few MB). It is also not
        # equivalent to skipping it: greedy NMS is order-dependent, so a
        # bright ridge would otherwise emit a point every radius along it.
        cand = torch.nonzero(small[f] > low_bar)
        if cand.numel() == 0:
            continue
        cand = cand[_is_local_max_at(small[f], cand)]
        if cand.numel() == 0:
            continue

        # Sort once, descending: every NMS below then reads in winner order.
        scores = small[f][cand[:, 0], cand[:, 1], cand[:, 2]]
        order = torch.argsort(scores, descending=True, stable=True)
        cand, scores = cand[order], scores[order]

        # One distance matrix for both passes, sliced per pass. Positions are
        # the true sub-block peaks, so the NMS balls are not quantised to the
        # search grid and stay comparable with the full-resolution GT.
        P = _to_full(f, cand).float() * vs
        D = torch.cdist(P, P)

        is_weak = scores > abs_threshold
        w_idx = torch.nonzero(is_weak).squeeze(1)

        # Weak: suppress among themselves at nms_um, *then* drop anything on a
        # GT cell. Order matters — a peak sitting on a GT cell is that cell,
        # so it should clear its own shoulder before being dropped as already
        # supervised. Suppressing by GT first would leave that shoulder behind
        # as a second, spurious weak positive.
        keep_w = w_idx[_nms_from_matrix(D[w_idx][:, w_idx] < nms_um).to(w_idx.device)]
        keep_w = keep_w[_drop_near(P[keep_w], gt_um, nms_um)]
        _mark(weak, f, cand[keep_w])

        if not use_ignore:
            continue

        # Ignore: the dim remainder of the same candidate list, so no second
        # threshold-and-local-max sweep. Same two steps at the narrower
        # radius, against GT *and* the weak points just kept.
        i_idx = torch.nonzero(~is_weak).squeeze(1)
        keep_i = i_idx[_nms_from_matrix(D[i_idx][:, i_idx] < ig_nms_um).to(i_idx.device)]
        known = torch.cat([gt_um, P[keep_w]])
        keep_i = keep_i[_drop_near(P[keep_i], known, ig_nms_um)]
        _mark(ignore, f, cand[keep_i])

    return weak.to(src_device), ignore.to(src_device)
