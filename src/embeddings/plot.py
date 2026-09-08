import numpy as np

from embeddings.augment import apply_photometric
from embeddings.patches import extract_patches, normalise_coords, normalise_frame


def render_view(vol, z, view, rng):
    """The whole slice under a view's geometry.

    Reuses extract_patches by asking for one patch the size of the frame, so
    the visualisation cannot drift from what the patches actually get.
    """
    Y, X = vol.shape[1:]
    centre = np.array([[z, (Y - 1) / 2, (X - 1) / 2]], np.float32)
    big = extract_patches(vol, centre, (1, Y, X), angle_deg=view["angle"],
                          scale=view["scale"], elastic=view["elastic"])
    # (N, n_frames, pz, py, px) -> the single slice of the single slab.
    return apply_photometric(big, view["photo"], rng)[0, 0, 0]


def map_coords_to_view(p, view, shape_yx, iters=2):
    """Where an input point ends up in the rendered view.

    The rendered frame is warped but the detections are not, so scattering the
    raw coordinates over it puts every circle in the wrong place -- the marker
    for a cell lands wherever that cell *used* to be. They have to go through
    the same geometry.

    extract_patches maps an output position g to the input it samples:
    ``g -> rotate_scale(g - centre) + centre + d(...)``. Inverting it means
    undoing the elastic displacement first (it is applied last), then the
    rotation and scale about the frame centre. The elastic inverse has no
    closed form -- d is evaluated at the *displaced* point -- so it is solved
    by fixed-point iteration; two passes are enough here, and one is visibly
    short when the field is large.
    """
    Y, X = shape_yx
    centre = np.array([(Y - 1) / 2, (X - 1) / 2], np.float32)
    q = p[:, 1:3].astype(np.float32).copy()

    if view["elastic"] is not None:
        u = q.copy()
        for _ in range(iters):
            iy = np.clip(np.rint(u[:, 0]).astype(int), 0, Y - 1)
            ix = np.clip(np.rint(u[:, 1]).astype(int), 0, X - 1)
            u = q - np.stack([view["elastic"][0][iy, ix],
                              view["elastic"][1][iy, ix]], 1)
        q = u

    d = q - centre
    th, s = np.deg2rad(view["angle"]), view["scale"]
    cos, sin = np.cos(th), np.sin(th)
    out = p.copy()
    out[:, 1] = centre[0] + s * (cos * d[:, 0] + sin * d[:, 1])
    out[:, 2] = centre[1] + s * (-sin * d[:, 0] + cos * d[:, 1])
    return out