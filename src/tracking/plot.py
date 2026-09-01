"""Visualization utilities for the tracking module."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from utils.plot import scatter_by_depth


# Show the two frames in the window after downsampling and normalisation
def plot_sample(sample, t_start=0, tol=2, z=None):
    """Overlay the three supervision point sets on each frame of a window.

    GT (green) is supervised at full weight, weak positives (blue) at
    ``det_weak_weight``, and ignore points (orange) contribute no gradient at
    all — see ``losses.build_detection_target``. Everything unmarked trains as
    background, so what is *absent* here is as informative as what is drawn.

    Points within ``tol`` slices of ``z`` are drawn: a cell centred a slice or
    two away is still visible in this one. All three sets live in the same
    downsampled voxel space as ``imgs``.
    """
    imgs = sample["imgs"].float().numpy()  # (W, Z_ds, Y_ds, X_ds)
    coords = sample["coords"].numpy()  # (W, max_nodes, 3)
    masks = sample["masks"].numpy()  # (W, max_nodes) bool

    W, Z_ds = imgs.shape[0], imgs.shape[1]

    def _padded(key):
        """(W,) list of (N, 3) valid points from a -1-padded sample entry."""
        pts = sample.get(key)
        if pts is None:
            return [np.empty((0, 3), dtype=np.float32) for _ in range(W)]
        # Test all three columns, not just z: a row is padding only when every
        # column is -1, and valid rows are not guaranteed to stay packed at the
        # front once augmentation has dropped points from the middle.
        pts = pts.numpy()
        return [p[(p >= 0).all(axis=-1)] for p in pts]

    point_sets = {
        "GT": ([coords[i][masks[i]] for i in range(W)], "forestgreen"),
        "weak": (_padded("weak_positives"), "dodgerblue"),
        "ignored": (_padded("ignore_positives"), "orangered"),
    }

    if z is None:
        # Slot 0 may be padding, so pick from the real GT of frame 0.
        gt0 = point_sets["GT"][0][0]
        z = int(gt0[0, 0]) if len(gt0) else Z_ds // 2
    z = int(np.clip(z, 0, Z_ds - 1))

    fig, axes = plt.subplots(1, W, figsize=(7 * W, 6))
    if W == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, z], cmap="gray")

        counts = []
        for label, (per_frame, color) in point_sets.items():
            c = per_frame[i]  # (N, 3) in downsampled space
            counts.append(f"{len(c)} {label}")
            scatter_by_depth(
                ax, c, z, color, tol=tol, marker="o", s=40, linewidths=1,
                label=label,
            )

        ax.set_title(f"t={t_start + i}  -  z={z}  -  " + ", ".join(counts))
        ax.axis("off")
        # Only when something was drawn, else matplotlib warns about a legend
        # with no labelled artists.
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_detection(
    img: np.ndarray,  # (Z, Y, X) or (Y, X)
    det_logit: np.ndarray,  # same shape
    det_coords: np.ndarray | None,  # (N, 3) or (N, 2)
    z: int = 0,
    title: str = "Detection",
) -> plt.Figure:
    """Plot raw image vs detection logit at a given Z-slice, with detected peaks."""
    volumetric = img.ndim == 3
    if volumetric:
        img_slice = img[z]
        logit_slice = det_logit[z]
    else:
        img_slice = img
        logit_slice = det_logit

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(logit_slice, cmap="hot")
    axes[1].set_title("Detection logit")

    if det_coords is not None and len(det_coords) > 0:
        det_coords = np.asarray(det_coords)
        # A (N, 3) peak list against a single slice: every peak in the volume
        # used to be drawn on it, so a plane with two cells in it came out
        # covered in markers. Fading by |dz| is what separates the peaks that
        # are in this section from the ones that only project onto it.
        depth = volumetric and det_coords.ndim == 2 and det_coords.shape[1] == 3
        for ax in axes:
            if depth:
                scatter_by_depth(
                    ax, det_coords, z, "cyan", facecolor="cyan", s=10,
                    linewidths=0.5,
                )
            else:
                ax.scatter(
                    det_coords[:, 1], det_coords[:, 0], c="cyan", s=10,
                    linewidths=0.5,
                )

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_edges(
    coords_t: np.ndarray,  # (N_t, 3) or (N_t, 2)
    coords_t1: np.ndarray,  # (N_t1, 3) or (N_t1, 2)
    edges: list[tuple[int, int]],
    spatial_ax: tuple[int, int] = (1, 2),
    title: str = "Tracking edges",
) -> plt.Figure:
    """Plot GT or predicted edges between two consecutive frames."""
    ay, ax = spatial_ax

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(coords_t[:, ax], coords_t[:, ay], c="blue", label="frame t", s=20)
    axis.scatter(coords_t1[:, ax], coords_t1[:, ay], c="red", label="frame t+1", s=20)

    for i, j in edges:
        x_vals = [coords_t[i, ax], coords_t1[j, ax]]
        y_vals = [coords_t[i, ay], coords_t1[j, ay]]
        axis.plot(x_vals, y_vals, "k-", linewidth=0.5, alpha=0.6)

    axis.set_title(title)
    axis.legend()
    axis.invert_yaxis()
    plt.tight_layout()
    return fig
