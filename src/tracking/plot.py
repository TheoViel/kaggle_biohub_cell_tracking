"""Visualization utilities for the tracking module."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# Show the two frames in the window after downsampling and normalisation
def plot_sample(sample, t_start=0):
    imgs = sample["imgs"].float().numpy()   # (W, Z_ds, Y_ds, X_ds)
    coords = sample["coords"].numpy()   # (W, max_nodes, 3)
    masks  = sample["masks"].numpy()    # (W, max_nodes) bool

    W, Z_ds, Y_ds, X_ds = imgs.shape

    z = int(coords[0, 0, 0])
    fig, axes = plt.subplots(1, W, figsize=(7 * W, 6))
    if W == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, z], cmap="gray")
        # GT nodes near this z slice
        n_real = int(masks[i].sum())
        c = coords[i, :n_real]   # (N, 3) in downsampled space
        near = np.abs(c[:, 0] - z) <= 2
        if near.any():
            ax.scatter(c[near, 2], c[near, 1], c="cyan", s=80,
                    edgecolors="white", linewidths=0.8, zorder=5)
        ax.set_title(f"t={t_start + i}  -  z={z}  -  {n_real} GT cell(s)")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_detection(
    img: np.ndarray,               # (Z, Y, X) or (Y, X)
    det_logit: np.ndarray,         # same shape
    det_coords: np.ndarray | None, # (N, 3) or (N, 2)
    z: int = 0,
    title: str = "Detection",
) -> plt.Figure:
    """Plot raw image vs detection logit at a given Z-slice, with detected peaks."""
    if img.ndim == 3:
        img_slice = img[z]
        logit_slice = det_logit[z]
        coords_2d = det_coords[:, 1:] if det_coords is not None and det_coords.ndim == 2 and det_coords.shape[1] == 3 else det_coords
    else:
        img_slice = img
        logit_slice = det_logit
        coords_2d = det_coords

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(logit_slice, cmap="hot")
    axes[1].set_title("Detection logit")

    if coords_2d is not None and len(coords_2d) > 0:
        for ax in axes:
            ax.scatter(coords_2d[:, 1], coords_2d[:, 0], c="cyan", s=10, linewidths=0.5)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_edges(
    coords_t: np.ndarray,     # (N_t, 3) or (N_t, 2)
    coords_t1: np.ndarray,    # (N_t1, 3) or (N_t1, 2)
    edges: list[tuple[int, int]],
    spatial_ax: tuple[int, int] = (1, 2),
    title: str = "Tracking edges",
) -> plt.Figure:
    """Plot GT or predicted edges between two consecutive frames."""
    ay, ax = spatial_ax

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(coords_t[:, ax], coords_t[:, ay], c="blue", label="frame t", s=20)
    axis.scatter(coords_t1[:, ax], coords_t1[:, ay], c="red", label="frame t+1", s=20)

    for (i, j) in edges:
        x_vals = [coords_t[i, ax], coords_t1[j, ax]]
        y_vals = [coords_t[i, ay], coords_t1[j, ay]]
        axis.plot(x_vals, y_vals, "k-", linewidth=0.5, alpha=0.6)

    axis.set_title(title)
    axis.legend()
    axis.invert_yaxis()
    plt.tight_layout()
    return fig
