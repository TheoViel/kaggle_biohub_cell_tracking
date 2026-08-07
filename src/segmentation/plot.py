import matplotlib.pyplot as plt
import numpy as np


def plot_offsets(offsets, z=32):
    dz_n, dy_n, dx_n = offsets[0], offsets[1], offsets[2]  # normalised, (D,H,W)
    magnitude = np.sqrt(dz_n**2 + dy_n**2 + dx_n**2)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"Offset targets at z={z}", fontsize=13, y=1.01)

    kw = dict(interpolation="none")
    im0 = axes[0].imshow(magnitude[z], cmap="magma", **kw)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[0].set_title("|offset| (normalised)")

    for i, (comp, label) in enumerate(zip([dz_n, dy_n, dx_n], ["Δz", "Δy", "Δx"]), start=1):
        vmax = max(abs(comp[z].min()), abs(comp[z].max())) + 1e-6
        im = axes[i].imshow(comp[z], cmap="bwr", vmin=-vmax, vmax=vmax, **kw)
        plt.colorbar(im, ax=axes[i], fraction=0.046)
        axes[i].set_title(f"{label} (normalised)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()



def plot_sample(img, masks, fg_soft, heatmap, z=32):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"z = {z}  (mid-slice)", fontsize=13, y=1.01)

    # Raw image
    axes[0].imshow(img[z], cmap="gray", interpolation="none")
    axes[0].set_title("Image (uint16)")

    # Instance masks
    mask_slice = masks[z].astype(float)
    mask_slice[mask_slice == 0] = np.nan
    axes[1].imshow(img[z], cmap="gray", alpha=0.5, interpolation="none")
    axes[1].imshow(mask_slice, cmap="nipy_spectral", alpha=0.6, interpolation="none")
    axes[1].set_title("Instance masks")

    # Foreground probability
    im = axes[2].imshow(fg_soft[z], cmap="hot", vmin=0, vmax=1, interpolation="none")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    axes[2].set_title("FG probability (target)")

    # Heatmap
    im2 = axes[3].imshow(heatmap[z], cmap="inferno", vmin=0, vmax=1, interpolation="none")
    plt.colorbar(im2, ax=axes[3], fraction=0.046)
    axes[3].set_title("Cell-centre heatmap (target)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_preds(
    vol_img,
    masks=None,
    fg_soft=None,
    heatmap=None,
    pred_masks=None,
    pred_centroids=None,
    pred_fg=None,
    pred_heatmap=None,
    z=32,
):
    if pred_masks is None or pred_fg is None or pred_heatmap is None:
        raise ValueError("pred_masks, pred_fg, and pred_heatmap must be provided")

    gt_values = (masks, fg_soft, heatmap)
    has_gt = all(value is not None for value in gt_values)
    if any(value is not None for value in gt_values) and not has_gt:
        raise ValueError("masks, fg_soft, and heatmap must be provided together")

    nrows = 2 if has_gt else 1
    fig, axes = plt.subplots(nrows, 4, figsize=(20, 5 * nrows), squeeze=False)
    title = (
        f"Ground truth (top) vs Predictions (bottom)   z={z}"
        if has_gt
        else f"Predictions   z={z}"
    )
    fig.suptitle(title, fontsize=13)

    if has_gt:
        axes[0, 0].imshow(vol_img[z], cmap="gray", interpolation="none")
        axes[0, 0].set_title("Image")

        ms = masks[z].astype(float)
        ms[ms == 0] = np.nan
        axes[0, 1].imshow(vol_img[z], cmap="gray", alpha=0.4, interpolation="none")
        axes[0, 1].imshow(ms, cmap="nipy_spectral", alpha=0.6, interpolation="none")
        axes[0, 1].set_title("GT instance masks")

        im = axes[0, 2].imshow(
            fg_soft[z], cmap="hot", vmin=0, vmax=1, interpolation="none"
        )
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        axes[0, 2].set_title("GT FG probability")

        im2 = axes[0, 3].imshow(
            heatmap[z], cmap="inferno", vmin=0, vmax=1, interpolation="none"
        )
        plt.colorbar(im2, ax=axes[0, 3], fraction=0.046)
        axes[0, 3].set_title("GT heatmap")

    pred_row = 1 if has_gt else 0
    axes[pred_row, 0].imshow(vol_img[z], cmap="gray", interpolation="none")
    axes[pred_row, 0].set_title("Image")

    pm = pred_masks[z].astype(float)
    pm[pm == 0] = np.nan
    axes[pred_row, 1].imshow(vol_img[z], cmap="gray", alpha=0.4, interpolation="none")
    axes[pred_row, 1].imshow(pm, cmap="nipy_spectral", alpha=0.6, interpolation="none")
    # overlay predicted centroids in this slice
    if pred_centroids is not None and len(pred_centroids):
        in_slice = pred_centroids[np.abs(pred_centroids[:, 0] - z) <= 2]
        axes[pred_row, 1].scatter(
            in_slice[:, 2],
            in_slice[:, 1],
            c="yellow",
            s=30,
            marker="x",
            linewidths=1.5,
            label="pred centres",
        )
    axes[pred_row, 1].set_title("Pred instance masks")

    im3 = axes[pred_row, 2].imshow(
        pred_fg[z], cmap="hot", vmin=0, vmax=1, interpolation="none"
    )
    plt.colorbar(im3, ax=axes[pred_row, 2], fraction=0.046)
    axes[pred_row, 2].set_title("Pred FG probability")

    im4 = axes[pred_row, 3].imshow(
        pred_heatmap[z], cmap="inferno", vmin=0, vmax=1, interpolation="none"
    )
    plt.colorbar(im4, ax=axes[pred_row, 3], fraction=0.046)
    axes[pred_row, 3].set_title("Pred heatmap")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()