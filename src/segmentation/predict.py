"""
Inference + post-processing for the 3D cell segmentation model.

Post-processing pipeline:
1. Threshold sigmoid(fg_logits) → foreground binary mask
2. Accumulate centroid votes: for each fg voxel, predict centroid = voxel + offset * (D/2, H/2, W/2)
3. Smooth vote map and find local maxima → candidate cell centres
4. Assign each fg voxel to its nearest centre → instance mask
"""
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import gaussian_filter, maximum_filter, label as scipy_label

from segmentation.post_process import postprocess
from model_zoo.models_seg import CELL_SEG_HEADS, define_model


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_model(
    checkpoint_path,
    in_channels=1,
    device='cuda',
    decoder_name='Unet',
    encoder_name='resnet18',
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Checkpoints written by train.py carry the architecture they were built with
    config = ckpt.get('model_config', {})
    model = define_model(
        decoder_name=config.get('decoder_name', decoder_name),
        encoder_name=config.get('encoder_name', encoder_name),
        n_channels=config.get('in_channels', in_channels),
        pretrained=False,
        heads=CELL_SEG_HEADS,
        use_3d=True,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()

def _pad_to_multiple(x, multiple=16):
    """Pad spatial dims to the next multiple so the U-Net stride chain works."""
    D, H, W = x.shape[-3:]
    pd = (-D) % multiple
    ph = (-H) % multiple
    pw = (-W) % multiple
    if pd or ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph, 0, pd))
    return x, (D, H, W)


@torch.no_grad()
def predict_volume(model, data, use_cellpose=False, device='cuda'):
    """
    Run inference on a single 3D image array.

    Args:
        img_array: (D, H, W) uint16 or float32 in any range
        device: torch device string

    Returns dict with keys: fg_logits, heatmap_logits, offsets — all numpy (D, H, W) / (3, D, H, W).
    """
    img = data['img'].astype(np.float32)
    if img.max() > 1.0:
        img /= 65535.0

    if use_cellpose:
        flows = data['flows']
        probas = sigmoid(data['probas'])
        # img = np.stack([img, probas], axis=0)
        img = probas[None]
    else:
        img = img[None]

    print(img.min(), img.max())

    x = torch.from_numpy(img[None]).to(device)   # (1, c, D, H, W)
    x_padded, orig_shape = _pad_to_multiple(x)

    with torch.amp.autocast('cuda'):
        preds = model(x_padded)

    D, H, W = orig_shape
    return {
        'fg_logits': preds['fg_logits'][0, 0, :D, :H, :W].float().cpu().numpy(),
        'heatmap_logits': preds['heatmap_logits'][0, 0, :D, :H, :W].float().cpu().numpy(),
        'offsets': preds['offsets'][0, :, :D, :H, :W].float().cpu().numpy(),  # (3, D, H, W)
    }

# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run(checkpoint_path, npz_path, output_path=None, device='cuda', **postprocess_kwargs):
    """
    Load model, run on an npz file, and optionally save results.

    Returns: (masks, centroids, preds)
    """
    model = load_model(checkpoint_path, device=device)
    data = np.load(npz_path)
    preds = predict_volume(model, data['img'], device=device)
    masks, centroids = postprocess(preds, **postprocess_kwargs)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            masks=masks,
            centroids=centroids,
            fg_prob=sigmoid(preds['fg_logits']),
            heatmap=sigmoid(preds['heatmap_logits']),
        )

    return masks, centroids, preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('npz_path')
    parser.add_argument('--output', default=None)
    parser.add_argument('--fg_thresh', type=float, default=0.5)
    parser.add_argument('--vote_sigma', type=float, default=2.0)
    parser.add_argument('--nms_size', type=int, default=7)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    masks, centroids, _ = run(
        args.checkpoint, args.npz_path, args.output,
        device=args.device,
        fg_thresh=args.fg_thresh,
        vote_sigma=args.vote_sigma,
        nms_size=args.nms_size,
    )
    print(f"Found {len(centroids)} cells | mask shape {masks.shape} | unique IDs {masks.max()}")
