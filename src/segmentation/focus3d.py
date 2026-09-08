"""
FOCUS-3D inference on arrays, without the file I/O.

The packaged ``infer_volume`` reads a tif off disk, rebuilds and reloads the
4.5 GB model on every call, writes the instance map back out and prints a step
log as it goes. Over a hundred timepoints that is a hundred model loads and a
hundred round trips through the filesystem for volumes we already have in
memory, so the pipeline below is the same one with those three things removed:
the model is loaded once by ``load_model`` and passed in, the volume is a numpy
array, and nothing is printed unless asked.

Every stage is the packaged function, so results match ``infer_volume`` for the
same parameters.
"""

import contextlib
import io

import numpy as np
import torch
from pathlib import Path

from focus3d.segmentation.FOCUS3D import inference_win as _f3d
from focus3d.segmentation.FOCUS3D.inference_win import (
    PatchDataset,
    build_predictor,
    compute_percentiles_fast,
    filter_instances_by_size,
    finalize_instance_and_confidence,
    generate_patch_coords,
    normalize_and_pad_volume,
    patch_collate_fn,
    patch_postprocess_argmax,
    resample_volume_for_inference,
    resize_volume_to_shape,
    setup_cfg,
    stitch_patch_instance_results,
)
from tqdm.auto import tqdm


# The only config shipped with the package, and the one the reference notebook
# uses. It is where the patch size and the model architecture come from.
CONFIG_FILE = str(Path(_f3d.__file__).resolve().parent / "configs" / "3d_test.yaml")

# The radius, in xy pixels, that the model was trained at. A volume is rescaled
# by reference_cell_radius / cell_radius before inference.
REFERENCE_CELL_RADIUS = 15.0


def load_model(weights_path, config_file=CONFIG_FILE, device=None, verbose=False):
    """
    Build a FOCUS-3D model and load a checkpoint into it, once.

    Args:
        weights_path (str or Path): Checkpoint, e.g. model_final_nuclei.pth.
        config_file (str or Path, optional): Model config.
        device (str, optional): Torch device. Defaults to cuda:0 if there is a
            GPU, else cpu.
        verbose (bool, optional): Let the package print which device and
            checkpoint it picked.

    Returns:
        The model, in eval mode on ``device``.
    """
    cfg = setup_cfg(config_file, weights_path, device=device)

    # build_predictor announces the checkpoint it is loading and there is no
    # flag for it, so the print is swallowed rather than patched out.
    quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with quiet:
        model = build_predictor(cfg)

    return model


def infer_volume(
    model,
    volume,
    z_ratio=1.0,
    cell_radius=REFERENCE_CELL_RADIUS,
    lower_percentile=1.0,
    upper_percentile=99.0,
    background_threshold=None,
    patch_size=(32, 96, 96),
    stride=None,
    batch_size=1,
    num_workers=0,
    score_thresh=0.5,
    mask_thresh=0.5,
    min_edge_area=20,
    topk_postprocess=300,
    size_filter_min_size=0,
    size_filter_max_size=None,
    use_amp=True,
    amp_dtype="float16",
    restore_resampled_mask=True,
    stitch_face_iom_thresh=0.2,
    stitch_min_core_voxels=64,
    stitch_min_contact_cc_pixels=10,
    stitch_min_contact_cc_ratio=0.5,
    stitch_contact_connectivity=2,
    return_confidence=False,
    verbose=False,
):
    """
    Segment one 3D volume into instances with an already loaded FOCUS-3D model.

    Args:
        model: Model from load_model.
        volume (np.ndarray): Raw volume as (z, y, x). Any dtype; it is
            normalized here by the percentiles below, so pass it unscaled.
        z_ratio (float, optional): Voxel spacing along z relative to xy, so 4.0
            for 1.625 um slices of 0.40625 um pixels. The volume is resampled to
            undo the anisotropy before inference.
        cell_radius (float, optional): Cell radius in xy pixels of the raw
            image. The volume is rescaled so this becomes 15 px, the radius the
            model was trained at, which makes it the parameter worth getting
            right before any threshold below.
        lower_percentile, upper_percentile (float, optional): Intensity
            percentiles mapped to 0 and 1.
        background_threshold (float, optional): Raw intensity under which a
            patch is skipped as empty. Defaults to the volume's 5th percentile.
        patch_size (tuple, optional): Sliding window, as (z, y, x).
        stride (tuple, optional): Step between windows. Defaults to half the
            patch, so neighbouring windows overlap and cells split across a
            border can be stitched back together.
        batch_size (int, optional): Patches per forward pass.
        num_workers (int, optional): Dataloader workers. Zero, unlike the
            package's four, because the patches come from an array already in
            memory and spawning workers per call costs more than it saves.
        score_thresh (float, optional): Instance confidence to keep. Lower it if
            cells are missing, raise it if dim areas produce spurious ones.
        mask_thresh (float, optional): Mask probability to keep a voxel.
        min_edge_area (int, optional): Smallest 2D component kept per slice.
        topk_postprocess (int, optional): Instance queries kept per patch.
        size_filter_min_size, size_filter_max_size (int, optional): Instance
            volume bounds, in voxels of the resampled image.
        use_amp (bool, optional): Autocast on CUDA.
        amp_dtype (str, optional): "float16" or "bfloat16".
        restore_resampled_mask (bool, optional): Resize the instance map back to
            the shape of ``volume``. Turning it off returns the map in the
            resampled frame, where the coordinates no longer match the input.
        stitch_* (optional): Passed to the patch stitcher; see the package.
        return_confidence (bool, optional): Also return the confidence map.
        verbose (bool, optional): Show the patch progress bar and the
            post-processing counts.

    Returns:
        np.ndarray: (z, y, x) uint32 instance map, one label per cell, aligned
            with ``volume``. With return_confidence, a (instance map, confidence
            map) pair, the second being float32 of the same shape.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"volume must be (z, y, x), got shape {volume.shape}")
    if cell_radius <= 0:
        raise ValueError(f"cell_radius must be positive, got {cell_radius}")
    if z_ratio <= 0:
        raise ValueError(f"z_ratio must be positive, got {z_ratio}")

    device = next(model.parameters()).device
    amp_enabled = bool(use_amp and device.type == "cuda")
    if amp_dtype not in ("float16", "bfloat16"):
        raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
    autocast_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16

    # These live on the model rather than in the call, so they have to be set
    # each time in case the same model is reused with different thresholds.
    model.fast_score_thresh = float(score_thresh)
    model.fast_mask_thresh = float(mask_thresh)

    # Resample so a cell is the size the model expects, and z is no longer
    # coarser than xy.
    raw_shape = volume.shape
    infer_volume_, resample_info = resample_volume_for_inference(
        volume=volume,
        cell_radius=cell_radius,
        z_ratio=z_ratio,
        reference_cell_radius=REFERENCE_CELL_RADIUS,
        order=1,
    )
    infer_shape = infer_volume_.shape

    patch_size = tuple(patch_size)
    stride = tuple(s // 2 for s in patch_size) if stride is None else tuple(stride)

    # One pass over the volume for every percentile needed, including the 5th
    # when the background threshold is left to us.
    percentiles = [float(lower_percentile), float(upper_percentile)]
    if background_threshold is None:
        percentiles.append(5.0)
    values = compute_percentiles_fast(infer_volume_, percentiles, chunk_z=8)

    p_low, p_high = float(values[0]), float(values[1])
    bg_raw = float(values[2]) if background_threshold is None else float(background_threshold)
    if p_high <= p_low:
        # A flat volume: normalization has nothing to stretch, so no patch can
        # be called background either.
        background_threshold_norm = 0.0
    else:
        background_threshold_norm = float(
            np.clip((bg_raw - p_low) / (p_high - p_low), 0.0, 1.0)
        )

    padded_volume, _ = normalize_and_pad_volume(
        volume=infer_volume_,
        p_low=p_low,
        p_high=p_high,
        patch_size=patch_size,
        stride=stride,
        chunk_z=8,
    )

    all_coords = generate_patch_coords(padded_volume.shape, patch_size, stride)
    max_coords = tuple(max(c[axis] for c in all_coords) for axis in range(3))

    pz, py, px = patch_size
    if background_threshold_norm <= 0.0:
        infer_coords = all_coords
    else:
        infer_coords = []
        for z0, y0, x0 in all_coords:
            patch = padded_volume[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
            # The max is the cheap exact test; the percentile then drops patches
            # carrying nothing but a few bright specks.
            if patch.max() < background_threshold_norm:
                continue
            if np.percentile(patch, 99.9) >= background_threshold_norm:
                infer_coords.append((z0, y0, x0))

    global_instance_map = np.zeros(padded_volume.shape, dtype=np.uint32)
    confidence_sum = np.zeros(padded_volume.shape, dtype=np.float32)
    confidence_count = np.zeros(padded_volume.shape, dtype=np.uint16)
    next_instance_id = 1

    loader = torch.utils.data.DataLoader(
        PatchDataset(
            padded_volume=padded_volume,
            infer_coords=infer_coords,
            patch_size=patch_size,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=patch_collate_fn,
    )

    for batch_images, batch_coords in tqdm(
        loader, desc="Inferencing patches", unit="batch", disable=not verbose
    ):
        batch_images = batch_images.to(device, non_blocking=True)
        inputs = [
            {"image": batch_images[i], "coord": batch_coords[i]}
            for i in range(batch_images.shape[0])
        ]

        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=amp_enabled),
        ):
            outputs = model(inputs)

        for sample, output in zip(inputs, outputs, strict=False):
            z0, y0, x0 = sample["coord"]
            post = patch_postprocess_argmax(
                model_output=output,
                patch_shape=patch_size,
                score_thresh=score_thresh,
                mask_thresh=mask_thresh,
                min_voxels=min_edge_area,
                topk_postprocess=topk_postprocess,
            )

            # Labels are merged into the global map as each patch arrives, so
            # the overlap between windows is resolved here rather than at the
            # end, and only the one global map is ever held.
            next_instance_id = stitch_patch_instance_results(
                global_instance_map=global_instance_map,
                confidence_sum=confidence_sum,
                confidence_count=confidence_count,
                patch_instance_map=post["patch_instance_map"],
                patch_confidence=post["patch_confidence"],
                z0=z0,
                y0=y0,
                x0=x0,
                next_instance_id=next_instance_id,
                max_coords=max_coords,
                patch_size=patch_size,
                stride=stride,
                face_iom_thresh=stitch_face_iom_thresh,
                stitch_min_core_voxels=stitch_min_core_voxels,
                min_contact_cc_pixels=stitch_min_contact_cc_pixels,
                min_contact_cc_ratio=stitch_min_contact_cc_ratio,
                contact_connectivity=stitch_contact_connectivity,
            )

    instance_map, confidence_map = finalize_instance_and_confidence(
        global_instance_map=global_instance_map,
        confidence_sum=confidence_sum,
        confidence_count=confidence_count,
    )

    instance_map, confidence_map = filter_instances_by_size(
        instance_map=instance_map,
        confidence_map=confidence_map,
        min_area=min_edge_area,
        intensity_volume=padded_volume,
        min_size=size_filter_min_size,
        max_size=size_filter_max_size,
        background_threshold=background_threshold_norm,
        verbose=verbose,
    )

    # Drop the padding, then undo the resampling so the labels line up with the
    # volume that came in.
    z, y, x = infer_shape
    instance_map = instance_map[:z, :y, :x]
    confidence_map = confidence_map[:z, :y, :x]

    if resample_info["enabled"] and restore_resampled_mask:
        instance_map = resize_volume_to_shape(
            instance_map.astype(np.uint32, copy=False), target_shape=raw_shape, order=0
        )  # nearest neighbour, so labels are not blended into new ones
        confidence_map = resize_volume_to_shape(
            confidence_map.astype(np.float32, copy=False), target_shape=raw_shape, order=1
        )

    instance_map = instance_map.astype(np.uint32, copy=False)
    confidence_map = confidence_map.astype(np.float32, copy=False)

    if verbose:
        print(
            f"{instance_map.max()} instances from {len(infer_coords)} patches "
            f"({len(all_coords) - len(infer_coords)} skipped as background)"
        )

    return (instance_map, confidence_map) if return_confidence else instance_map
