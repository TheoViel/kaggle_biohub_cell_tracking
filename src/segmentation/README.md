### What's in each file

- `targets.py` -  heatmap & offset targets from instance masks only (no reliance on the external centroids array). Offsets are normalized to [-1,1] so all loss terms are on the same scale.
- `dataset.py` - CellSegDataset — loads npz files, computes targets on-the-fly, supports random cropping (--patch_size) and augmentation (flips + intensity jitter). Flip-consistent: flipping axis k also negates offsets[k].
- `model.py` - CellSegNet — nnUNet-style 3D U-Net. Strided Conv downsampling, transposed Conv upsampling, InstanceNorm + LeakyReLU. Three output heads: fg_logits, heatmap_logits, offsets. Default 5M params (configurable).
- `losses.py` - CellSegLoss — BCE for foreground + BCE for heatmap + SmoothL1 for offsets (masked to fg). All three terms start ~equal thanks to offset normalization.
- `train.py` - Full training loop: mixed precision, gradient clipping, cosine LR, checkpoint save/resume.
- `predict.py` - Inference + post-processing: vote accumulation (offset + voxel_pos → predicted centroid), Gaussian smoothing + local maxima NMS, nearest-centroid assignment → instance masks.

### Running the pipeline

-To train:

`python -m src.segmentation.train --data_dir output/segs --output_dir output/models`

- To predict:

`python -m src.segmentation.predict output/models/ckpt_final.pt output/segs/vol_00000.npz.npz --output output/pred.npz`


- Pipeline

Visualize the steps in the `notebooks/nnUNet.ipynb`