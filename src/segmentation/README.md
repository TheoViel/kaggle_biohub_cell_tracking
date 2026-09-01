# 3D Cellpose segmentation

Cell detection as a *flow field* rather than a blob map, after Stringer et al.,
[Cellpose](https://www.nature.com/articles/s41592-020-01018-x) (Nat. Methods
2021), in native 3D.

The detector used elsewhere in this repo predicts a peak per cell and takes
local maxima. That fails in exactly one place, and it is the place that costs
the score: two nuclei pressed together share a ridge of high probability with no
minimum between them, so no threshold cuts them apart, and one bright cell with
chromatin texture peaks twice. Cellpose does not decide cells by thresholding at
all. The network outputs a vector field pointing at each cell's centre; cells
are whatever the field's trajectories converge onto. Touching cells have two
centres, so their voxels part company even where their probabilities never do.

## The pieces

| file | what it does |
|---|---|
| `flows.py` | instance masks → `(cellprob, flows)`. Heat diffused from each cell's centre, confined to the cell; flows are the normalised gradient of its log. |
| `dataset.py` | random crops of the synthetic **static** volumes, with the flow field cached to disk. |
| `losses.py` | `MSE(flows)/2 + BCE(cellprob)`, plus `flow_error`, the interior cosine that actually tracks reconstruction quality. |
| `dynamics.py` | the reconstruction heuristic: Euler-integrate every foreground voxel along the flow, histogram where trajectories land, cluster the peaks. |
| `predict.py` | inference on real frames, and the quantile normalisation the model was trained under. |
| `eval.py` | node recall and the node-count adjustment, on real annotated frames. |

Train with `python main_seg.py`; visualise with `notebooks/SegmentationInf.ipynb`.

## Why static volumes only

`generate_sequence` writes nodes and edges but no per-voxel labels, so the
*static* volumes are the only synthetic data carrying instance masks — and a
flow field can only be built from an instance mask. An epoch is therefore a
count of random crops (`epoch_size`), not a pass over the files.

## Reading the two metrics

They are reported together because either one alone is gameable.

- **Node recall** — annotated cells with a prediction within 7 µm. The real
  ground truth annotates ~3% of cells, so predictions *without* an annotation
  are usually unlabelled real cells, not false positives. Precision against this
  ground truth would be meaningless and is not reported.
- **Adjustment** — the competition's own penalty,
  `1 - 0.1 * (T_pred - T_true) / T_true`, against the dataset's estimate of the
  true cell count. Predicting everything scores perfect recall and a terrible
  adjustment.

`eval.sweep_threshold` is worth more than either number at a single operating
point: recall that holds while the node count falls means the flows are right
and the probability head is merely conservative, while recall that collapses as
the threshold rises means real cells are not the model's highest-scoring voxels
— which no post-processing recovers.

## The control to run first

`dynamics.masks_from_flows` fed the flows built from a ground-truth mask
recovers 448/448 cells at a median centroid error of 0.27 µm. Run that oracle
(last cell of the notebook) before debugging a bad score: it separates a model
problem from a reconstruction problem, and they have nothing to do with each
other.

## Known gaps

- Only 10 static volumes exist in `input/synthetic/v3`. Crops from ten fields do
  not cover the imaging regimes the generator can produce; raise `N_STATIC` in
  `SyntheticData.ipynb` before reading any result as a ceiling. Training warns
  when it sees fewer than 50.
- Training is 100% synthetic, so the same domain gap that cost the tracking
  detector 0.227 applies here. Validate on real frames, never on held-out
  synthetic.
- Geometric augmentation is restricted to flips and y/x 90° rotations, whose
  effect on a vector is exact. An arbitrary affine would have to rotate the
  vectors as well as resample them, and interpolating a rotated unit field
  shortens it wherever neighbouring vectors disagree.
- `targets.py` and `post_process.py` are the earlier CenterNet-style path
  (heatmap + centroid offsets). Nothing in the Cellpose pipeline imports them.
