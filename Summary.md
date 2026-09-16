# Pipeline Summary

What is actually in the pipeline, stage by stage, with the configuration of the
best chain to date. For the competition itself — task, data formats,
submission CSV, and the exact metric definition — see [`summary.md`](summary.md)
and [`metrics.md`](metrics.md); nothing here repeats them.

**Best chain as of 2026-09-15: `cv 0.8699`**

| | |
|---|---|
| edge model | `logs/edge_run/2026-09-13/0` |
| node set | `logs/2026-09-13/1/preds_r6_0.40/` |
| embeddings | `embeddings_contrastive_2026-09-09_1` (`logs/contrastive/2026-09-09/1`) |
| scored file | `logs/edge_run/2026-09-13/0/scores_preds_0.45_pp2.json` |

```
macro   score 0.8699   adj_edge_jaccard 0.8538   division_jaccard 0.1615   node_recall 0.9578
44b6    score 0.9016   adj 0.8860   div 0.1562   recall 0.9866   (n=71)
6bba    score 0.8382   adj 0.8215   div 0.1667   recall 0.9290   (n=128)
```

`cv` is the macro score — `mean(score_44b6, score_6bba)`, one vote per embryo.
The sample-weighted pooling was removed on purpose: the embryos differ 71 vs
128 in sample count and pooling lets 6bba decide the number on its own. Any
`scores_*.json` in the repo reports the macro figure as `cv_score`.

---

## The chain

```
   .zarr image
        |
  [1] detector            main_track.py            -> logs/<date>/<id>/fold_*/ckpt_*.pt
        |                                              TemporalUNet3D + detection heads
  [2] node extraction     main_predict.py          -> preds_r<radius>_<threshold>/*.npz
        |                                              coords, edges, node_probs, annot_probs
  [3] embeddings          inference_contrastive.py -> the same npz, one extra array
        |                                              embeddings_contrastive_<run>
  [4] edge model          main_edge.py             -> logs/edge_run/<date>/<id>/
        |                 main_edge_predict.py     -> preds_<link_threshold>/*.npz + candidates
        |                                              every candidate link, scored
  [5] edge postprocess    main_edge_postprocess.py -> preds_<th>_pp*/ + geff_<th>_pp*/
        |                                              assign -> recover -> divisions -> prune -> linefit
  [6] submission          submission.py            -> submission.csv
```

Entry points are the three shell scripts at the repo root — `track.sh`,
`contrastive.sh`, `edge.sh` — each of which sets `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` before `torchrun`. That is not decoration: several stages
call torch from inside a worker pool, and its intra-op pool sizes itself from
the core count regardless of those variables, so the post-processing modules
also call `torch.set_num_threads(1)` at import.

---

## [1] Detector — `main_track.py`

`TemporalUNet3D` (`model_zoo/temporal_unet.py`) plus a `SimpleNodeTransformer`
edge head, wired together in `model_zoo/models_track.py` as
`UNetNodeTransformer`. The detector is a CenterNet-style heatmap of cell
*centres*, trained with penalty-reduced focal loss on Gaussian-splatted
targets — not a segmentation, because the ground truth is integer centroids
and the metric is node/edge based.

Architecture of `logs/2026-09-13/1`:

| | |
|---|---|
| `unet_layers` | `[64, 128, 256]`, `blocks_per_stage [3, 3, 3]` |
| `unet_pool_stride` | `(1, 2, 2)`, `n_anisotropic 2` |
| `unet_stem` | `unshuffle` (folds the first downsample into channels rather than discarding 3/4 of the Y/X detail) |
| `unet_n_decode_steps` | 1 → node grid `(1, 4, 4)`, which is isotropic at 1.625 µm |
| `unet_n_extra_decode_steps` | 1 → detection grid `(1, 2, 2)`, with the coarse head kept as deep supervision |
| `unet_hires_fusion` | True |
| `unet_temporal_attention` | True (per-voxel attention across the 2-frame window) |
| `annot_head` | **True** |
| `offset_head` | False |
| `window_size` | 2 |

**The extra decode step** produces a finer detection map while `self.head` stays
on the node grid, so node features and every distance measured on that grid are
unchanged. The coarse head supervises the fine one (`det_ds_weight 1.0`), and
the intermediate head's hidden state plus its logit are concatenated into the
fine head's input rather than discarded.

**`annot_head`** is a second detection head trained to separate "annotated
cell" from "cell". The hypothesis was that the public models score well partly
because they are trained only on annotated cells. It is fed the same input as
the fine detection head, with `annot_neg_weight 1e-3` to survive the much
smaller positive set. Alone it is a weaker detector (AUC 0.6198 against the
node-probability control's 0.5949, and `annot − det` = 0.5454, i.e. no
information beyond the control), but the run carrying it is the best detector
to date and its node set is the one in use. Every `main_predict` sweep writes
both heads' node sets; the `_annot_head` suffix distinguishes them.

### Training recipe

Three stages, all in one `main_track.py` invocation:

1. **`synthetic_pretraining`** — 1,000 iterations on `input/synthetic/v5/` only
   (`synthetic_proportion 1`), `aug_strength 3`, no weak positives.
2. **`pretraining`** — 5,000 iterations, real data mixed 50/50 with
   `input/synthetic/v4/`, crops `32×128×128`, `det_neg_weight 0.05`.
3. **main fit** — full frames, `aug_strength 1`, edge head enabled
   (`transformer_only: True`), `det_neg_weight 1`.

`fullfit: True` trains a final model on everything after the two folds, and
`fold_fullfit/ckpt_final.pt` is what inference uses — the out-of-fold split has
no meaning on a test set whose embryos are in neither fold.

### Weak positives and pseudo-labels

The annotation is sparse, so an unlabelled cell is not background. `losses.py`
supports a third state: `weak_positives` are detections from a previous run
(`pl_folder`), splatted as low-weight positives (`det_weak_weight 0.2`) and
excluded from the negative set within `ignore_min_dist_um 7.0`. The current run
reads `logs/2026-09-01/9/preds_pl_pp/`, whose measured quality is
`node_recall 0.9718` macro (scored after the fact into
`logs/2026-09-01/9/scores_pp_pl.json`; in-sample, since a PL pass runs on its
own training embryos).

`main_postprocess.py --pl` produces these folders and deliberately runs a
**different chain**: `tracking/dataset.py::_load_pl` reads only `coords` from
the npz and never the edges, so a node dropped there is a real cell the next run
is trained to call background. Short-track pruning and isolated-node pruning are
switched off under `--pl` (`PL_PIPELINE` / `PL_PP_PUBLIC`), measured at +3.74%
nodes kept and node recall 0.9498 → 0.9534; `--pl_prune` restores the val chain.

---

## [2] Node extraction — `main_predict.py`

Runs the detector over sliding windows and turns heatmaps into coordinates.
`track.sh` sweeps `--det_threshold 0.4,0.5,0.6 --nms_radius_um 5,6,7`, writing
one `preds_r<radius>_<threshold>/` folder per grid point plus `_annot_head`
variants — 18 node sets per run. Each npz carries `coords`, `edges`,
`node_probs`, `annot_probs` and `coords_f` (the unrounded positions).

Detection is greedy NMS at an exact physical radius (`--nms greedy`), not
max-pooling, so the radius means the same thing regardless of grid stride.

Two options that exist and are off by default:

- **`--n_tta {1,4,8}`** — axis-flip TTA averaged in *logit* space (the heads are
  trained on logits and the sigmoid is monotone but not affine). 4 flips Y/X, 8
  adds Z, which is in-distribution here because training flips all three. Adds a
  `_tta<N>` suffix to the folders and scores.
- **`merge_windows`** (on by default, `--no_window_merge` to revert) — with
  `window_size=2` and the stride-1 sweep, frame *t* appears in windows
  `(t-1, t)` and `(t, t+1)`. The old first-wins rule gave every frame its
  **left** neighbour only, discarding half the temporal evidence at no saving,
  since both forwards happen anyway. Measured on fold 0: raw cv +0.0027, but
  post-PP only +0.0002 — edge jaccard +0.0010 against division jaccard −0.0081.

---

## [3] Contrastive embeddings — `inference_contrastive.py`

`logs/contrastive/2026-09-09/1`: a DINO ViT-B/16 backbone over `(3, 48, 48)`
patches, one frame at a time, with a 4-layer / 256-d transformer on top
(`use_visual: true`, `use_position: false` — appearance only, so the edge model
is not handed the geometry twice). Trained 1,000 iterations, `fullfit`.

Inference writes a `(N, 256)` L2-normalised array back into the *same* npz under
`embeddings_contrastive_<date>_<id>`, keyed to the node coordinates it was run
on — so a node set and its embeddings are inseparable, and the edge model's
`embeddings_key` must name the run that produced them. Cosine of that embedding
separates truly-linked pairs from random ones at 0.874 against 0.065 median,
which is cleaner separation than the detector's own edge probabilities.

With no arguments the script picks the latest contrastive run and the
coordinates folder recorded in its args.

---

## [4] Edge model — `main_edge.py` / `main_edge_predict.py`

`SimpleNodeTransformer` scoring every candidate link. Candidates are the 5
nearest neighbours within 20 µm across one frame (`max_dt 1`), which is what
makes the problem tractable; everything outside that is unlinkable by
construction.

Configuration of `logs/edge_run/2026-09-13/0`:

| | |
|---|---|
| `d_model` | 288, 4 heads, 4 node blocks + 4 edge blocks |
| `n_neighbors` / `max_distance_um` | 5 / 20.0 |
| `use_embeddings` | True (`embeddings_contrastive_2026-09-09_1`) |
| `use_conf_score` | True — the detector's per-node probability is a feature |
| `edge_drift_correction` | True |
| positional encoding | `pos_scale 20 µm`, wavelengths 4–200 µm, Householder |
| `attn_max_distance_um` | 40.0 |
| training | 2,000 iterations, AdamW 3e-4, focal γ=2, `fullfit` |

`main_edge_predict.py` sweeps link thresholds in one pass — scoring the
candidates is the GPU cost and does not depend on the threshold, only the
Hungarian linking does — writing `preds_<th>/` plus a `--dump_candidates`
array holding *every* scored candidate, which is what lets the post-processing
re-solve the assignment offline.

---

## [5] Edge post-processing — `main_edge_postprocess.py`

The stage that turns scored candidates into a graph. Nothing here needs a GPU
or the image data. Chain, in order:

```
assign (+ per-track momentum) -> recover -> divisions -> prune -> linefit
```

The settings that produced **0.8699** (from the run's own `settings` block):

| knob | value | |
|---|---|---|
| `threshold` | 0.40 | link probability to commit |
| `vel_weight` / `vel_momentum` | 0.02 / 0.7 | log-odds charged per µm a link departs from the track's momentum |
| `recover_max_gap` | 3 | longest run of missing frames to fill |
| `recover_max_residual` | 5.0 | µm of drift-corrected motion allowed across a one-frame gap |
| `recover_min_dup` | 3.0 | µm; do not insert beside an existing detection |
| `div_min_prob` | 0.25 | the model's probability for the parent/second-child pair |
| `div_diverge_um` | 4.0 | separation the grandchildren must reach by t+2 |
| `div_max_parent_um` | 11.0 | |
| `div_global_frac_cap` | 0.00375 | |
| `min_track` | 5 | drop chains shorter than this |
| `linefit` / `linefit_window` | 0.6 / 3 | blend each node toward a line along its own track |
| `ilp` | 0 | off |

**These are not the values in `DEFAULTS`.** The file's defaults were tuned on
`edge_run/2026-09-11/0` and the four that differ here — `recover_max_residual`,
`recover_min_dup`, `div_min_prob`, `div_diverge_um` — were retuned for this
checkpoint. The knobs trade against how much the detector over-detects and how
well its probabilities separate a fork, neither of which is a property of the
post-processing, so **retune on every new checkpoint** and never read a scores
json written under one set of defaults against one written under another. The
same four cost −0.0066 when applied backwards to `edge_run/2026-09-09/0`.

`postprocessing/tune_edge_pp.py` automates that: coordinate descent over 12
knobs, scoring through the identical `postprocess` + `evaluate` path but
writing no npz or geff. `--both` makes "must not lose on either embryo" binding
rather than advisory.

### The other path — `main_postprocess.py`

Post-processes a graph whose edges are **already committed**, re-deciding them
on geometry alone. Used for scoring the detector without the edge model, and
for the `--pl` passes. Chain is `pipeline → pp_public(tuned) → divisions`,
where `pp_public` is the public 946-notebook repair with `safe_divisions` off,
since `run_divisions` is the same repair with the gates that one lacks.

The detector alone, through this path, scores **0.8378** — so the edge model
plus its own post-processing is worth about **+0.032**.

---

## Inference / submission

`notebooks/biohub-inf.ipynb` runs the whole chain on Kaggle: detector →
contrastive → edge model → edge post-processing → `submission.py`. Constraints
that are baked in and easy to break:

- **GPU memory must stay under 15,360 MiB.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is set in the first cell; without it allocator fragmentation reserved 48,980 MiB
  for the same work that otherwise takes 3,220.
- **`MAX_NODES`** caps the node set before the edge stage, keeping the most
  confident nodes. Ranking by confidence matters — the dataset's own cap drops
  at random, which is right for training and wrong here, because a dropped node
  gets no candidate edges at all.
- Every post-processing knob is passed **explicitly** on the command line rather
  than left to `DEFAULTS`, for the reason above.
- `predict_video_thresholds` returns results keyed `(head, radius, threshold)`
  with five values; ask for `"det"` — the node set the edge model was fitted on.

---

## Facts worth not rediscovering

- **Ground-truth coordinates are 100% integers.** There is no sub-voxel truth,
  which caps every refinement, fusion and offset-head idea.
- **z is the hard axis.** It carries 58.9% of the localisation MSE and the median
  error is exactly 1.625 µm — one voxel. The anisotropic schedule never pools z,
  so its displacement in *feature* voxels does not shrink as the decoder
  coarsens the way y/x does.
- **Frame-to-frame motion**, over 17,882 consecutive GT links: median 1.68 µm,
  p90 3.63, p99 7.56, max 21.05. Per axis in original voxels, p99 is z 4, y/x 12.
- **False negatives**, over 199 stacks of `2026-09-13/1/preds_r6_0.40`: of 7,490
  missed GT nodes, 10.7% have both temporal neighbours detected (bridgeable),
  32.7% exactly one, 56.6% neither. The ceiling on any temporal recovery is
  node recall 0.9438 → 0.9682.
- **Divisions are rare**: 26 in the 71 fold-0 stacks, 125 in fold 1. Fold-0
  results on the division stage move on single events and should not be ranked.
- **The node-count adjustment is signed and uncapped.** `adj = J · (1 − 0.1 · ratio)`
  with `ratio = (n_pred − n_est)/n_est`, so predicting *fewer* nodes than the
  organisers' estimate multiplies the score by more than 1. Any experiment that
  changes node count is partly measuring this and not link quality.
- **The edge linker is at ~95% of its reachable ceiling.** 8,707 GT edges are
  unreachable (an endpoint is undetected) against 5,764 reachable but unlinked,
  so detection, not linking, is the binding constraint.

## Measured, and negative

Recorded so nobody spends the compute twice.

| idea | result |
|---|---|
| `offset_head` | −0.0058 after post-processing; off in the best run |
| ILP node filter in the edge chain | +0.0002 for 9–16 s/stack once divisions are refit to its graph; off |
| Ultrack-style multi-hypothesis node selection | fold 0: 1 set −0.0039, 2 sets −0.0456. Pooling per-set *edges* shatters chains; associations must be recomputed over the pool |
| Terminal-extrapolation FN recovery | +0.0101 on FN-dense stacks, −0.0009 on a difficulty spread. Precision per insertion is the binding constraint, not locating the misses |
| Reverse-time division detection (as generator) | fold 0: best arm +0.0008 with 6,804 forks for 2 true positives. Thousands of track starts against 26 divisions — the base rate defeats it |
| Reverse-time division test (as filter) | +0.0003, drops 1 false fork of 11. Directionally right, inside the noise |

---

## Repo map

| path | |
|---|---|
| `src/main_track.py` | detector training, config `DEFAULTS` at the top |
| `src/main_predict.py` | node extraction, threshold × radius × head sweep, TTA |
| `src/inference_contrastive.py` | embeddings into the node npz |
| `src/main_edge.py` / `main_edge_predict.py` | edge model train / predict |
| `src/main_edge_postprocess.py` | the chain that produced the best score |
| `src/main_postprocess.py` | geometry-only path, `--pl` mode |
| `src/submission.py` | geff folder → CSV |
| `src/model_zoo/temporal_unet.py` | the detector backbone |
| `src/model_zoo/models_track.py` | heads + `UNET_BACKBONES` registry |
| `src/model_zoo/simple_node_transformer.py` | edge scorer, `estimate_drift` |
| `src/postprocessing/edge_pp.py` | `assign`, `recover_nodes`, `propose_divisions`, … |
| `src/postprocessing/tune_edge_pp.py` | coordinate descent over the PP knobs |
| `src/postprocessing/README_edge.md` | why each edge-PP stage exists |
| `Experiments.md` | every run with both folds, ranked |
| `summary.md`, `metrics.md` | the competition and the metric |
