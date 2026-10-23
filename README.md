# Biohub – Cell Tracking During Development

10th place solution to the
[Biohub – Cell Tracking During Development](https://kaggle.com/competitions/biohub-cell-tracking-during-development)
Kaggle competition. The approach is described in the
[write-up](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/10th-place-solution-grandmaster-powered-agentic-a).

## The competition

![Pipeline overview](assets/44b6_e31261b4_t40_fast.gif)

The task is to track cells through 3D+time microscopy videos of developing
embryos: detect every cell in every frame, and link each cell to itself in the
next frame, or to its two daughters when it divides. Each video is an OME-Zarr
volume `(T, Z, Y, X)` with voxels of 1.625 × 0.406 × 0.406 µm. The
training set covers two embryos, `44b6` and `6bba`, and the public and private
test sets are two further embryos.

The ground truth is sparse. Only a few percent of the cells are annotated, as
[tracksdata](https://github.com/royerlab/tracksdata) `.geff` graphs whose nodes
are cell centres `(t, z, y, x)` and whose edges link a cell to the next frame.
A division is a node with two children. Predictions are scored only where the
annotation can judge them, by edge Jaccard (adjusted for the number of
predicted nodes) plus 0.1 × division Jaccard. See [`metrics.md`](metrics.md).

## This repository

![Pipeline overview](assets/pipeline.png)

It contains the training code for the models used by Theo's part of the pipeline.
- 3D cell detectors
- A Contrastive model that gives each cell an appearance embedding
- A Transformer that scores candidate links,
- The post-processing that builds the tracks
- An image classifier that cleans up the divisions.

`train.sh` runs every step below, from raw data to the folders the notebook
loads:

```bash
uv sync                  # 1. environment
# 2. data: see "Data" below
./train.sh               # 3. every training step, in order
```

We share trained weights for the final models on Kaggle:
[biohub-weights-2](https://www.kaggle.com/datasets/theoviel/biohub-weights-2) and
[biohub-weights-3](https://www.kaggle.com/datasets/theoviel/biohub-weights-3).

Inference runs locally in `notebooks/Inference.ipynb` (see "4. Inference").

## Pipeline

| Step (`train.sh`) | Trains / produces | Notebook variable | Code |
|---|---|---|---|
| `detector_r0` | bootstrap detector: synthetic-pretraining checkpoint + fold models for pseudo-labels | — | `src/main_track.py`, `src/tracking/` |
| `pseudo_labels` | pseudo-labels on the train split | — | `src/main_predict.py --pl`, `src/main_postprocess.py --pl` |
| `detector` | the base detector (temporal 3D U-Net) and its out-of-fold detections | `TRACK_W_BASE` | `src/main_track.py`, `src/main_predict.py` |
| `detector_add` | a second detector, same network, and its out-of-fold detections | `TRACK_W_ADD` | `src/main_track.py`, `src/main_predict.py` |
| `merge` | node set: base detections + the add detector's extra ones | — | `scripts/merge_detectors.py` |
| `netr` | NETr node embeddings, written into the node set | `CONTRASTIVE_W` | `src/main_contrastive.py`, `src/inference_contrastive.py`, `src/embeddings/` |
| `edge` | edge transformer (HOCT) and its out-of-fold link candidates | `EDGE_W` | `src/main_edge.py`, `src/main_edge_predict.py`, `src/edge_transformer/` |
| `forks` | the division forks the chain proposes, labelled against the annotation | — | `src/division_cls/gen_fork_candidates.py` |
| `division` | the division classifier | `DIVISION_W` | `src/main_division.py`, `src/division_cls/` |
| `package` | copies every model the notebook loads into `dataset/` | — | `scripts/copy_to_dataset.py` |

Each step trains on the previous step's output. The post-processing the notebook
applies to the edge candidates (ILP, gap recovery, fork harvest) is
`src/main_edge_postprocess.py` and `src/postprocessing/`; the `edge` step also
scores it out of fold.

## 1. Environment

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync      # creates .venv/ from uv.lock
```

`train.sh` uses `.venv/bin/python` (override with `PY=...`) and one process per
visible GPU (override with `N_GPUS=...`). The pipeline was last run on 8 GPUs
(the shipped models) and smoke-tested on 4×H100 80 GB.

## 2. Data

Everything goes under `input/` and `logs/`, both git-ignored.

**Competition data** (~80 GB). Needs a Kaggle API token
([docs](https://github.com/Kaggle/kaggle-api#api-credentials)) and the
competition rules accepted:

```bash
uv run kaggle competitions download -c biohub-cell-tracking-during-development -p input
unzip -q input/biohub-cell-tracking-during-development.zip -d input
rm input/biohub-cell-tracking-during-development.zip
```

This gives `input/train/{name}.zarr` + `{name}.geff` (the annotated stacks;
some stacks have no `.geff` and are not used), `input/test/`,
`input/metadata.csv` and `input/sample_submission.csv`.

**Synthetic sequences** (1000 sequences, ~48 GB), used to pretrain
the detector. The generator (`src/data_gen/`) is calibrated on the first
embryo's stacks:

```bash
uv run python scripts/make_synthetic.py --version 5
ln -s v5 input/synthetic/v4
```

The detector reads `input/synthetic/v5` (synthetic pretraining) and
`input/synthetic/v4` (pretraining, mixed with real data). The shipped models
used a `v4` generated by an earlier version of the generator, data can be shared on request.
The current one gives identical data for both versions, hence the link.
To compare a generated set with the real data (contrast, cell size, density),
run `python -m data_gen.calibrate --real 40 --synth ../input/synthetic/v5` from
`src/`.

## 3. Training

```bash
./train.sh                         # every step
./train.sh netr edge               # only some steps
```

Each training step creates a run folder:

- the detector runs go in `logs/<date>/<n>/`,
- NETr runs in `logs/contrastive/<date>/<n>/`,
- edge runs in `logs/edge_run/<date>/<n>/`,
- division runs in `logs/division/<date>/<n>/`.

`train.sh` records each run folder in `logs/pipeline_runs.env`, and later steps
read it from there. So a step can be rerun alone, and it picks up the latest
runs of the steps before it. Each run trains one model per held-out embryo
(`fold_0` holds out `44b6`, `fold_1` holds out `6bba`). These give the
out-of-fold numbers and predictions that the next step trains on. Each run also
trains a `fold_fullfit` model on both embryos, and that is the one the notebook
loads.

Measured on 8 GPUs for comparable runs: a detector ~3.5 h, NETr ~40 min, edge
~35 min. There are three detector trainings (bootstrap, base, add).

What each step does:

1. **`detector_r0`**: a first detector. Stage 1 pretrains on synthetic
   sequences only and saves `fold_0/ckpt_synthetic.pt`. Stage 1.5 then trains
   on real stacks mixed with synthetic ones, with weak positives taken from
   image peaks. Two fold models, no fullfit.
2. **`pseudo_labels`**: each fold model predicts its own training embryo at
   thresholds 0.3–0.6. The best threshold is post-processed, without the
   node-dropping stages, into `<run>/preds_pl_pp/`.
3. **`detector`**: the base detector, with the shipped recipe
   (`main_track.py` defaults: its own synthetic pretraining, then 5000
   iterations on real stacks mixed 50/50 with synthetic ones), using step 2's
   pseudo-labels as weak positives. It then predicts the held-out embryos into
   `preds_0.30/` and `preds_0.40/` (NMS radius 6 µm).
4. **`detector_add`**: a second detector of the same architecture, starting
   from step 3's synthetic checkpoint, with 3000 iterations, a 40% synthetic
   share and more weight on the pseudo-labels. It predicts into `preds_0.30/`.
5. **`merge`**: keeps the base detections ≥ 0.45, and adds the add detector's
   detections ≥ 0.40 that lie more than 7 µm from every base detection, per
   frame. The output goes to `logs/ensemble/v4_retrain/`. The notebook checks
   that the edge model's node set has `v4_` in its name.
6. **`netr`**: trains NETr on the base detector's `preds_0.40` crops; half of
   the samples are consecutive frames matched by the annotation. Then it
   writes one embedding per merged node, out of fold, into the same npz files
   under `embeddings_contrastive_<date>_<n>`.
7. **`edge`**: trains the edge transformer on the merged nodes and their
   embeddings. It then writes every candidate link out of fold to
   `preds_0.45/`, and scores the notebook's post-processing on them.
8. **`forks`**: harvests the division forks the ILP proposes from those
   candidates. Each fork is labelled positive or negative where the annotation
   can say, and unlabelled elsewhere.
9. **`division`**: trains the ConvNeXt-tiny classifier on those forks
   (fullfit only). The notebook deletes the second child of every fork it scores
   below 0.5.
10. **`package`**: see the next section.

**Differences from the shipped models.** The shipped runs used intermediate
runs that no longer exist, or tools that were removed from this repository, so
a rerun differs in these places:

- **Pseudo-labels.** The base detector's came from a chain of earlier
  detectors (`2026-09-01/9`, which itself used `2026-08-31/33`). The add
  detector's came from a second chain (`2026-09-23/5`, `2026-09-23/3`) that also
  used FOCUS-3D segmentation. Steps 1–2 replace both with one bootstrap round.
- **Add detector.** It was also pretrained with 10% zebrahub crops, and with
  FOCUS-3D weak positives. Both are left out. Its synthetic checkpoint came
  from another run (`2026-09-22/10`); step 4 uses the base detector's.
- **NETr.** It was trained on the detections of an earlier detector
  (`2026-09-09/4`). Step 6 uses the base detector's instead.
- **Division classifier.** Its forks were harvested from an earlier edge run
  (`edge_run/2026-09-13/0`, on the base detector's nodes only). Step 8 uses the
  edge model of step 7.
- **Synthetic `v4`** (see "Data").

Everything else (configurations, thresholds, the merge, the post-processing)
is the shipped recipe. The run folders the shipped models came from are the
`main_*.py` defaults.

**Smoke test.** `SMOKE=1 ./train.sh` runs every step on 3 stacks per embryo
and 8 synthetic sequences, with 4 iterations per training stage. It writes to
`input_smoke/` and `logs_smoke/` and takes about an hour on 4 GPUs. It only
checks that the chain runs: the models it trains detect nothing. So in smoke
mode, the merge reads the shipped detectors' predictions
(`logs/2026-09-13/1`, `logs/2026-09-24/5`), NETr crops around the merged nodes,
and the forks are harvested from the shipped edge run's candidates
(`logs/edge_run/2026-09-24/4`).

## 4. Inference

`notebooks/Inference.ipynb` runs the whole pipeline on the stacks in
`input/test/`: both detectors, the merge, NETr, the edge transformer,
post-processing and the division classifier. It writes every intermediate step
and `submission.csv` to `output/inference/`. Run it with the `.venv` Jupyter kernel; it
uses every visible GPU, and takes about 7 minutes for the 4 local test stacks on
4 GPUs.

By default it loads the shipped models from their run folders in `logs/`:

```
TRACK_W_BASE  = logs/2026-09-13/1/
TRACK_W_ADD   = logs/2026-09-24/5/
CONTRASTIVE_W = logs/contrastive/2026-09-09/1/
EDGE_W        = logs/edge_run/2026-09-24/4/
DIVISION_W    = logs/division/2026-09-16/0/
```

Set `USE_RETRAINED = True` in its configuration cell to load the latest
`./train.sh` run instead, from `logs/pipeline_runs.env`. The detector and edge
steps run `src/infer_detector.py` and `src/infer_edge.py`, which are
`main_predict.py` and `main_edge_predict.py` for folders without ground truth.

`./train.sh package` (included in a full run) also copies the retrained models
into `dataset/`, in the layout of the Kaggle weights datasets:

```
tracking_<date>_<n>/        -> TRACK_W_BASE   (base detector)
tracking_<date>_<n>/        -> TRACK_W_ADD    (add detector)
contrastive_<date>_<n>/     -> CONTRASTIVE_W
edge_<date>_<n>/            -> EDGE_W
division_<date>_<n>/        -> DIVISION_W
```

To publish them, upload it as a new version of the weights dataset
(`dataset/dataset-metadata.json` holds its id):

```bash
uv run kaggle datasets version -p dataset -m "retrain" --dir-mode zip
```
