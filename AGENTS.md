# AGENTS.md

Guidance for coding agents working in this repository. Read `README.md` first:
it describes the competition, the pipeline and every training step.

## What this repository is

Training code for the models loaded by the inference notebook
`notebooks/Inference.ipynb`: two 3D detectors (base and add), NETr
(contrastive embeddings), an edge transformer (HOCT) and a division classifier,
plus the post-processing it runs. Anything not needed to retrain those models or to
run the notebook has been removed; do not reintroduce experiment scripts.

## Layout

```
train.sh                 every training step, end to end (see README "3. Training")
scripts/                 merge_detectors.py, make_synthetic.py, copy_to_dataset.py
src/params.py            paths and constants (SCALE, EMBRYOS, INPUT_FOLDER, LOG_FOLDER)
src/main_*.py            training / prediction entry points, one per stage
src/infer_detector.py, src/infer_edge.py, src/inference_contrastive.py, src/submission.py   (inference)
src/tracking/            detector: data, model building, training, prediction
src/model_zoo/           networks (temporal U-Net, node transformer, HOCT)
src/embeddings/          NETr
src/edge_transformer/    edge transformer
src/division_cls/        division classifiers, fork harvesting, the notebook's fork gate
src/postprocessing/      ILP, gap recovery, divisions (imported by bare module name)
src/tracking_cellmot/    I/O and the competition metric
src/data_gen/            synthetic data generator
notebooks/               Inference.ipynb
```

`input/`, `logs/`, `output/`, `ensemble/`, `dataset*/` and the `*_smoke/`
folders hold data and runs. They are git-ignored, large, and in places the only
copy of a trained model: never delete or overwrite them.

## Running things

- Environment: `uv sync`, then `.venv/bin/python`. The system Python lacks the
  dependencies.
- Every entry point in `src/` runs **from `src/`**: paths in configs and
  `args.json` files are relative to it (`../input/`, `../logs/`).
  `postprocessing/` is put on `sys.path` by its callers, and its modules import
  each other by bare name (`from recover import ...`).
- `BIOHUB_INPUT` / `BIOHUB_LOGS` (in `src/params.py`) redirect the data and log
  roots. `train.sh` sets them.
- Training uses `torchrun`, one process per GPU; `train.sh` builds the command.

## Conventions that matter

- **Configuration is the `DEFAULTS` dict** at the top of each `main_*.py`,
  exposed as CLI flags. A dict-valued block is overridden by passing one JSON
  object, merged key by key with the defaults:
  `--pretraining '{"iterations": 4}'`. Passing the same flag twice keeps only
  the last one. Booleans use `--flag` / `--no-flag`.
- **The defaults are the shipped recipes.** `main_track.py` (the base
  detector), `main_contrastive.py`, `main_edge.py` and `main_division.py`
  reproduce the `args.json` of the runs the inference notebook loads. Do not
  change a default to run an experiment; pass a flag.
- **Folds are by embryo**: `fold_0` holds out `44b6`, `fold_1` holds out
  `6bba`. `fold_fullfit` trains on both and is what the notebook loads. Never
  split at random. A cell appears at many frames, so a random split leaks.
- **Run folders** are created by `utils.logger.prepare_log_folder` as
  `<log_path>/<date>/<n>/`. Each one holds `args.json`, `logs.txt` and
  `fold_*/ckpt_final.pt`. Checkpoints are rebuilt from `args.json`, so a
  renamed or removed config key must still load old runs. Read it with
  `getattr(args, key, default)`.
- **Code style**: match the surrounding code. Comments explain *why*, often
  with the measurement that justified a choice; keep that density. Line length
  120 (ruff config in `pyproject.toml`).

## The inference notebook depends on this code

`notebooks/Inference.ipynb` runs, from `src/`, the command lines of
`infer_detector.py`, `../scripts/merge_detectors.py`, `inference_contrastive.py`,
`infer_edge.py`, `main_edge_postprocess.py`, `division_cls/apply_fork_gate.py`
and `submission.py`. Keep those stable, or update the notebook in the same
change. It also checks that the edge model's `nodes_folder` contains `v4_`, and
that its `embeddings_key` matches the NETr run.

Pushing to `master` still zips `src/` into the `biohub-cell-tracking-code`
Kaggle dataset (`.github/workflows/main.yaml`).

## Verifying a change

There are no unit tests. After changing code:

1. Run `.venv/bin/python -m py_compile` on the changed files.
2. Load the shipped checkpoints if a model or its config changed: build the
   model from the run's `args.json` and `load_state_dict(..., strict=True)`.
   The shipped runs are in `dataset/` (`tracking_2026-09-13_1`,
   `contrastive_2026-09-09_1`, `division_2026-09-16_0`) and `dataset_2/`
   (`tracking_2026-09-24_5`, `edge_2026-09-24_4`).
3. For anything touching a training step, run `SMOKE=1 ./train.sh <steps>`
   (the whole chain takes about an hour on 4 GPUs). It runs the real code on 3 stacks
   per embryo and a few iterations, in `input_smoke/` and `logs_smoke/`. It
   checks that the chain runs, not that the models are good.

Scores move by about 0.0015 between seeds of the same configuration, so a
smaller difference is noise.
