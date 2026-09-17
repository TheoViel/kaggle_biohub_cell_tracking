#!/usr/bin/env bash
# End-to-end evaluation of one division-classifier run, on the metric that counts.
#
#   eval_fork_model.sh <run_dir> <name> [gpu] [embryo]
#
# AP and AUC rank forks; the competition scores a graph. This scores the graph:
# the model's probabilities go onto every fork, low-scoring forks are deleted,
# `prune_short` is re-run (worth +0.0069 macro -- without it 6bba is negative),
# and the sweep reports adj_edge_jaccard and division_jaccard per threshold.
#
# Fold-1-only runs can only score 6bba, which is the embryo fold 1 held out.
# `score_forks.py` skips an embryo it has no out-of-fold model for rather than
# scoring it with a model that trained on it.
#
# 6bba reference points, same pipeline, from division_comparison_reprune.md:
#   production chain            cv 0.8429   adj 0.8264   div_J 0.1647
#   current CNN + reprune       cv 0.8481   adj 0.8250   div_J 0.2304
set -euo pipefail

RUN=${1:?run_dir with fold_<n>/ckpt_final.pt}
NAME=${2:?short name for the output files}
GPU=${3:-0}
EMB=${4:-6bba}

ROOT=/data/kaggle_biohub_cell_tracking
OUT=$ROOT/logs/fork_cls
PREDS=$ROOT/logs/edge_run/2026-09-13/0/preds_0.45
cd "$ROOT/src"

CUDA_VISIBLE_DEVICES=$GPU "$ROOT/.venv/bin/python" -u division_cls/score_forks.py \
  --candidates "$OUT/forks_2026-09-13_0.npz" \
  --run_dir "$RUN" --out "$OUT/scores_$NAME.npz"

"$ROOT/.venv/bin/python" -u division_cls/tune_fork_threshold.py \
  --preds_dir "$PREDS" --scores "$OUT/scores_$NAME.npz" \
  --embryo "$EMB" --workers 26 --reprune \
  --taus 0,0.1,0.3,0.5,0.7,0.9
