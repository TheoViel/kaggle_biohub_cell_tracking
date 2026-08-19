#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1

set -e

cd src

# Usage: ./predict_track.sh ../logs/2026-08-14/17/ [extra args...]
LOG_FOLDER="${1:?usage: ./predict_track.sh <log_folder> [extra args]}"
shift || true

torchrun --standalone --nproc_per_node=8 main_predict.py \
    --log_folder "$LOG_FOLDER" \
    "$@"
