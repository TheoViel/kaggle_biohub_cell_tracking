#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1

set -e

cd src

# Usage: ./predict_track.sh ../logs/2026-08-14/17/ [--pl] [extra args...]
#
#   (default)  score the fold's validation embryo  -> <log_folder>/preds
#   --pl       label the fold's TRAINING embryos   -> <log_folder>/preds_pl
#              (pseudo-labels for improving the segmentation model)
LOG_FOLDER="${1:?usage: ./predict_track.sh <log_folder> [--pl] [extra args]}"
shift || true

# --local-addr: this host's FQDN does not resolve, so torchrun's default
# lookup hangs when it hands the address to the workers.
torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=8 main_predict.py \
    --log_folder "$LOG_FOLDER" \
    --det_threshold 0.3,0.4,0.5,0.6 \
    "$@"


echo 
echo Predictions done ! Postprocessing...
echo

python main_postprocess.py --log_folder "$LOG_FOLDER"
