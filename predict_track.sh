#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

set -e

cd src

n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Usage: ./predict_track.sh [log_folder] [--pl] [extra args...]
#
#   (default)  score the fold's validation embryo  -> <log_folder>/preds
#   --pl       label the fold's TRAINING embryos   -> <log_folder>/preds_pl
#              (pseudo-labels for improving the segmentation model)
#
# The log folder is optional; without one both scripts below take the most
# recent run under ../logs/, which is what you want right after training. A
# leading "-" is an option rather than a path, so --pl alone still works.
LOG_ARGS=()
if [[ -n "${1:-}" && "$1" != -* ]]; then
    LOG_ARGS=(--log_folder "$1")
    shift
fi

# --local-addr: this host's FQDN does not resolve, so torchrun's default
# lookup hangs when it hands the address to the workers.
torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_predict.py \
    "${LOG_ARGS[@]}" \
    --det_threshold 0.4,0.5 \
    --nms_radius_um 6,7 \
    "$@"


echo
echo Predictions done ! Postprocessing...
echo

python main_postprocess.py "${LOG_ARGS[@]}" #  --pl