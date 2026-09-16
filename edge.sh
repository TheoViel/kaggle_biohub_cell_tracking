#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

set -e

cd src

# ---------------------------------------------

echo
echo "Training"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge.py --nodes_folder ../logs/ens_1/preds_ens_r6_0.25/

echo
echo "Running Inference"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge_predict.py #  --log_folder ../logs/edge_run/2026-09-07/4/

echo
echo "Postprocessing"
echo

python main_edge_postprocess.py # --preds_dir ../logs/edge_run/2026-09-16/2/preds_0.45/

# python main_edge_postprocess.py --ilp --ilp_disappearance 0.75 --ilp_threshold 0.4


# ---------------------------------------------

# # PP Tuning
# cd src/postprocessing
# python tune_edge_pp.py --preds_dir ../logs/edge_run/2026-09-14/5/preds_0.45
