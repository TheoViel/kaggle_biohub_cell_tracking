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

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge.py --nodes_folder ../ensemble/v16_2026-09-26_1_eduardo_current_mean3_append_b0.45_a0.35_d7

echo
echo "Running Inference"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge_predict.py #  --log_folder ../logs/edge_run/2026-09-07/4/

echo
echo "Postprocessing"
echo

python main_edge_postprocess.py --ilp  # --preds_dir ../logs/edge_run/2026-09-24/4/preds_0.45/


# ---------------------------------------------

echo
echo "Training"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge.py --nodes_folder ../ensemble/v17_2026-09-26_1_eduardo_current_mean3_append_b0.45_a0.45_d7

echo
echo "Running Inference"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge_predict.py #  --log_folder ../logs/edge_run/2026-09-07/4/

echo
echo "Postprocessing"
echo

python main_edge_postprocess.py --ilp  # --preds_dir ../logs/edge_run/2026-09-24/4/preds_0.45/

# ---------------------------------------------

echo
echo "Training"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge.py --nodes_folder ../ensemble/v18_2026-09-25_4_eduardo_current_mean3_append_b0.60_a0.45_d7

echo
echo "Running Inference"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge_predict.py #  --log_folder ../logs/edge_run/2026-09-07/4/

echo
echo "Postprocessing"
echo

python main_edge_postprocess.py --ilp  # --preds_dir ../logs/edge_run/2026-09-24/4/preds_0.45/

# ---------------------------------------------

echo
echo "Training"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge.py --nodes_folder ../ensemble/v19_2026-09-25_4_eduardo_current_mean3_append_b0.65_a0.45_d7

echo
echo "Running Inference"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge_predict.py #  --log_folder ../logs/edge_run/2026-09-07/4/

echo
echo "Postprocessing"
echo

python main_edge_postprocess.py --ilp  # --preds_dir ../logs/edge_run/2026-09-24/4/preds_0.45/

# ---------------------------------------------

# # PP Tuning
# cd src/postprocessing
# python tune_edge_pp.py --preds_dir ../logs/edge_run/2026-09-27/0/preds_0.45 --fold all
