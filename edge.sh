#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

set -e

cd src

echo
echo "Training Edge Transformer"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge.py

echo
echo "Running Inference"
echo

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_edge_predict.py #  --log_folder ../logs/edge_run/2026-09-07/4/

echo
echo "Postprocessing"
echo

python main_edge_postprocess.py  # --preds_dir ../logs/edge_run/2026-09-07/0/preds_0.30/

# Upper bound: the same model on a node set with recall 1.0, to read its linking
# with the detector's misses taken out. Embed the annotated nodes once into the
# nodes folder, then predict with them appended. Writes preds_gt_<threshold>/,
# so it leaves the real numbers above alone.
#
# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" \
#     inference_contrastive.py --gt \
#     --exp_folder ../logs/contrastive/2026-09-03/13/ \
#     --coords_log_folder ../logs/2026-09-07/4/preds_r6_0.50/
#
# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" \
#     main_edge_predict.py --gt --log_folder ../logs/edge_run/2026-09-07/2/
