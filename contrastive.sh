#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

set -e

cd src

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_contrastive.py

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py


# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-09/1/ --coords_log_folder ../logs/2026-09-24/5/preds_r6_0.45/

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-09/1/ --coords_log_folder ../ensemble/v2_2026-09-13_1_2026-09-24_5_consensus_r5_th0.40

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-09/1/ --coords_log_folder ../ensemble/v8_2026-09-13_1_2026-09-24_5_append_b0.50_a0.45_d7

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-09/1/ --coords_log_folder ../ensemble/v9_2026-09-24_5_2026-09-13_1_append_b0.50_a0.45_d7
