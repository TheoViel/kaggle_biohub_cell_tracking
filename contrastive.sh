#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

set -e

cd src

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_contrastive.py

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py



# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-09/1/ --coords_log_folder ...

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-26/0/ --coords_log_folder ../ensemble/v14_2026-09-26_1_2026-09-24_5_append_b0.45_a0.40_d7
torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-26/0/ --coords_log_folder ../ensemble/v15_2026-09-26_1_2026-09-24_5_append_b0.45_a0.45_d7