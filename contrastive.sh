#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

set -e

cd src

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_contrastive.py --backbone vit_base_patch16_224.dino

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py


# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-09/1/ --coords_log_folder ../logs/2026-09-14/7/preds_r6_0.40/

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-09/1/ --coords_log_folder ../logs/2026-09-14/7/preds_r7_0.40/


# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-09/1/ --coords_log_folder ../logs/2026-09-15/0/preds_r6_0.40/

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-09/1/ --coords_log_folder ../logs/ens_1/preds_ens_r6_0.25/

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" inference_contrastive.py --exp_folder ../logs/contrastive/2026-09-09/1/ --coords_log_folder ../logs/2026-09-15/0/preds_r6_0.30/
