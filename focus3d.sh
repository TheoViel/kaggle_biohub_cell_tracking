#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
# torch reads MKL_NUM_THREADS before OMP_NUM_THREADS, so OMP alone leaves
# it at 80 intra-op threads. One rank per GPU plus their dataloader workers
# then oversubscribe the box by ~30x and the small per-window tensor ops
# slow down ~4x. Both are set, so whichever torch reads says 1.
export MKL_NUM_THREADS=1

set -e

n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

cd src

# --local-addr: this host's FQDN does not resolve, so torchrun's default
# lookup hangs when it hands the address to the workers.
torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" pl_focus_3D.py --skip-existing "$@"
