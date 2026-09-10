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

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_track.py --mixup '{"p": 0.0, "alpha": 4.0}'

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_predict.py \
#     --det_threshold 0.4,0.5 \
#     --nms_radius_um 6,7 \

# python main_postprocess.py 


# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_track.py

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_predict.py \
    --det_threshold 0.4,0.5 \
    --nms_radius_um 6,7 \

python main_postprocess.py 
