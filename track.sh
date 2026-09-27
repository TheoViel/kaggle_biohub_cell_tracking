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

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_predict.py \
    --det_threshold 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7 \
    --nms_radius_um 6,7 \

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_track.py

torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_predict.py \
    --det_threshold 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7 \
    --nms_radius_um 6,7 \

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_track.py

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_predict.py \
#     --det_threshold 0.4,0.45,0.5,0.55,0.6,0.65,0.7 \
#     # --nms_radius_um 6,7 \

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_predict.py \
#     --det_threshold 0.4,0.45,0.5,0.55,0.6,0.65,0.7 \
#     --nms_radius_um 6,7 \
#     --pl \
#     --log_folder ../logs/2026-09-25/4/

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_predict.py \
#     --det_threshold 0.3,0.4,0.5,0.6 \
#     --nms_radius_um 6,7 \
#     --pl 

# python main_postprocess.py --pl  # --preds_dir ../logs/2026-09-23/4/preds_pl_r6_0.30/

# torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="$n_gpus" main_inference_ensemble.py \
#     --experiments ../logs/2026-09-11/1 ../logs/2026-09-15/0 \
#     --det_threshold 0.15,0.20,0.25,0.30,0.35,0.40 --nms_radius_um 5,6,7
