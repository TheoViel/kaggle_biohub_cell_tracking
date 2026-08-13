#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1

set -e

cd src

torchrun --standalone --nproc_per_node=8 main_track.py
