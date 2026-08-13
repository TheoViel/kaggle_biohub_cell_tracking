#!/usr/bin/env bash
set -e
cd src
torchrun --standalone --nproc_per_node=2 main_track.py
