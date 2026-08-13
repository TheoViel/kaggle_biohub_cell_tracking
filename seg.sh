cd src

torchrun --standalone --nproc_per_node=2 main_seg.py
