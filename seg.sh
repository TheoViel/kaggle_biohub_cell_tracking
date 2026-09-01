cd src

# --local-addr: this host's FQDN does not resolve, so torchrun's default
# lookup hangs when it hands the address to the workers.
torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=8 main_seg.py
