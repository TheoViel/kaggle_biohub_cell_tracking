"""
Distributed helpers for single-process and torchrun/DDP training.

A process group is only created when torchrun sets WORLD_SIZE > 1, so the same
code path runs unchanged in notebooks and in single-GPU scripts.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def init_distributed(backend=None):
    """Read the torchrun environment, pick a device, and join the process group."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        if backend is None:
            backend = (
                "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
            )
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return DistContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
        device=device,
    )


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model):
    return getattr(model, "module", model)


def all_reduce_mean(values, device):
    """Average a name -> float mapping over every rank."""
    if not (dist.is_available() and dist.is_initialized()):
        return dict(values)

    keys = sorted(values)
    tensor = torch.tensor(
        [values[key] for key in keys], dtype=torch.float64, device=device
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return dict(zip(keys, tensor.tolist()))


def seed_everything(seed, rank=0):
    """Seed with a rank offset so ranks draw different augmentations."""
    seed = (int(seed) + int(rank)) % (2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """Give every dataloader worker its own numpy seed.

    Workers inherit the parent numpy state on fork, so without this every worker
    of a rank would emit identical crops and flips.
    """
    np.random.seed(torch.initial_seed() % (2**32))
