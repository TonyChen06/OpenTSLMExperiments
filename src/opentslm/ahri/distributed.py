"""
Minimal distributed helpers. Supports plain single-GPU/CPU, single-node
torchrun (`torchrun --nproc_per_node=N`), and multi-node SLURM with
torchrun.

We don't depend on accelerate or composer — these are just thin wrappers
around torch.distributed so the trainers stay legible.

Env vars (set by torchrun or our SLURM launchers):
    RANK         global rank
    LOCAL_RANK   rank within the node
    WORLD_SIZE   total processes
    MASTER_ADDR, MASTER_PORT  rendezvous

If none of these are set, we run single-process.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_rank() -> bool:
    return rank() == 0


def setup_distributed(backend: str = "nccl") -> torch.device:
    """Initialise process group if needed; pin device. Returns the device
    to use on this rank."""
    if is_distributed() and not dist.is_initialized():
        # NCCL for GPU, gloo for CPU
        if backend == "nccl" and not torch.cuda.is_available():
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        return torch.device(f"cuda:{local_rank()}")
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Average a scalar across ranks. No-op if single-process."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size())


@contextmanager
def main_first():
    """Context manager so only rank 0 runs the protected block first; other
    ranks wait at a barrier, then run after."""
    if is_main_rank():
        yield
        barrier()
    else:
        barrier()
        yield
