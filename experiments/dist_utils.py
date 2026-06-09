"""Minimal data-parallel helpers (manual DDP) for our custom training loops.
Launch with: torchrun --nproc_per_node=N <script.py> ...  (no code flag needed; auto-detects LOCAL_RANK).
Each rank runs the same loop on its own GPU + its own data shard; we all-reduce(AVG) the gradients
each step so all ranks apply the identical update (= effective batch = N x per-rank batch).
Manual (not the DDP wrapper) so MambaTSLM's .get_input_embeddings()/.generate()/.config still work."""
import os
import torch
import torch.distributed as dist


def dist_init():
    """-> (rank, world_size, device). Non-distributed (single GPU) if not launched via torchrun."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, "cuda"
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dist.init_process_group(backend=os.environ.get("TSLM_BACKEND", "nccl"))   # nccl segfaults on this box; gloo fallback
    return dist.get_rank(), dist.get_world_size(), f"cuda:{lr}"


def broadcast_params(params):
    """Make every rank start from rank-0's weights (LoRA init is random per process)."""
    if dist.is_initialized():
        for p in params:
            dist.broadcast(p.data, src=0)


def sync_grads(params):
    """Average gradients across ranks after backward(), before optimizer.step()."""
    if dist.is_initialized():
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def cleanup():
    if dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()
