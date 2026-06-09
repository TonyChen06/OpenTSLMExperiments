#!/bin/bash
# Minimal data-parallel (torchrun) launcher for the multi-GPU experiment harnesses
# (e.g. run_pretrain.py, which uses dist_utils.broadcast_params / sync_grads).
#
# USAGE: CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/ddp.sh <nproc> <script.py> [args...]
#   e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/ddp.sh 4 experiments/run_pretrain.py --steps 6000
#
# Assumes the package is installed (e.g. `uv sync --extra mamba`). If your cluster needs NCCL
# tuning (interconnect/driver quirks), set the usual NCCL_* env vars before invoking.
set -e
N="$1"; shift
PORT=$((29500 + RANDOM % 2000))
python -m torch.distributed.run --nproc_per_node="$N" --master_port="$PORT" "$@"
