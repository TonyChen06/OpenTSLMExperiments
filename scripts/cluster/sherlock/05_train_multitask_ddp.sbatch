#!/bin/bash
#SBATCH --job-name=ahri-rq2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rq2-%x-%j.out
#SBATCH --error=logs/rq2-%x-%j.err

# Multi-task RQ2 training with 4-GPU DDP on a single node.
#
# Args: SCHEDULE (simultaneous | easy_to_hard | hard_to_easy | random_introduction)
#       LLM (default Pythia-410m)
#       STEPS (default 50000)
#       OUT_TAG (default = schedule name)
#
# Example:
#   SCHEDULE=easy_to_hard LLM=EleutherAI/pythia-410m \
#     sbatch scripts/cluster/sherlock/05_train_multitask_ddp.sbatch

source "$(dirname "${BASH_SOURCE[0]:-$0}")/99_common.sh"
mkdir -p logs

: "${SCHEDULE:=simultaneous}"
: "${LLM:=EleutherAI/pythia-410m}"
: "${STEPS:=50000}"
TAG="${OUT_TAG:-$SCHEDULE}"
OUT="$AHRI_SCRATCH/runs/rq2/${TAG}-$(basename $LLM)"

# torchrun rendezvous on the local node; one process per GPU
NGPU=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1 || echo 1)
PYTHONPATH=src torchrun \
    --standalone --nproc_per_node="$NGPU" \
    scripts/ahri/train_multitask.py \
    --tasks all --schedule "$SCHEDULE" \
    --llm "$LLM" \
    --data "$AHRI_DATA_ROOT" \
    --out "$OUT" \
    --steps "$STEPS" --batch_size 8 --eval_every 500
