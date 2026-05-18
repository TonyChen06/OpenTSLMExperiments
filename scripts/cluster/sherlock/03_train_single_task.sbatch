#!/bin/bash
#SBATCH --job-name=ahri-rq1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rq1-%x-%j.out
#SBATCH --error=logs/rq1-%x-%j.err

# Single-task training for RQ1 (one task, one model scale).
#
# Args via env: TASK_ID, LLM, OUT_TAG (optional).
#   TASK_ID=1.2 LLM=EleutherAI/pythia-410m sbatch scripts/cluster/sherlock/03_train_single_task.sbatch
#
# To submit the full 21x3 grid, use 04_grid_submit.sh which loops sbatch.

source "$(dirname "${BASH_SOURCE[0]:-$0}")/99_common.sh"
mkdir -p logs

: "${TASK_ID:?must set TASK_ID env var (e.g. 1.2)}"
: "${LLM:?must set LLM env var (e.g. EleutherAI/pythia-410m)}"
TAG="${OUT_TAG:-$(basename $LLM)}"
OUT="$AHRI_SCRATCH/runs/rq1/${TASK_ID}-${TAG}"

PYTHONPATH=src python scripts/ahri/train_single_task.py \
    --task "$TASK_ID" \
    --llm "$LLM" \
    --data "$AHRI_DATA_ROOT" \
    --out "$OUT" \
    --epochs 20 --batch_size 32 --patience 10
