#!/bin/bash
#SBATCH --job-name=ahri-interference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/interference-%j.out
#SBATCH --error=logs/interference-%j.err

# Build the task-interference matrix (RQ2 Experiment 6.3).
# This is quadratic in tasks — even for 8 tasks at 10k steps/pair it's heavy.

source "$(dirname "${BASH_SOURCE[0]:-$0}")/99_common.sh"
mkdir -p logs

: "${LLM:=EleutherAI/pythia-410m}"
: "${STEPS:=10000}"

PYTHONPATH=src python scripts/ahri/run_interference_matrix.py \
    --tasks 1.2 2.1 2.2 3.1 3.4 4.1 4.3 5.1 \
    --llm "$LLM" --steps "$STEPS" \
    --data "$AHRI_DATA_ROOT" \
    --out "$AHRI_SCRATCH/runs/rq2/interference"
