#!/bin/bash
#SBATCH --job-name=ahri-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

# Smoke test on a real GPU node: runs the full pytest suite then a 2-step
# train. If this passes, larger jobs will work too.
#
# Submit from the repo root:
#   sbatch scripts/cluster/sherlock/01_smoke_test.sbatch

source "$(dirname "${BASH_SOURCE[0]:-$0}")/99_common.sh"

mkdir -p logs

# 1. Environment check
PYTHONPATH=src python scripts/cluster/common/check_env.py \
    --gpu --hf-cache "$HF_HOME" --data-dir "$AHRI_DATA_ROOT" \
    --models EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m \
    --check-distributed

# 2. Full pytest
PYTHONPATH=src python -m pytest tests/ahri/ -q --tb=short

# 3. Real 2-step train on Pythia-160m + tiny generated data
SMOKE_DATA="$AHRI_SCRATCH/smoke_data"
SMOKE_OUT="$AHRI_SCRATCH/smoke_run"
PYTHONPATH=src python -c "
from pathlib import Path
from opentslm.ahri.generate import SPLIT_SIZES, generate_task
SPLIT_SIZES.update({'train': 64, 'val': 32, 'test': 64})
generate_task('1.2', Path('$SMOKE_DATA'), overwrite=True)
"

PYTHONPATH=src python scripts/ahri/train_single_task.py \
    --task 1.2 \
    --llm EleutherAI/pythia-160m \
    --data "$SMOKE_DATA" \
    --out "$SMOKE_OUT" \
    --epochs 1 --batch_size 8 --patience 1 --max_new_tokens 6

echo "[smoke] PASS — environment ready for full runs."
