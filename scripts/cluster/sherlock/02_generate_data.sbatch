#!/bin/bash
#SBATCH --job-name=ahri-gen
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen-%j.out
#SBATCH --error=logs/gen-%j.err

# One-shot dataset generator. Runs on CPU; needs no internet.
# All 21 tasks * 12k examples = ~252k examples; takes ~30 min single-threaded.
#
# After this completes, optionally push to HF Hub from a login node:
#   PYTHONPATH=src python scripts/ahri/push_to_hub.py \
#       --local "$AHRI_DATA_ROOT" --repo your-org/ahri-v1
#
# Submit:
#   sbatch scripts/cluster/sherlock/02_generate_data.sbatch

source "$(dirname "${BASH_SOURCE[0]:-$0}")/99_common.sh"
mkdir -p logs

PYTHONPATH=src python -m opentslm.ahri.generate \
    --out "$AHRI_DATA_ROOT" \
    --seed 0 --overwrite

du -sh "$AHRI_DATA_ROOT"
echo "[gen] data written to $AHRI_DATA_ROOT"
