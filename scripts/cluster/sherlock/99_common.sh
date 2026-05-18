#!/bin/bash
# Common preamble sourced by every Sherlock Ahri SLURM job.
# Loads modules, activates the venv, configures HF cache paths, exports
# RUNTIME env vars.

set -euo pipefail

# Modules — same versions as setup
module load python/3.12.1 || true
module load cuda/12.4.0 || module load cuda/12.2.0 || true

# Resolve scratch paths
: "${SCRATCH:=$HOME/scratch}"
: "${AHRI_SCRATCH:=$SCRATCH/ahri}"

if [[ ! -f "$AHRI_SCRATCH/env.sh" ]]; then
    echo "[99_common] missing $AHRI_SCRATCH/env.sh — run 00_setup_env.sh first" >&2
    exit 1
fi
source "$AHRI_SCRATCH/env.sh"
source "$VENV_DIR/bin/activate"

cd "$REPO_ROOT"

# Compute nodes lack internet — force offline mode
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
# Avoid tokenizer-fork warnings flooding logs
export TOKENIZERS_PARALLELISM=false

# Logging
echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_PROCID:-?}/${SLURM_NTASKS:-?} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
