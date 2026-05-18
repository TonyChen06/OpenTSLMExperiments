#!/bin/bash
# Sherlock environment setup (run once on a login node).
#
# Stanford Sherlock specifics (https://www.sherlock.stanford.edu/docs/):
#   - SLURM scheduler.
#   - Modules system; Python via `module load python/3.12.1`.
#   - $HOME is tiny (15 GB). Heavy stuff goes in $SCRATCH (1 TB, purged after
#     90 days of inactivity) or $GROUP_SCRATCH (shared).
#   - Compute nodes typically have NO internet — fetch all weights+datasets
#     from a login node first.
#   - GPU partition: `--partition=gpu`, common types via `-C` constraints
#     (e.g. `-C GPU_GEN:VLT` for newer GPUs). Owners partition is preempt-able.
#
# This script sets up a uv-managed venv under $SCRATCH and points HF_HOME at
# scratch so model caches survive across jobs.

set -euo pipefail

# Pretty section headers so the operator can see which step is running.
section() { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
echo "[setup] repo root: $REPO_ROOT"

# 1. Modules
section "[1/6] Load Sherlock modules"
module load python/3.12.1
module load cuda/12.4.0 || module load cuda/12.2.0 || true   # CUDA toolkit (optional; PyTorch ships its own)
module list 2>&1 | head -10

section "[2/6] Set up scratch paths"
# 2. Choose a scratch root for venv + caches
: "${SCRATCH:=$HOME/scratch}"
mkdir -p "$SCRATCH/ahri"
export AHRI_SCRATCH="$SCRATCH/ahri"
echo "  AHRI_SCRATCH=$AHRI_SCRATCH"

export VENV_DIR="$AHRI_SCRATCH/venv"
export HF_HOME="$AHRI_SCRATCH/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export AHRI_DATA_ROOT="$AHRI_SCRATCH/data"
mkdir -p "$HF_HUB_CACHE" "$AHRI_DATA_ROOT"

section "[3/6] Install uv (if missing)"
# 3. Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
    echo "  installing uv to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv: $(uv --version)"

section "[4/6] Create venv and install Ahri (editable)"
# 4. Create venv and install
cd "$REPO_ROOT"
if [[ ! -d "$VENV_DIR" ]]; then
    uv venv "$VENV_DIR" --python 3.12
fi
source "$VENV_DIR/bin/activate"
uv pip install -e ".[notebook]" -q
uv pip install pytest -q
echo "  python: $(python --version) at $(which python)"

section "[5/6] Persist env vars + pre-fetch Pythia weights"
# 5. Persist env vars so child SLURM jobs see them
cat > "$AHRI_SCRATCH/env.sh" <<EOF
# Source this from any Ahri SLURM job (sourced by 99_common.sh).
export AHRI_SCRATCH="$AHRI_SCRATCH"
export VENV_DIR="$VENV_DIR"
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export AHRI_DATA_ROOT="$AHRI_DATA_ROOT"
export REPO_ROOT="$REPO_ROOT"
EOF
echo "  wrote $AHRI_SCRATCH/env.sh"

# 6. Pre-fetch Pythia weights (needs internet — login node)
echo "  pre-fetching Pythia weights to $HF_HOME"
PYTHONPATH="$REPO_ROOT/src" python "$REPO_ROOT/scripts/cluster/common/prefetch_models.py"

section "[6/6] Environment self-check"
PYTHONPATH="$REPO_ROOT/src" python "$REPO_ROOT/scripts/cluster/common/check_env.py" \
    --hf-cache "$HF_HOME" \
    --data-dir "$AHRI_DATA_ROOT" \
    --models EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1.4b

echo ""
echo "[setup] DONE."
echo "[setup] Source env in future shells: source $AHRI_SCRATCH/env.sh"
echo "[setup] Activate venv:                source $VENV_DIR/bin/activate"
