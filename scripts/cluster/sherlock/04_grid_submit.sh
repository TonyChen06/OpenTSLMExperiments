#!/bin/bash
# Submit the full 21-task x 3-scale grid for RQ1.
#
# Usage:
#   bash scripts/cluster/sherlock/04_grid_submit.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

TASKS=(1.1 1.2 1.3 1.4 2.1 2.2 2.3 2.4 3.1 3.2 3.3 3.4 4.1 4.2 4.3 4.4 4.5 5.1 5.2 5.3 5.4)
LLMS=("EleutherAI/pythia-160m" "EleutherAI/pythia-410m" "EleutherAI/pythia-1.4b")

for task in "${TASKS[@]}"; do
    for llm in "${LLMS[@]}"; do
        tag="$(basename "$llm")"
        echo "submit: task=$task llm=$llm"
        TASK_ID="$task" LLM="$llm" OUT_TAG="$tag" \
            sbatch "$REPO_ROOT/scripts/cluster/sherlock/03_train_single_task.sbatch"
    done
done

echo "Submitted $(( ${#TASKS[@]} * ${#LLMS[@]} )) jobs. Use 'squeue -u $USER' to monitor."
