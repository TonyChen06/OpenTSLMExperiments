#!/bin/bash
# Train + evaluate on every Ahri task with a single LLM.
# Sequential — best for single-GPU machines. Use the SLURM grid script for
# multi-GPU parallel submission.
#
# Args (env vars):
#   LLM          HF model id (default: EleutherAI/pythia-160m)
#   DATA         dataset root (local path or HF repo id; default: HF Hub)
#   OUT          output dir (default: results/ahri/<llm-short-name>)
#   EPOCHS       (default: 20)
#   BATCH        (default: 32)
#   TASKS        space-separated; default = all 21
#   GPU          single GPU index to pin to (default: 0)
#
# Example:
#   LLM=EleutherAI/pythia-160m GPU=6 bash scripts/ahri/run_all_tasks.sh
#   TASKS="1.1 1.2 1.3" LLM=EleutherAI/pythia-160m bash scripts/ahri/run_all_tasks.sh

set -euo pipefail

LLM="${LLM:-EleutherAI/pythia-160m}"
DATA="${DATA:-TonyChen06/AscendingHarmonicReasoningInstruction}"
TAG="$(basename "$LLM")"
OUT="${OUT:-results/ahri/${TAG}}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-32}"
GPU="${GPU:-0}"
TASKS="${TASKS:-1.1 1.2 1.3 1.4 2.1 2.2 2.3 2.4 3.1 3.2 3.3 3.4 4.1 4.2 4.3 4.4 4.5 5.1 5.2 5.3 5.4}"

mkdir -p "$OUT" logs
echo "[run_all_tasks] llm=$LLM data=$DATA out=$OUT epochs=$EPOCHS batch=$BATCH gpu=$GPU"
echo "[run_all_tasks] tasks: $TASKS"

START=$(date +%s)
for task in $TASKS; do
    TASK_OUT="$OUT/$task"
    LOG="logs/${TAG}-${task}.log"
    if [[ -f "$TASK_OUT/summary.json" ]]; then
        echo "[skip] task $task already has summary.json"
        continue
    fi
    echo
    echo "========== task $task ==========  (log: $LOG)"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=src python scripts/ahri/train_single_task.py \
        --task "$task" \
        --llm "$LLM" \
        --data "$DATA" \
        --out "$TASK_OUT" \
        --epochs "$EPOCHS" --batch_size "$BATCH" --patience 5 \
        2>&1 | tee "$LOG" | grep -E "^\[(setup|epoch|test|done)" || true
done
END=$(date +%s)

echo
echo "========== aggregate =========="
PYTHONPATH=src python scripts/ahri/aggregate_results.py "$OUT" --out "$OUT/table.md"
echo
echo "[run_all_tasks] total wall time: $(( (END - START) / 60 )) min"
echo "[run_all_tasks] results table:  $OUT/table.md"
