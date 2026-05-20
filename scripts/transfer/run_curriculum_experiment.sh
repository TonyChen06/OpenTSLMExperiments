#!/bin/bash
# Sequential curriculum transfer experiment (2 runs).
#
# Control:   llama -> TSQA -> HAR -> Sleep
# Treatment: llama -> AHRI -> TSQA -> HAR -> Sleep
#
# Run both in parallel across two GPUs. Comparison reads the per-stage
# test accuracies and prints a control-vs-treatment table.
#
# Env vars (defaults in parens):
#   LLM            (meta-llama/Llama-3.2-1B-Instruct)
#   AHRI_DATA      (data/ahri)
#   OUT_ROOT       (results/transfer)
#   EPOCHS         (3)
#   BATCH          (2)
#   CONTROL_GPU    (6)
#   TREATMENT_GPU  (7)
#
# Example:
#   CONTROL_GPU=6 TREATMENT_GPU=7 bash scripts/transfer/run_curriculum_experiment.sh

set -euo pipefail

LLM="${LLM:-meta-llama/Llama-3.2-1B-Instruct}"
AHRI_DATA="${AHRI_DATA:-data/ahri}"
OUT_ROOT="${OUT_ROOT:-results/transfer}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-2}"
CONTROL_GPU="${CONTROL_GPU:-6}"
TREATMENT_GPU="${TREATMENT_GPU:-7}"

mkdir -p "$OUT_ROOT" logs/transfer

echo "[$(date)] starting transfer experiment"
echo "  llm=$LLM epochs=$EPOCHS batch=$BATCH"
echo "  control on GPU $CONTROL_GPU  (TSQA -> HAR -> Sleep)"
echo "  treatment on GPU $TREATMENT_GPU  (AHRI -> TSQA -> HAR -> Sleep)"

# ---- Control: TSQA -> HAR -> Sleep ----
(
    CUDA_VISIBLE_DEVICES="$CONTROL_GPU" PYTHONPATH=src python scripts/transfer/train_curriculum.py \
        --stages tsqa har_cot sleep \
        --llm "$LLM" --epochs "$EPOCHS" --batch_size "$BATCH" \
        --out "$OUT_ROOT/control" \
        > logs/transfer/control.log 2>&1
    echo "[control] done"
) &
CONTROL_PID=$!

# ---- Treatment: AHRI -> TSQA -> HAR -> Sleep ----
(
    CUDA_VISIBLE_DEVICES="$TREATMENT_GPU" PYTHONPATH=src python scripts/transfer/train_curriculum.py \
        --stages ahri tsqa har_cot sleep \
        --ahri_data "$AHRI_DATA" \
        --llm "$LLM" --epochs "$EPOCHS" --batch_size "$BATCH" \
        --out "$OUT_ROOT/treatment" \
        > logs/transfer/treatment.log 2>&1
    echo "[treatment] done"
) &
TREATMENT_PID=$!

echo "[run] control pid=$CONTROL_PID treatment pid=$TREATMENT_PID"
echo "[run] tail -f logs/transfer/{control,treatment}.log to watch"

wait $CONTROL_PID
wait $TREATMENT_PID

echo
echo "========== compare =========="
PYTHONPATH=src python scripts/transfer/compare_curriculum.py "$OUT_ROOT"
