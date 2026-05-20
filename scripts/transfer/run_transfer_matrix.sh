#!/bin/bash
# AHRI-as-pretraining transfer experiment.
#
# Runs the full 7-job matrix:
#   1) pretrain OpenTSLM-SP on all 21 Ahri tasks (warm-up encoder+projector+LoRA)
#   2) downstream control: TSQA from scratch
#   3) downstream control: HAR-CoT from scratch
#   4) downstream control: Sleep-CoT from scratch
#   5) downstream treatment: TSQA, warm-started from (1)
#   6) downstream treatment: HAR-CoT, warm-started from (1)
#   7) downstream treatment: Sleep-CoT, warm-started from (1)
#
# Jobs are sequential by default. If you have N GPUs available, set
# PARALLEL_GPUS="6 7 ..." and the script will round-robin downstream jobs
# across them (pretrain is always single-GPU).
#
# Env vars:
#   LLM            (default: HuggingFaceTB/SmolLM2-360M)
#   AHRI_DATA      (default: data/ahri  -- or use the HF repo id)
#   OUT_ROOT       (default: results/transfer)
#   PRETRAIN_STEPS (default: 5000)
#   DOWNSTREAM_EPOCHS (default: 5)
#   BATCH          (default: 4)
#   PRETRAIN_GPU   (default: 6)         single GPU for the pretrain
#   PARALLEL_GPUS  (default: "6 7")     GPUs to round-robin downstream
#
# Example:
#   PRETRAIN_STEPS=10000 bash scripts/transfer/run_transfer_matrix.sh

set -euo pipefail

LLM="${LLM:-HuggingFaceTB/SmolLM2-360M}"
AHRI_DATA="${AHRI_DATA:-data/ahri}"
OUT_ROOT="${OUT_ROOT:-results/transfer}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-5000}"
DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-5}"
BATCH="${BATCH:-4}"
PRETRAIN_GPU="${PRETRAIN_GPU:-6}"
PARALLEL_GPUS="${PARALLEL_GPUS:-6 7}"

mkdir -p "$OUT_ROOT" logs/transfer
PRETRAIN_DIR="$OUT_ROOT/pretrain-ahri"
PRETRAIN_CKPT="$PRETRAIN_DIR/best.pt"

# ---------------------------------------------------------------------
# 1. Pretrain on Ahri (always single GPU; depends on nothing else)
# ---------------------------------------------------------------------
if [[ -f "$PRETRAIN_CKPT" ]]; then
    echo "[1/7] pretrain checkpoint already exists at $PRETRAIN_CKPT, skipping pretrain"
else
    echo "[1/7] pretrain on Ahri (all 21 tasks) -> $PRETRAIN_DIR"
    CUDA_VISIBLE_DEVICES="$PRETRAIN_GPU" PYTHONPATH=src python scripts/transfer/train_opentslm_sp.py \
        --dataset ahri --ahri_data "$AHRI_DATA" \
        --llm "$LLM" \
        --out "$PRETRAIN_DIR" \
        --steps "$PRETRAIN_STEPS" --batch_size "$BATCH" \
        2>&1 | tee logs/transfer/pretrain-ahri.log
fi

# ---------------------------------------------------------------------
# 2-7. Six downstream runs, parallelised across PARALLEL_GPUS.
# ---------------------------------------------------------------------
DOWNSTREAM_TASKS=(tsqa har_cot sleep)
CONDITIONS=(control treatment)

read -r -a GPU_ARRAY <<< "$PARALLEL_GPUS"
NGPU=${#GPU_ARRAY[@]}

job_index=0
pids=()
for task in "${DOWNSTREAM_TASKS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        gpu="${GPU_ARRAY[$(( job_index % NGPU ))]}"
        out="$OUT_ROOT/${task}-${cond}"
        log="logs/transfer/${task}-${cond}.log"
        if [[ -f "$out/summary.json" ]]; then
            echo "[skip] $task-$cond already has summary.json"
            job_index=$(( job_index + 1 ))
            continue
        fi

        warm=""
        if [[ "$cond" == "treatment" ]]; then
            warm="--load_checkpoint $PRETRAIN_CKPT"
        fi

        echo "[$((job_index + 2))/7] downstream $task / $cond  on GPU $gpu  log=$log"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/transfer/train_opentslm_sp.py \
            --dataset "$task" --llm "$LLM" $warm \
            --out "$out" \
            --epochs "$DOWNSTREAM_EPOCHS" --batch_size "$BATCH" \
            > "$log" 2>&1 &
        pids+=($!)
        job_index=$(( job_index + 1 ))

        # if we've filled all GPUs, wait for one to finish before launching next
        if (( ${#pids[@]} >= NGPU )); then
            wait -n   # wait for any one job to complete
            # remove completed pids
            new_pids=()
            for p in "${pids[@]}"; do
                if kill -0 "$p" 2>/dev/null; then
                    new_pids+=("$p")
                fi
            done
            pids=("${new_pids[@]}")
        fi
    done
done

# wait for any stragglers
for p in "${pids[@]}"; do
    wait "$p"
done

# ---------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------
echo
echo "========== compare =========="
PYTHONPATH=src python scripts/transfer/compare_transfer.py "$OUT_ROOT"
echo "[done] see $OUT_ROOT/compare.md"
