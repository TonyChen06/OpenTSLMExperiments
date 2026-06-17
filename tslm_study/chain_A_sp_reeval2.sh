#!/bin/bash
# Re-eval SP HAR + Sleep with the position_ids GENERATION fix (commit 51650b7).
# Prior re-eval loaded the LoRA (text on-topic) but generation collapsed because generate()
# passed explicit position_ids → RoPE broke on long CoT → never reached "Answer:". Now fixed,
# generation terminates at EOS (~200 tok) so this is ~10x faster than the 2000-tok-cap collapse.
# TSQA was already correct (short gen), so only HAR + Sleep need redo. Scored with codebase_f1.py.
cd "$(dirname "$0")/.."
source tslm_env.sh
log(){ echo "[sp-reeval2 $(date +%m-%d_%H:%M:%S)] $*" | tee -a logs/official/driver.log; }
D=results/Llama_3_2_1B/OpenTSLMSP
reeval(){  # stage tag
  rm -f "$D/$1/results/metrics.json"
  log "RE-EVAL SP $2 ($1) — position_ids gen fix"
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash tslm_study/ddp.sh 4 curriculum_learning.py \
    --model OpenTSLMSP --llm_id meta-llama/Llama-3.2-1B --device cuda --stages "$1" --eval_only \
    > "logs/official/reeval2_sp_$2.log" 2>&1
  log "DONE $2 (exit $?)"
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src "$PY" tslm_study/codebase_f1.py "$2" \
    "$D/$1/results/test_predictions.jsonl" 2>/dev/null | sed "s/^/[$2] /" | tee -a logs/official/driver.log
}
reeval stage3_cot       har
reeval stage4_sleep_cot sleep
log "=== SP RE-EVAL2 DONE (HAR+Sleep, gen-fixed) ==="
