# Box A results — synced train logs + metrics (mamba-1.4b, SP)

Per-epoch logs: `results_logs/boxA_*.txt`. train/val = at best-val epoch.
**acc + F1 are computed the codebase's way** — the OpenTSLM `evaluation/opentslm/` parsers
(task-specific label extraction + canonicalization, then macro-F1), reproduced by
`tslm_study/codebase_f1.py <tsqa|har|sleep> <test_predictions.jsonl>`. Paper-consistent (incl.
macro-F1 for TSQA). NOTE vs the old naive "Answer:" scorer: codebase **Sleep** acc/F1 differ
materially — its canonicalization merges sleep-stage-4→3 and discovers all 5 stages, whereas the
naive scorer collapsed two stages and inflated acc (0.871→0.794). HAR/TSQA F1 unchanged.

| model | task | best-ep | train loss | val loss | acc | F1 |
|---|---|---|---|---|---|---|
| mamba-1.4b | TSQA | 40 | 0.0007 | 0.0018 | 0.9981 | 0.9981 |
| mamba-1.4b | HAR | 18 | 0.3898 | 0.5238 | 0.7132 | 0.6709 |
| mamba-1.4b | Sleep | 10 | 0.3959 | 0.5519 | 0.7940 | 0.6265 |
| SP | TSQA | 14 | 0.0489 | 0.0406 | 0.9240 | 0.9249 |
| SP | HAR | 27 | 0.2223 | 0.3293 | 0.6982 | 0.6500 |
| SP | Sleep | 13 | 0.2992 | 0.4501 | 0.7328 | 0.5844 |

SP HAR/Sleep are now FINAL (gen-fixed re-eval, 100% of CoTs reach "Answer:", full 8224/932 test sets).
Getting here took TWO bug fixes on the SP eval path:
  1. `_orig_mod` (a9b2eb0): torch.compile prefixed the saved LoRA keys, so reload silently dropped the
     adapter → eval ran on the unadapted base LM (random garbage).
  2. `position_ids` (51650b7): generate() passed explicit position_ids (from the bs>1 left-pad forward
     fix), which broke RoPE advancement for generated tokens — harmless for short MCQ (TSQA 0.924) but
     catastrophic for long CoT (looped to the 2000-tok cap, never emitting an answer → F1≈0). Dropping
     it gives clean terminating CoT and made eval ~12× faster. Mamba/Flamingo pass no position_ids.
mamba-1.4b > SP on both CoT tasks (HAR 0.671 vs 0.650, Sleep 0.627 vs 0.584); ~tie on TSQA acc.
Box B: append Flamingo / llama_bins / mamba-370m rows + `results_logs/boxB_*.txt`; re-eval llama_bins
(compiled → same _orig_mod bug). Compute all F1s with `tslm_study/codebase_f1.py` for parity.
