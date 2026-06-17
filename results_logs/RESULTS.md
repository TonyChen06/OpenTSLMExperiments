# Box A results — synced train logs + metrics (mamba-1.4b, SP)

Per-epoch logs: `results_logs/boxA_*.txt`. train/val = at best-val epoch. acc=accuracy, F1=macro-F1 (matches OpenTSLM paper, incl. TSQA).

| model | task | best-ep | train loss | val loss | acc | F1 |
|---|---|---|---|---|---|---|
| mamba-1.4b | TSQA | 40 | 0.0007 | 0.0018 | 0.9981 | 0.9981 |
| mamba-1.4b | HAR | 18 | 0.3898 | 0.5238 | 0.7192 | 0.6709 |
| mamba-1.4b | Sleep | 10 | 0.3959 | 0.5519 | 0.8712 | 0.6081 |
| SP | TSQA | 14 | 0.0489 | 0.0406 | 0.9240 | 0.9249 |
| SP | HAR | 27 | 0.2223 | 0.3293 | garbled | garbled |
| SP | Sleep | 13 | 0.2992 | 0.4501 | garbled | garbled |

SP HAR/Sleep F1='garbled' until the fixed-loader re-eval finishes (_orig_mod bug, fix a9b2eb0); SP val/train losses are real.
Box B: append Flamingo / llama_bins / mamba-370m rows + `results_logs/boxB_*.txt`; re-eval llama_bins (compiled → same bug).
