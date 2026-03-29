# OpenTSLM Noise Injection & Signal Blocking

## >> Ready-to-run experiment commands: [`run_signal_blocking_experiments.txt`](run_signal_blocking_experiments.txt) and [`run_partial_noise_experiments.txt`](run_partial_noise_experiments.txt) <<

Interpretability testing tools for [OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM). This branch adds two signal perturbation methods to measure how much the model relies on actual signal content vs. text metadata.

## Perturbation Methods

### 1. Partial Noise Injection

Blends the original signal with noise at a configurable level:

```
output[t] = (1 - noise_level) * signal[t] + noise_level * noise[t]
```

- `noise_level=0.0` — pure signal (no perturbation)
- `noise_level=0.3` — 70% signal + 30% noise (slight perturbation)
- `noise_level=1.0` — pure noise (complete replacement)

Noise types: `gaussian`, `shuffle`, `zero`, `uniform`

**Usage (eval):**
```bash
PYTHONPATH=src:src/open_flamingo python evaluation/opentslm/ecg_qa_cot/evaluate_ecg_flamingo.py \
    --checkpoint hf_checkpoints/model_checkpoint.pt \
    --use_noise \
    --noise_type gaussian \
    --noise_level 0.3 \
    --noise_seed 67 \
    --output results/eval_ecg_noise30.json
```

**Usage (training):**
```bash
PYTHONPATH=src:src/open_flamingo python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --stages stage5_ecg_cot \
    --noise_type gaussian \
    --noise_level 0.3 \
    --noise_seed 67 \
    --experiment_name noise30_ecg
```

### 2. Signal Blocking

Replaces random windows of the signal with straight-line interpolation between the window's boundary values. The signal stays visually continuous, but diagnostic features within blocked regions are erased.

```
blocked_window = linspace(signal[start], signal[end], window_length)
```

Parameters:
- `--block_total_sec` — total seconds of signal to block out
- `--block_avg_sec` — average block duration (sampled from Normal distribution)
- `--block_std_sec` — std of block durations
- `--block_seed` — random seed for reproducibility

Block lengths are sampled from `Normal(avg, std)` and placed randomly without overlap.

**Usage:**
```bash
PYTHONPATH=src:src/open_flamingo python evaluation/opentslm/ecg_qa_cot/evaluate_ecg_flamingo.py \
    --checkpoint hf_checkpoints/model_checkpoint.pt \
    --use_block \
    --block_total_sec 3.0 \
    --block_avg_sec 0.5 \
    --block_std_sec 0.1 \
    --block_seed 67 \
    --output results/eval_ecg_block_3s.json
```


## Sample Rates

| Dataset | Sample Rate | Signal Duration | Example: 30% blocked |
|---------|------------|-----------------|----------------------|
| ECG     | 100 Hz     | 10s (1000 pts)  | `--block_total_sec 3.0` |
| Sleep   | 100 Hz     | 30s (3000 pts)  | `--block_total_sec 9.0` |
| HAR     | 50 Hz      | 2.56s (128 pts) | `--block_total_sec 0.75` |
| TSQA    | 1 Hz*      | ~100 pts        | `--block_total_sec 30` |

*TSQA has no real sample rate; each data point is treated as 1 "second".

## Experiment Scripts

- `run_partial_noise_experiments.txt` — 30% noise injection on all 4 stages
- `run_signal_blocking_experiments.txt` — signal blocking experiments with various configurations

## Eval Scripts

Located in `evaluation/opentslm/`:
- `tsqa/evaluate_tsqa_flamingo.py`
- `har_cot/evaluate_har_flamingo.py`
- `sleep/evaluate_sleep_flamingo.py`
- `ecg_qa_cot/evaluate_ecg_flamingo.py`

All support `--use_noise`, `--use_block`, `--max_samples`, and `--output` flags. First 20 samples' full predictions are saved to the output JSON.
