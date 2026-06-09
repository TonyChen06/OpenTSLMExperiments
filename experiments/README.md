# experiments/ — MambaTSLM representation & training A/B harnesses

This directory is **experiment tooling, not part of the OpenTSLM core.** The core feature lives in
`src/opentslm/model/llm/MambaTSLM.py` and its curriculum integration; everything here is for
*iterating on* and *ablating* that model outside the full curriculum.

Each harness builds a `MambaTSLM` (or `OpenTSLM-SP`) directly via the shared `build_model` factory
in `run_har.py` and runs a single task end-to-end (train + eval) with a fixed, fair budget, writing a
YAML run-record per run. They're deliberately small and standalone.

## Harnesses

| script | task | metric | notes |
|---|---|---|---|
| `run_tsqa.py` | TSQA (trend/volatility/seasonality/outliers MCQ) | letter accuracy | **the sensitive testbed** — low-noise, discriminates representation choices |
| `run_har.py` | HAR-CoT (single stage) | macro/micro F1 | noisier; also hosts the shared `build_model` factory |
| `run_pretrain.py` | KernelSynth forecasting pretrain | next-bin CE | exercises the **signal-output** path (`forecast` / value-bin generation), multi-GPU via `ddp.sh` |

```bash
# single-GPU TSQA, quant value-bin Mamba, LoRA r16
CUDA_VISIBLE_DEVICES=0 python experiments/run_tsqa.py \
    --model mamba --llm-id state-spaces/mamba-370m-hf --lora-r 16 \
    --train-n 8000 --epochs 3 --bs 8 --test-n 2000

# representation ablation knobs (within the quant family)
#   --scaling {zscore,meanabs}  --bin-mode {uniform,quantile}  --init-mode {numeracy,random}
#   --n-bins N  --vrange V  --seed S   (paired seeds: test fixed @ seed 0, train varies per seed)

# multi-GPU forecasting pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/ddp.sh 4 experiments/run_pretrain.py --steps 6000
```

## What the study concluded (why MambaTSLM is quant-only)

The rep study swept signal representations and quant value-bin tokens were chosen as the locked
default — so the core model dropped the alternatives (vector encoders, numeric-text "floats",
bidirectional feed). The ablation knobs that remain (`scaling` / `bin-mode` / `init-mode`) stay
configurable here because they're sub-choices *within* the quant representation.

Findings (TSQA-370m, paired seeds, test-n ≥ 2000):
- **quant value-bin tokens** chosen over vector/float encoders — enables pure-token signal *output*
  (forecasting) and a head-free story; wins on value-precision tasks where averaging encoders blur values.
- **uniform bins > quantile** (~+2pt): equal-width spacing beats equal-mass; quantile over-concentrates
  near 0 and collapses the tails.
- **numeracy-grounded init > random** (~+4pt): seeding each bin embedding from the LLM's embedding of
  its center-value text is a real sample-efficiency win.
- **scaling (zscore vs meanabs) and bin-count are ~representation-agnostic at scale** — quant CE-on-bins
  is structurally normalization-agnostic; meanabs never wins, zscore is the safe default.
- **recipe:** LoRA `r=16` + trainable bin-embeddings (the new value-bin tokens must stay trainable so
  they can learn) at lr ≈ 1e-4/2e-4.

The headline curriculum head-to-head vs published OpenTSLM lives in `main_results/`.
