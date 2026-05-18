# Ahri: Ascending Harmonic Reasoning Instruction

A controlled evaluation framework for Time-Series Language Models, organised
into 21 synthetic tasks across 5 tiers (paper: *Physics of Time-Series
Language Models: Part 1, The Capability Frontier*).

## Layout

| Module / dir                  | What it does                                              |
|-------------------------------|-----------------------------------------------------------|
| `waveforms.py`                | 6 primitive waveforms + 3 composition rules + held-out sampling helpers. |
| `prompts.py`                  | Standardized prompt template, signal placeholder tokens.  |
| `graders.py`                  | Classification / regression / multi-label parsers.        |
| `tasks/tier{1,2,3,4,5}_*.py`  | 21 `AhriTask` subclasses. Each defines `sample()` and `grade()`. |
| `tasks/__init__.py`           | Task registry: `get_task("3.1")`, `list_task_ids()`.      |
| `generate.py`                 | One-shot offline dataset generator (writes parquet).      |
| `dataset.py`                  | PyTorch `Dataset` that reads pre-generated parquet shards. |
| `multitask.py`                | Multi-task mixture + 4 curriculum schedulers for RQ2.     |
| `eval.py`                     | Greedy generation + grading + bootstrap CIs.              |

The model lives at `src/opentslm/model/llm/PhysicsTSLM.py` (encoder-free,
single linear projection per ELF; Pythia backbone).

## Workflow

1. **Generate datasets, once.** Each task gets 6k train / 2k val / 2k test. Train + val exclude the held-out parameter region; test samples uniformly from the full range. The per-example `held` flag marks test examples that fell in the held-out region, so eval can report accuracy on the in-distribution and held subsets separately.
   ```bash
   PYTHONPATH=src python -m opentslm.ahri.generate --out data/ahri
   ```
   The generator is deterministic in `(task_id, split, seed)`.

2. **Train a single task (RQ1).**
   ```bash
   PYTHONPATH=src python scripts/ahri/train_single_task.py \
       --task 1.2 --llm EleutherAI/pythia-410m --data data/ahri \
       --out results/ahri/1.2-410m --epochs 20
   ```
   Writes `summary.json` with test_in / test_held accuracy and 95% CIs.

3. **Sample-efficiency sweep (RQ1).** Loop the single-task trainer with `--max_train` in {100, 250, 500, 1000, 2500, 4000, 6000}.

4. **Capability emergence (RQ2 Exp 6.1) + curricula (Exp 6.2).**
   ```bash
   PYTHONPATH=src python scripts/ahri/train_multitask.py \
       --tasks all --schedule simultaneous --steps 50000 \
       --eval_every 500 --llm EleutherAI/pythia-410m \
       --out results/ahri/multitask/simultaneous
   ```
   Writes `trajectory.json` — a list of per-eval-step task accuracies.
   Repeat with `--schedule {easy_to_hard,hard_to_easy,random_introduction}`.

5. **Interference matrix (RQ2 Exp 6.3).**
   ```bash
   PYTHONPATH=src python scripts/ahri/run_interference_matrix.py \
       --tasks 1.2 2.1 2.2 3.1 3.4 4.1 4.3 5.1 \
       --steps 10000 --llm EleutherAI/pythia-410m
   ```
   Writes `matrix.json` with single, joint, and interference scores.

## Notes

- **No CoT, no stats in text** by design. Every answer must derive from the signal pathway. Don't add reasoning prefixes to prompts.
- **Held-out splits.** Each task defines `heldout_*` in its class. `test_held` is sampled inside that region; `train`, `val`, `test_in` are sampled excluding it. Compare them per-task to detect memorisation vs concept learning.
- **Parquet is the contract.** Trainers never call `task.sample()`. If you change a task's sampler, regenerate that task's parquet.
- **Two-signal tasks.** Tier 3 prompts contain 64 `<|signal|>` placeholders plus one `<|signal_sep|>` between the two signal blocks. `PhysicsTSLM` injects 32+32 = 64 patch embeddings; the SEP token gets its normal LLM embedding.
