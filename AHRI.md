# Ahri — Cluster Quickstart

Four copy-paste blocks. Run them top to bottom on any cluster.
Each block stands alone (no hidden state between them).

The dataset lives on HuggingFace:
**[TonyChen06/AscendingHarmonicReasoningInstruction](https://huggingface.co/datasets/TonyChen06/AscendingHarmonicReasoningInstruction)**

---

## 1. Install dependencies

```bash
git clone https://github.com/<your-fork>/OpenTSLMExperiments.git
cd OpenTSLMExperiments
git checkout PhysicsTSLM1

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install pytest
```

**Expected last line:** `Successfully installed pytest-...`. If `pip install -e .` fails, paste the error and stop — Tony will ship a fix.

---

## 2. Test that every dependency is in place

```bash
PYTHONPATH=src python scripts/cluster/common/check_env.py --gpu
```

**Expected last line:** `Environment check PASSED. Ready to train.`

Every preceding line should end in `OK`. The check covers numpy, torch, transformers, huggingface_hub, datasets, pyarrow, tqdm, CUDA visibility, and the Ahri import path. If any line says `FAIL`, copy that line and the traceback to Tony.

Drop `--gpu` if you only want to verify CPU-side deps (no CUDA needed).

---

## 3. Download the dataset from HuggingFace

```bash
PYTHONPATH=src python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id='TonyChen06/AscendingHarmonicReasoningInstruction',
    repo_type='dataset',
    allow_patterns=['data/*/*'],
)
data_root = os.path.join(path, 'data')
print('AHRI_DATA_ROOT=' + data_root)
"
```

**Expected last line:** `AHRI_DATA_ROOT=/path/to/cache/.../data`. Copy that into the shell:

```bash
export AHRI_DATA_ROOT=$(PYTHONPATH=src python -c "from huggingface_hub import snapshot_download; import os; print(os.path.join(snapshot_download(repo_id='TonyChen06/AscendingHarmonicReasoningInstruction', repo_type='dataset', allow_patterns=['data/*/*']), 'data'))")
ls "$AHRI_DATA_ROOT" | head -5   # should print: 1.1  1.2  1.3  ...
```

Total download is ~1.2 GB across 84 parquet files. It is cached, so re-runs are a no-op.

**Alternative (no explicit download):** skip this block entirely. The trainer will lazily download when given the HF repo id as `--data`. Block 3 is just for separating download time from training time.

---

## 4. Train and evaluate on ALL 21 tasks with Pythia-160m (smallest scale)

Sequential single-GPU loop. Each task takes ~5–15 min on a modern GPU; full sweep is a few hours.

```bash
LLM=EleutherAI/pythia-160m \
DATA="$AHRI_DATA_ROOT" \
GPU=0 \
bash scripts/ahri/run_all_tasks.sh
```

This:
1. Loops train+eval over all 21 tasks (`1.1` through `5.4`).
2. Writes per-task `results/ahri/pythia-160m/{task_id}/summary.json` with in-dist and held-out accuracy.
3. Resumes — already-completed tasks are skipped on re-run.
4. At the end, prints + saves a markdown results table to `results/ahri/pythia-160m/table.md`.

**Expected final output:**

```
========== aggregate ==========
Results for results/ahri/pythia-160m (21 tasks)

| Task | Tier | n_in | acc_in | n_held | acc_held | gap |
|------|------|------|--------|--------|----------|-----|
| 1.1  | 1    | xxxx | xx.x%  |  xxx   | xx.x%    | +x.x%
| 1.2  | 1    | 1483 | 100.0% |  517   | 86.8%    | +13.2%
...
```

If you want to run just a subset of tasks:
```bash
TASKS="1.1 1.2 1.3" LLM=EleutherAI/pythia-160m bash scripts/ahri/run_all_tasks.sh
```

If you want a different scale (Pythia-410m or -1.4b), just change `LLM`:
```bash
LLM=EleutherAI/pythia-410m bash scripts/ahri/run_all_tasks.sh   # ~3x slower per task
LLM=EleutherAI/pythia-1.4b bash scripts/ahri/run_all_tasks.sh   # ~10x slower; may need --batch_size 8
```

---

## What's running underneath each block

| Block | Script                                              | What it does                          |
|-------|-----------------------------------------------------|---------------------------------------|
| 1     | `pip install -e .`                                  | Installs Ahri package + dependencies  |
| 2     | `scripts/cluster/common/check_env.py`               | Checks imports, CUDA, disk, HF cache  |
| 3     | `huggingface_hub.snapshot_download(...)`            | Fetches the 21-task parquet shards    |
| 4     | `scripts/ahri/run_all_tasks.sh` → `train_single_task.py` → `aggregate_results.py` | Loops train+eval, aggregates results |

## SLURM clusters

For Stanford Sherlock (or any SLURM cluster), use `scripts/cluster/sherlock/` for parallel submission. See [scripts/cluster/sherlock/QUICKSTART.md](scripts/cluster/sherlock/QUICKSTART.md).

The sequential Block 4 will work anywhere with one GPU; the SLURM scripts fan out across many.

## If something fails

Pin the error to whichever of the four blocks it came from. Tony fixes locally and you `git pull` + re-run that block.

| Block | If it fails, paste...                              |
|-------|-----------------------------------------------------|
| 1     | the last 20 lines of `pip install` output           |
| 2     | the full `check_env.py` output                      |
| 3     | the traceback + `python --version` + `pip show huggingface_hub` |
| 4     | the contents of the offending `logs/<task>.log`     |
