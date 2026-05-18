# Running Ahri on Compute Clusters

This dir contains everything needed to take the Ahri / PhysicsTSLM stack
from a fresh cluster to producing paper-quality results.

## Layout

```
scripts/cluster/
  common/
    check_env.py         portable sanity check (deps, CUDA, HF cache, disk)
    prefetch_models.py   download Pythia weights to a writable cache
  sherlock/
    00_setup_env.sh      one-time bootstrap on Sherlock login node
    99_common.sh         sourced by every SLURM job (modules + venv + offline)
    01_smoke_test.sbatch single-GPU smoke: pytest + 2-step train
    02_generate_data.sbatch  one-time data gen for all 21 tasks
    03_train_single_task.sbatch  RQ1 per-task training
    04_grid_submit.sh    submits the full 21x3 grid for RQ1
    05_train_multitask_ddp.sbatch  RQ2 multi-task on 4 GPUs via torchrun
    06_interference.sbatch  RQ2 Experiment 6.3 (pairwise interference)
```

The Sherlock scripts can be copied verbatim for any SLURM cluster; only
`00_setup_env.sh` and `99_common.sh` need adjusting (module names, scratch
path conventions).

## Stanford Sherlock — recipe

### One-time login node setup

```bash
ssh <SUNetID>@login.sherlock.stanford.edu
cd $SCRATCH && git clone https://github.com/<you>/OpenTSLMExperiments.git
cd OpenTSLMExperiments
bash scripts/cluster/sherlock/00_setup_env.sh
```

This:

1. Loads `python/3.12.1` and CUDA modules
2. Creates a uv-managed venv at `$SCRATCH/ahri/venv`
3. Points `$HF_HOME` at `$SCRATCH/ahri/hf_cache` (compute nodes lack internet,
   so caches must live somewhere both can see)
4. Installs the repo in editable mode
5. Pre-fetches Pythia-160m / 410m / 1.4b weights
6. Runs `check_env.py` to confirm the environment is sane

Re-run any time you bump a dependency.

### Smoke test (single GPU)

```bash
sbatch scripts/cluster/sherlock/01_smoke_test.sbatch
```

This runs the full pytest suite (164 tests) plus a real 2-step training run
on a tiny dataset using Pythia-160m. Should finish in ~10 min on a single
GPU. Inspect `logs/smoke-*.out` — last line must be `[smoke] PASS`.

### Generate datasets (one time)

```bash
sbatch scripts/cluster/sherlock/02_generate_data.sbatch
```

Writes ~250k examples (21 tasks × 12k each) under `$AHRI_DATA_ROOT`.
CPU-only, ~30 min. Optional: upload to HF Hub from a login node:

```bash
source $SCRATCH/ahri/env.sh && source $VENV_DIR/bin/activate
PYTHONPATH=src python scripts/ahri/push_to_hub.py \
    --local "$AHRI_DATA_ROOT" --repo your-org/ahri-v1 --private
```

Once uploaded, other clusters can pull with:

```python
from opentslm.ahri.dataset import AhriParquetDataset
ds = AhriParquetDataset.from_hub("your-org/ahri-v1", "1.2", "train")
```

### RQ1: Learnability map (21 tasks × 3 scales)

```bash
bash scripts/cluster/sherlock/04_grid_submit.sh
```

Submits 63 single-GPU jobs. Each writes `results/.../summary.json`. Monitor:

```bash
squeue -u $USER
ls $AHRI_SCRATCH/runs/rq1/
```

### RQ2: Training dynamics

Multi-task on 4 GPUs via torchrun (1 node):

```bash
SCHEDULE=simultaneous   sbatch scripts/cluster/sherlock/05_train_multitask_ddp.sbatch
SCHEDULE=easy_to_hard   sbatch scripts/cluster/sherlock/05_train_multitask_ddp.sbatch
SCHEDULE=hard_to_easy   sbatch scripts/cluster/sherlock/05_train_multitask_ddp.sbatch
SCHEDULE=random_introduction sbatch scripts/cluster/sherlock/05_train_multitask_ddp.sbatch
```

Interference matrix (single GPU, ~48h budget):

```bash
sbatch scripts/cluster/sherlock/06_interference.sbatch
```

## Porting to another cluster

The non-Sherlock-specific machinery (paths, modules) lives in
`00_setup_env.sh` and `99_common.sh`. To port:

1. Copy `scripts/cluster/sherlock/` to `scripts/cluster/<new_cluster>/`.
2. Adjust `00_setup_env.sh`:
   - Module names: each cluster has its own naming (`python/3.12.x`, `cuda/12.x`).
     Or skip modules entirely if uv ships a Python.
   - Scratch path: most clusters expose `$SCRATCH`; if not, point at a
     writable parallel filesystem (Lustre, GPFS, NFS scratch).
3. Adjust `99_common.sh`:
   - `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are conservative; set
     to `0` if compute nodes have outbound internet.
4. Adjust SLURM directives (`--partition`, `--qos`, GPU constraint flags)
   to your cluster's policies.

## Internet-aware notes

- **Compute nodes without internet** (Sherlock, many HPC sites): run
  `00_setup_env.sh` on the login node — it pre-fetches all Pythia weights to
  `$HF_HOME`. Compute nodes load from cache via `HF_HUB_OFFLINE=1`.
- **Compute nodes with internet** (some cloud setups, university clusters
  with NAT'd egress): set `HF_HUB_OFFLINE=0` in `99_common.sh` and skip
  the prefetch step.
- **Air-gapped clusters**: copy a pre-built HF cache via rsync. The data is
  in `~/.cache/huggingface/hub/` by default.

## Reproducibility

- Data is deterministic in `(task_id, split, seed)`. Identical generator
  output on every cluster as long as Python + numpy match major versions.
- Pythia checkpoints are pinned by HF revision hash via the model id.
- Each `summary.json` records the full argv + world size + seed, so any
  result can be regenerated bit-for-bit.

## Common failure modes

| Symptom                                    | Fix                                                  |
|-------------------------------------------|------------------------------------------------------|
| `ImportError: cannot import name 'Flamingo'` | Expected — `open_flamingo` is optional; Ahri ignores. |
| `Pythia tokenizer not cached`              | Run `00_setup_env.sh` on a login node first.         |
| `disk full` during data gen                | Move `$AHRI_DATA_ROOT` to a bigger scratch.          |
| `NCCL timeout`                             | Reduce `--nproc_per_node` or check GPU interconnect. |
| `OOM` on Pythia-1.4b                       | Drop `--batch_size` from 32 to 8.                    |
| Tests pass locally, fail on cluster        | Re-run `check_env.py` — usually a missing module.    |
