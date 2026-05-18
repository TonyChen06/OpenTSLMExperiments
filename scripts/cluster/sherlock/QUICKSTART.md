# Sherlock Quickstart

A copy-paste sequence to bring up the Ahri pipeline on Stanford Sherlock.
Run each block, watch the output, move on when it says it's done.

If anything fails: stop, Tony will look at the error and ship a fix.

---

## 0. ssh in and clone

```bash
ssh sherlock
cd $SCRATCH
git clone https://github.com/<repo>/OpenTSLMExperiments.git
cd OpenTSLMExperiments
git checkout PhysicsTSLM1
```

**Expected:** clean clone, no errors. Then:

```bash
git log --oneline -3
```
should show recent commits.

---

## 1. Bootstrap (login node, ~5 min)

```bash
bash scripts/cluster/sherlock/00_setup_env.sh
```

What it does, in order:
1. `module load python/3.12.1 cuda/12.x` — load Sherlock modules
2. Create a uv venv at `$SCRATCH/ahri/venv`
3. Install the project in editable mode + pytest
4. Pre-fetch Pythia-160m / 410m / 1.4b weights to `$SCRATCH/ahri/hf_cache`
5. Run a local environment check

**Expected last line:**
```
[setup] DONE.
[setup] Source env in future shells: source $SCRATCH/ahri/env.sh
[setup] Activate venv:                source $SCRATCH/ahri/venv/bin/activate
```

If it complains about a missing module name (e.g. `python/3.12.1` not found),
that's the first thing to fix — Tony will adjust the module string.

---

## 2. Activate the environment in this shell

```bash
source $SCRATCH/ahri/env.sh
source $VENV_DIR/bin/activate
```

**Expected:** prompt now shows `(venv)`. Sanity:

```bash
python -c "import opentslm.ahri; from opentslm.ahri.tasks import TASK_REGISTRY; print(len(TASK_REGISTRY))"
```
prints `21`.

---

## 3. Generate the dataset (login node, ~1 min)

```bash
PYTHONPATH=src python -m opentslm.ahri.generate --out "$AHRI_DATA_ROOT" --seed 0 2>&1 | tail -5
```

**Expected:** last lines say `Done.` Disk usage:

```bash
du -sh "$AHRI_DATA_ROOT"
```
should be ~1.2 GB.

---

## 4. Smoke test on one GPU (submits a SLURM job)

```bash
sbatch scripts/cluster/sherlock/01_smoke_test.sbatch
squeue -u $USER
```

The job:
1. Re-runs the environment check on a compute node
2. Runs the full pytest suite (~1 min)
3. Trains Pythia-160m on task 1.2 for 1 epoch on 64 examples (~2 min)

**Expected:** when it finishes, the last line of `logs/smoke-*.out` is:
```
[smoke] PASS — environment ready for full runs.
```

To watch progress:
```bash
tail -f logs/smoke-*.out
```

To see the verdict:
```bash
tail -1 logs/smoke-*.out
```

---

## 5. (Optional) Push the dataset to HuggingFace

Only if Tony wants to publish for cross-cluster use. Needs `huggingface-cli login`
on a login node first.

```bash
PYTHONPATH=src python scripts/ahri/push_to_hub.py \
    --local "$AHRI_DATA_ROOT" --repo <org>/ahri-v1 --private
```

---

## 6. Kick off the real runs

Once smoke passes:

```bash
# RQ1 grid (63 single-GPU jobs)
bash scripts/cluster/sherlock/04_grid_submit.sh

# RQ2 multi-task on 4 GPUs (one node)
SCHEDULE=simultaneous sbatch scripts/cluster/sherlock/05_train_multitask_ddp.sbatch
```

Monitor with `squeue -u $USER` and `ls $AHRI_SCRATCH/runs/`.

---

## What to do if something fails

**Don't try to debug.** Just run:

```bash
# capture the failing job's output for Tony
cat logs/smoke-*.err logs/smoke-*.out > /tmp/sherlock_failure.txt
head -200 /tmp/sherlock_failure.txt
```

Tony reads the failure, ships a fix, you `git pull` and try again.

Common things that may need a fix:

| Symptom                              | Likely fix                                       |
|--------------------------------------|--------------------------------------------------|
| `module: command not found`          | Wrong shell — run `bash`, not `sh`               |
| `python/3.12.1 not found`            | Tony adjusts the module name in `00_setup_env.sh` |
| `cuda/12.x not found`                | Same, for the CUDA module                        |
| pytest failure                       | Code-side bug — Tony fixes locally               |
| HF auth error on push                | Run `huggingface-cli login` first                |
| `disk full`                          | Move `$AHRI_DATA_ROOT` to bigger scratch          |
| GPU OOM on Pythia-1.4b               | Drop `--batch_size` in the sbatch                |

That's the whole flow.
