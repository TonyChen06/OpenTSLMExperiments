<!--
SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.

SPDX-License-Identifier: MIT
-->

# Physics of Time-Series Language Models — Part 1: The Capability Frontier

This branch (`PhysicsTSLM1`) is the code for an in-progress paper that maps
the primitive capabilities of Time-Series Language Models (TSLMs) using a
controlled synthetic benchmark, **Ahri** — *Ascending Harmonic Reasoning
Instruction* — and a minimal encoder-free model, **PhysicsTSLM**.

The work builds on top of [OpenTSLM (StanfordBDHG)](https://github.com/StanfordBDHG/OpenTSLM)
but the contributions here are independent of OpenTSLM's clinical pipeline:
a separate dataset framework, a separate model class, and a separate test +
training stack.

**Dataset:** [🤗 TonyChen06/AscendingHarmonicReasoningInstruction](https://huggingface.co/datasets/TonyChen06/AscendingHarmonicReasoningInstruction)

---

## What's in this branch

| Component | Path | Purpose |
|---|---|---|
| **Ahri framework** | [`src/opentslm/ahri/`](src/opentslm/ahri/) | 6 waveform primitives + 3 composition rules; 21 `AhriTask` classes across 5 tiers; standardised prompt template; bootstrap-CI eval harness; multi-task curriculum schedulers. |
| **PhysicsTSLM model** | [`src/opentslm/model/llm/PhysicsTSLM.py`](src/opentslm/model/llm/PhysicsTSLM.py) | Encoder-free TSLM: single `Linear(32, d_model)` patch projection + learned positional encoding + Pythia/HF backbone. DDP-compatible. |
| **Trainers** | [`scripts/ahri/`](scripts/ahri/) | Single-task (RQ1), multi-task with curriculum scheduling (RQ2), pairwise interference matrix (RQ2 Exp 6.3), results aggregator, HF Hub publisher. |
| **Cluster scripts** | [`scripts/cluster/`](scripts/cluster/) | Portable env smoke check, model prefetch, Stanford Sherlock SLURM bootstrap + smoke + grid + multi-GPU DDP jobs. |
| **Test suite** | [`tests/ahri/`](tests/ahri/) | 172 tests: waveforms, all 21 tasks, graders, dataset loader, model forward/backward/generate, eval, multi-task schedules, distributed helpers, environment smoke, end-to-end pipeline. |

---

## Quick start (any cluster, 4 copy-paste blocks)

The dataset is on HuggingFace; the four blocks below take you from a clean
clone to a full results table on the smallest scale (Pythia-160m, all 21
tasks). Run top to bottom.

### 1 — Install dependencies

```bash
git clone https://github.com/TonyChen06/OpenTSLMExperiments.git
cd OpenTSLMExperiments
git checkout PhysicsTSLM1

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install pytest
```

Expected last line: `Successfully installed pytest-…`

### 2 — Verify dependencies + CUDA

```bash
PYTHONPATH=src python scripts/cluster/common/check_env.py --gpu
```

The diagnostic prints (1) a context header (hostname, Python, OS, CUDA
driver, git branch/commit, SLURM job info if any), (2) one line per
check tagged `[ OK ]` / `[FAIL]` / `[SKIP]`, and (3) on failure, a `FIX:`
hint for each failed check and a summary at the bottom.

Exit code is `0` if everything passes, `1` if anything fails, `2` if the
diagnostic itself crashed.

**If something fails:** the FIX hint tells you what to do. If unclear,
re-run with `--json` and send the output:

```bash
PYTHONPATH=src python scripts/cluster/common/check_env.py --gpu --json > diag.json
```

`diag.json` contains the context, per-check status + detail + fix hint +
traceback. Pasting that one file gives Tony everything needed to ship a
fix.

Optional flags to add deeper checks:
- `--models EleutherAI/pythia-160m EleutherAI/pythia-410m` — verify these
  HF models are already in the local cache (compute nodes typically lack
  internet, so weights must be prefetched on a login node).
- `--ahri-dataset TonyChen06/AscendingHarmonicReasoningInstruction` —
  verify the Ahri parquet is cached.
- `--data-dir /path/to/scratch` — verify a path exists, is writable, and
  has enough free disk.
- `--check-distributed` — verify `torch.distributed` can initialize
  (single-proc gloo round-trip).

### 3 — Download the dataset

```bash
export AHRI_DATA_ROOT=$(PYTHONPATH=src python -c "from huggingface_hub import snapshot_download; import os; print(os.path.join(snapshot_download(repo_id='TonyChen06/AscendingHarmonicReasoningInstruction', repo_type='dataset', allow_patterns=['data/*/*']), 'data'))")

ls "$AHRI_DATA_ROOT" | head   # should print: 1.1  1.2  1.3  1.4 …
```

~1.2 GB across 84 parquet files (21 tasks × 4 splits). The trainer will
also lazy-download if you skip this step.

### 4 — Train + evaluate on all 21 tasks (Pythia-160m, smallest scale)

```bash
LLM=EleutherAI/pythia-160m \
DATA="$AHRI_DATA_ROOT" \
GPU=0 \
bash scripts/ahri/run_all_tasks.sh
```

Sequential single-GPU loop. Resumable (tasks with `summary.json` are
skipped on re-run). At the end, prints + writes
`results/ahri/pythia-160m/table.md`:

```
| Task | Tier | n_in | acc_in | n_held | acc_held | gap |
|------|------|------|--------|--------|----------|-----|
| 1.1  | 1    | 1421 | 100.0% |  579   |  98.8%   | +1.2%
| 1.2  | 1    | 1483 | 100.0% |  517   |  86.8%   | +13.2%
…

Mean in-dist accuracy:  …
Mean held-out accuracy: …
Tasks solved (>=95%):   in-dist X/21, held-out Y/21
```

Subset: `TASKS="1.1 1.2 1.3" bash scripts/ahri/run_all_tasks.sh`
Larger scale: `LLM=EleutherAI/pythia-410m …` or `pythia-1.4b`.

For parallel SLURM submission on Stanford Sherlock, see
[`scripts/cluster/sherlock/QUICKSTART.md`](scripts/cluster/sherlock/QUICKSTART.md).

---

## What Ahri is

A controlled evaluation framework for TSLMs. **21 synthetic tasks** in
**5 tiers** of increasing cognitive demand:

| Tier | Capability | Tasks |
|------|------------|-------|
| 1 | **Detection** — perceive a categorical property | trend direction, frequency band, periodicity, event presence |
| 2 | **Measurement** — extract a quantitative value | frequency estimation, peak counting, event localization, change-point |
| 3 | **Comparison** — relate two signals | frequency comparison, amplitude comparison, count comparison, temporal lag |
| 4 | **Temporal reasoning** — track change over time | freq change direction, freq change measurement, envelope class, segment labeling, regularity |
| 5 | **Compositional reasoning** — combine capabilities | 2-feature conjunction, 3-feature conjunction, anomaly ID, spectral decomposition |

Every task uses parametrically generated waveforms at $f_s = 200$ Hz,
$N = 1024$ samples. Each task defines a **held-out parameter region**
(paper Table 1): train + val exclude it; the test split samples uniformly
from the full range, and the per-example `held` flag distinguishes
in-distribution from held-out at evaluation time. This lets us tell
concept learning from interpolation.

## What PhysicsTSLM is

A deliberately minimal architecture so results reflect what the LLM can
extract from raw signal patches, not what a specialised encoder feeds it:

```
signal (N=1024)  ──┬─ Linear(32 → d_model) + learned positional encoding ──┐
                   │                                                       │
                   └─ 32 patch tokens replace <|signal|> placeholders ─────┴── Pythia LLM
                                                                              (fully fine-tuned)
```

Three scales supported: `EleutherAI/pythia-{160m, 410m, 1.4b}` — the Pythia
suite shares architecture and training data across scales, isolating the
effect of scale.

---

## Reproducing the paper

| Research question | Script | Notes |
|---|---|---|
| RQ1 — learnability map (21 tasks × 3 scales) | `scripts/ahri/run_all_tasks.sh` (loop over `LLM`) | Sequential or SLURM grid via `scripts/cluster/sherlock/04_grid_submit.sh` |
| RQ1 — sample efficiency | `train_single_task.py --max_train {100,250,…,6000}` | Per-task learning curves |
| RQ2 — capability emergence | `scripts/ahri/train_multitask.py --schedule simultaneous` | Writes a per-eval-step trajectory JSON |
| RQ2 — curricula | `--schedule {easy_to_hard, hard_to_easy, random_introduction}` | Same script |
| RQ2 — task interference | `scripts/ahri/run_interference_matrix.py` | Pairwise on a representative 8-task subset |

Documentation: [`src/opentslm/ahri/README.md`](src/opentslm/ahri/README.md)
(framework internals), [`scripts/cluster/README.md`](scripts/cluster/README.md)
(cluster portability guide).

---

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ahri/ -q
```

172 tests; full suite runs in ~50 s. The end-to-end test
([`tests/ahri/test_pipeline_end_to_end.py`](tests/ahri/test_pipeline_end_to_end.py))
generates tiny data, trains 2 steps via subprocess, and verifies the JSON
summary — that's the integration check for the full stack.

---

## Status

| Component | Status |
|---|---|
| Ahri dataset (21 tasks, 6k/2k/2k) | ✅ on [HF Hub](https://huggingface.co/datasets/TonyChen06/AscendingHarmonicReasoningInstruction) |
| PhysicsTSLM model | ✅ forward / backward / generate verified |
| Single-task trainer (RQ1) | ✅ DDP-tested |
| Multi-task trainer + curricula (RQ2) | ✅ DDP-tested |
| Interference matrix runner | ✅ written; awaiting compute |
| Test suite | ✅ 172 passing |
| Pipeline validated end-to-end | ✅ Pythia-160m on task 1.2: 100% in-dist, 86.85% held-out |

---

## Relationship to OpenTSLM (upstream)

This branch lives inside a fork of [`StanfordBDHG/OpenTSLM`](https://github.com/StanfordBDHG/OpenTSLM)
and reuses its packaging (`pyproject.toml`), prompt classes, and overall
project structure. It does **not** reuse OpenTSLM's clinical model classes
(`OpenTSLMSP`, `OpenTSLMFlamingo`) — those targets clinical multi-modal
benchmarks; PhysicsTSLM is a different architecture designed for the
synthetic-perception experiments of this paper. The two coexist in the
package and don't import each other.

If you're looking for the OpenTSLM clinical pipeline (ECG-QA, HAR, sleep,
TSQA), use the upstream [`main`](https://github.com/StanfordBDHG/OpenTSLM)
branch.

---

## License

MIT. See [`LICENSE.md`](LICENSE.md).
