"""
One-shot dataset generator. Produces, for each task, four parquet files:

    data/ahri/{task_id}/train.parquet     (6000 examples)
    data/ahri/{task_id}/val.parquet       (2000 examples)
    data/ahri/{task_id}/test_in.parquet   (2000 examples)
    data/ahri/{task_id}/test_held.parquet (2000 examples)

Plus a single `manifest.json` per task with the task metadata. We split test
into in-distribution and held-out files explicitly so the trainer/evaluator
can read them as separate splits — matching the paper's reporting (Table 3).

Each row contains:
    signals: list[list[float32]]  # shape (num_signals, 1024)
    prompt:  str
    answer:  str
    held:    bool
    params:  json-serialized dict
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from opentslm.ahri.tasks import TASK_REGISTRY, list_task_ids
from opentslm.ahri.tasks.base import AhriTask, Split


# Paper Section 3.3: 6k train / 2k val / 2k test.
# Test is sampled uniformly from the FULL parameter range (including the
# held-out region). At eval time, accuracy is reported overall AND broken
# down by the per-example `held` flag.
#
# `pretrain` is a separate split for the AHRI-as-pretraining transfer
# experiment: 3k examples per task drawn from the FULL parameter range
# (no held-out exclusion). It does not feed RQ1/RQ2; it exists only so a
# downstream model can see the full waveform vocabulary in one pass.
SPLIT_SIZES: dict[Split, int] = {
    "train": 6000,
    "val": 2000,
    "test": 2000,
    "pretrain": 3000,
}


def _generate_split(task: AhriTask, split: Split, size: int, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    rows = {
        "signals": [],
        "prompt": [],
        "answer": [],
        "held": [],
        "params": [],
    }
    # mininterval=2 keeps tqdm from flooding non-tty terminals with refreshes
    for _ in tqdm(range(size), desc=f"  {task.task_id} {split}", leave=False, mininterval=2.0):
        ex = task.sample(rng, split)
        # serialize signals as list-of-lists; pyarrow handles ragged outer list
        rows["signals"].append([s.tolist() for s in ex.signals])
        rows["prompt"].append(ex.prompt)
        rows["answer"].append(ex.answer)
        rows["held"].append(bool(ex.held))
        rows["params"].append(json.dumps(_to_jsonable(ex.params), default=str))
    return pa.table(rows)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def generate_task(task_id: str, out_root: Path, base_seed: int = 0, overwrite: bool = False):
    task = TASK_REGISTRY[task_id]()
    task_dir = out_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task_id": task.task_id,
        "tier": task.tier,
        "question": task.question,
        "output_format": task.output_format,
        "num_signals": task.num_signals,
        "labels": list(task.labels) if task.labels else None,
        "split_sizes": SPLIT_SIZES,
    }
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for i, (split, size) in enumerate(SPLIT_SIZES.items()):
        out_path = task_dir / f"{split}.parquet"
        if out_path.exists() and not overwrite:
            print(f"  skip {task_id}/{split} (exists)")
            continue
        # split-specific seed = base_seed * 1000 + task index * 10 + split index
        seed = base_seed + int(task_id.replace(".", "")) * 100 + i
        table = _generate_split(task, split, size, seed)
        pq.write_table(table, out_path, compression="zstd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/ahri")
    ap.add_argument("--tasks", nargs="*", default=None, help="Subset of task IDs; default = all 21")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    task_ids = args.tasks or list_task_ids()
    print(f"Generating {len(task_ids)} task(s) into {out_root.resolve()}")
    for tid in task_ids:
        if tid not in TASK_REGISTRY:
            print(f"  unknown task '{tid}', skipping")
            continue
        print(f"Task {tid}")
        generate_task(tid, out_root, base_seed=args.seed, overwrite=args.overwrite)
    print("Done.")


if __name__ == "__main__":
    main()
