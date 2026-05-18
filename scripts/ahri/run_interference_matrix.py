#!/usr/bin/env python3
"""
Build the task-interference matrix (paper Experiment 6.3).

For a set of representative tasks, compute:
  a_i              = single-task accuracy on task i
  a_i^{joint(i,j)} = accuracy on i when trained jointly with j
  I_{ij}           = a_i - a_i^{joint(i,j)}

This script orchestrates per-pair runs by calling train_single_task and
train_multitask in-process (sharing the same Python interpreter so we don't
re-load Pythia weights for every pair).

Output: results/ahri/interference/matrix.json with the full matrix.

Use a SMALL set of representative tasks (paper uses 8) and a SHORT step
budget per pair — this experiment is quadratic in tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from opentslm.ahri.dataset import AhriParquetDataset
from opentslm.ahri.eval import evaluate
from opentslm.ahri.multitask import MultiTaskMixture, SCHEDULES
from opentslm.model.llm.PhysicsTSLM import PhysicsTSLM, PhysicsTSLMConfig


def _run_short(args, task_ids: list[str], steps: int, device) -> dict[str, float]:
    """Train fresh model for `steps` on the given task(s); return final
    test_in accuracy for each task. Always uses uniform sampling."""
    rng = np.random.default_rng(args.seed)
    mixture = MultiTaskMixture(args.data, task_ids, split="train", seed=args.seed)

    cfg = PhysicsTSLMConfig(llm_id=args.llm)
    model = PhysicsTSLM(cfg).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(steps * 0.05), steps)

    for step in tqdm(range(1, steps + 1), desc=f"train {task_ids}", leave=False):
        task_id = task_ids[int(rng.integers(0, len(task_ids)))]
        batch = mixture.sample_batch(task_id, args.batch_size)
        model.train()
        sigs = batch["signals"].to(device)
        tok = model.tokenize(batch["prompts"], batch["answers"], device=device)
        out = model(sigs, **tok)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    accs = {}
    for tid in task_ids:
        ds = AhriParquetDataset(args.data, tid, "test_in")
        res = evaluate(model, ds, batch_size=args.eval_batch_size, device=device, max_new_tokens=args.max_new_tokens)
        accs[tid] = res.accuracy
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["1.2", "2.1", "2.2", "3.1", "3.4", "4.1", "4.3", "5.1"])
    ap.add_argument("--data", default="data/ahri")
    ap.add_argument("--out", default="results/ahri/interference")
    ap.add_argument("--llm", default="EleutherAI/pythia-410m")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    single = {}
    for tid in args.tasks:
        print(f"[single] {tid}")
        single[tid] = _run_short(args, [tid], args.steps, device)[tid]

    joint: dict[tuple[str, str], dict[str, float]] = {}
    n = len(args.tasks)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = args.tasks[i], args.tasks[j]
            print(f"[joint] {ti} + {tj}")
            joint[(ti, tj)] = _run_short(args, [ti, tj], args.steps, device)

    matrix = {}
    for i, ti in enumerate(args.tasks):
        matrix[ti] = {}
        for j, tj in enumerate(args.tasks):
            if i == j:
                matrix[ti][tj] = None
                continue
            key = (ti, tj) if i < j else (tj, ti)
            matrix[ti][tj] = single[ti] - joint[key][ti]   # interference of j on i

    out = {
        "tasks": args.tasks,
        "steps_per_run": args.steps,
        "llm": args.llm,
        "single_task_acc": single,
        "joint_pair_acc": {f"{a}__{b}": v for (a, b), v in joint.items()},
        "interference_matrix": matrix,
    }
    (out_dir / "matrix.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {out_dir/'matrix.json'}")


if __name__ == "__main__":
    main()
