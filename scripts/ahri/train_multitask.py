#!/usr/bin/env python3
"""
Multi-task / curriculum trainer for RQ2 (Training Dynamics).

Trains a single PhysicsTSLM on a set of Ahri tasks using one of four
curricula. At regular step intervals (`--eval_every`), it evaluates on each
task's test_in split and writes the trajectory to JSON — this is the data
that produces Figure 4 (capability emergence) in the paper.

Examples:
    # all 21 tasks, simultaneous, 50k steps, eval every 500
    python scripts/ahri/train_multitask.py --tasks all --schedule simultaneous \\
        --steps 50000 --eval_every 500 --llm EleutherAI/pythia-410m

    # easy-to-hard curriculum
    python scripts/ahri/train_multitask.py --tasks all --schedule easy_to_hard \\
        --steps 50000

    # pairwise interference (just two tasks)
    python scripts/ahri/train_multitask.py --tasks 1.2 3.1 --schedule simultaneous \\
        --steps 20000 --out results/ahri/interference/1.2-vs-3.1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from opentslm.ahri.dataset import AhriParquetDataset, resolve_root
from opentslm.ahri.distributed import (
    barrier,
    cleanup_distributed,
    is_main_rank,
    setup_distributed,
    world_size,
)
from opentslm.ahri.eval import evaluate
from opentslm.ahri.multitask import SCHEDULES, MultiTaskMixture
from opentslm.ahri.tasks import list_task_ids
from opentslm.model.llm.PhysicsTSLM import PhysicsTSLM, PhysicsTSLMConfig


def _log(msg: str):
    if is_main_rank():
        print(msg, flush=True)


def train(args):
    distributed = world_size() > 1
    device = setup_distributed() if (distributed or args.device == "auto") else torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + (10_000 * (0 if not distributed else int(args.seed))))

    task_ids = list_task_ids() if args.tasks == ["all"] else args.tasks
    _log(f"[setup] schedule={args.schedule} tasks={task_ids} world_size={world_size()}")

    root = resolve_root(args.data)
    mixture = MultiTaskMixture(root, task_ids, split="train", seed=args.seed)
    schedule = SCHEDULES[args.schedule](task_ids)
    eval_sets = {tid: AhriParquetDataset(root, tid, "test") for tid in task_ids}

    cfg = PhysicsTSLMConfig(llm_id=args.llm)
    model = PhysicsTSLM(cfg).to(device)
    inner = model
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=False)
    _log(f"[setup] trainable_params={inner.num_trainable_params():,}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=int(args.steps * 0.05),
        num_training_steps=args.steps,
    )

    out_dir = Path(args.out)
    if is_main_rank():
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    trajectory: list[dict] = []

    pbar = tqdm(range(1, args.steps + 1), desc="train", disable=not is_main_rank(), mininterval=2.0)
    running_loss = 0.0
    n_running = 0
    for step in pbar:
        active = schedule.active_tasks(step, args.steps)
        task_id = active[int(rng.integers(0, len(active)))]
        batch = mixture.sample_batch(task_id, args.batch_size)

        model.train()
        sigs = batch["signals"].to(device)
        tok = inner.tokenize(batch["prompts"], batch["answers"], device=device)
        out = model(sigs, **tok)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running_loss += out.loss.item()
        n_running += 1

        if is_main_rank() and step % args.log_every == 0:
            pbar.set_postfix(loss=f"{running_loss / max(n_running, 1):.3f}", active=len(active))
            running_loss = 0.0
            n_running = 0

        if step % args.eval_every == 0 or step == args.steps:
            barrier()
            if is_main_rank():
                entry = {"step": step, "active_tasks": active, "task_acc": {}}
                for tid in task_ids:
                    res = evaluate(
                        inner,
                        eval_sets[tid],
                        batch_size=args.eval_batch_size,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                        desc=f"eval@{step} {tid}",
                    )
                    entry["task_acc"][tid] = {
                        "accuracy": res.accuracy,
                        "accuracy_ci": res.accuracy_ci,
                        "n": res.n,
                    }
                trajectory.append(entry)
                (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
                torch.save({"model_state": inner.state_dict(), "step": step}, out_dir / "last.pt")
            barrier()

    _log(f"[done] wrote {out_dir/'trajectory.json'}")
    cleanup_distributed()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True, help="Task IDs or 'all'")
    ap.add_argument("--schedule", choices=list(SCHEDULES.keys()), default="simultaneous")
    ap.add_argument("--data", default="data/ahri")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm", default="EleutherAI/pythia-410m")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
