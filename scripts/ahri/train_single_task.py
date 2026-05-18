#!/usr/bin/env python3
"""
Single-task trainer for RQ1 (Learnability Map).

Trains one PhysicsTSLM on one Ahri task using the standard protocol from
paper Section 4.3: AdamW, lr=1e-4, cosine decay with 5% warmup, batch 32,
grad clip 1.0, early stop on val loss with patience 10.

After training, evaluates on test_in and test_held and writes a JSON
summary to the results dir.

Usage:
    PYTHONPATH=src python scripts/ahri/train_single_task.py \\
        --task 1.2 --llm EleutherAI/pythia-410m --data data/ahri \\
        --out results/ahri/1.2-410m --epochs 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from opentslm.ahri.dataset import AhriParquetDataset, resolve_root
from opentslm.ahri.distributed import (
    all_reduce_mean,
    barrier,
    cleanup_distributed,
    is_main_rank,
    setup_distributed,
    world_size,
)
from opentslm.ahri.eval import evaluate
from opentslm.model.llm.PhysicsTSLM import PhysicsTSLM, PhysicsTSLMConfig


def _collate(batch: list[dict]) -> dict:
    return {
        "signals": torch.stack([b["signals"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "answers": [b["answer"] for b in batch],
    }


def evaluate_loss(model, loader: DataLoader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    inner = model.module if isinstance(model, DDP) else model
    with torch.no_grad():
        for batch in loader:
            sigs = batch["signals"].to(device)
            tok = inner.tokenize(batch["prompts"], batch["answers"], device=device)
            out = model(sigs, **tok) if isinstance(model, DDP) else inner(sigs, **tok)
            total += out.loss.item() * sigs.size(0)
            n += sigs.size(0)
    return total / max(n, 1)


def _log(msg: str):
    if is_main_rank():
        print(msg, flush=True)


def train(args):
    distributed = world_size() > 1
    device = setup_distributed() if (distributed or args.device == "auto") else torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = resolve_root(args.data)

    train_ds = AhriParquetDataset(root, args.task, "train")
    val_ds = AhriParquetDataset(root, args.task, "val")

    if args.max_train is not None:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(args.max_train)))

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=_collate, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, collate_fn=_collate, num_workers=args.num_workers,
    )

    cfg = PhysicsTSLMConfig(llm_id=args.llm)
    model = PhysicsTSLM(cfg).to(device)
    inner = model
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=False)

    _log(f"[setup] task={args.task} llm={args.llm} train_n={len(train_ds)} val_n={len(val_ds)} "
         f"world_size={world_size()} device={device} trainable_params={inner.num_trainable_params():,}")

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.05)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    out_dir = Path(args.out)
    if is_main_rank():
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    ckpt_path = out_dir / "best.pt"

    best_val = math.inf
    patience = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        nsteps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False, disable=not is_main_rank(), mininterval=2.0)
        for batch in pbar:
            sigs = batch["signals"].to(device)
            tok = inner.tokenize(batch["prompts"], batch["answers"], device=device)
            out = model(sigs, **tok)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += out.loss.item()
            nsteps += 1
            if is_main_rank() and nsteps % 20 == 0:
                pbar.set_postfix(loss=f"{running / nsteps:.4f}")
        train_loss = running / max(nsteps, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        val_loss = all_reduce_mean(val_loss, device) if distributed else val_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        _log(f"[epoch {epoch:3d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience = 0
            if is_main_rank():
                torch.save(
                    {"model_state": inner.state_dict(), "config": asdict(cfg), "epoch": epoch},
                    ckpt_path,
                )
            barrier()
        else:
            patience += 1
            if patience >= args.patience:
                _log(f"[early stop] no val improvement for {patience} epochs")
                break

    # eval — only rank 0 runs the eval on the full split (cheap; loaders not sharded)
    barrier()
    if is_main_rank():
        _log("[eval] loading best checkpoint")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        inner.load_state_dict(state["model_state"])

        ds = AhriParquetDataset(root, args.task, "test")
        res = evaluate(inner, ds, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, device=device)
        results = {
            "test": {
                "n": res.n,
                "accuracy": res.accuracy,
                "accuracy_ci": res.accuracy_ci,
                "n_in": res.n_in,
                "accuracy_in": res.accuracy_in,
                "accuracy_in_ci": res.accuracy_in_ci,
                "n_held": res.n_held,
                "accuracy_held": res.accuracy_held,
                "accuracy_held_ci": res.accuracy_held_ci,
                "extras": res.extras,
            }
        }
        _log(f"[test] overall:  acc={res.accuracy:.4f} ci=({res.accuracy_ci[0]:.3f}, {res.accuracy_ci[1]:.3f}) n={res.n}")
        _log(f"[test] in_dist:  acc={res.accuracy_in:.4f} ci=({res.accuracy_in_ci[0]:.3f}, {res.accuracy_in_ci[1]:.3f}) n={res.n_in}")
        _log(f"[test] held-out: acc={res.accuracy_held:.4f} ci=({res.accuracy_held_ci[0]:.3f}, {res.accuracy_held_ci[1]:.3f}) n={res.n_held}")
        _log(f"[test] extras: {res.extras}")

        summary = {
            "task": args.task,
            "llm": args.llm,
            "epochs_run": history[-1]["epoch"] if history else 0,
            "best_val_loss": best_val,
            "history": history,
            "results": results,
            "args": vars(args),
            "world_size": world_size(),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        _log(f"[done] wrote {out_dir/'summary.json'}")
    cleanup_distributed()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--llm", default="EleutherAI/pythia-410m")
    ap.add_argument("--data", default="data/ahri")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_train", type=int, default=None, help="Sample-efficiency sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu (auto picks cuda+local_rank under torchrun)")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
