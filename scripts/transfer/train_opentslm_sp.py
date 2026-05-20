#!/usr/bin/env python3
"""
Trainer for the AHRI-as-pretraining transfer experiment.

Trains OpenTSLM-SP (TransformerCNNEncoder + MLPProjector + frozen LLM) on
any QADataset-compatible dataset and writes a checkpoint of the encoder +
projector. Optionally warm-starts from a pretrained encoder+projector
checkpoint (the "treatment" condition).

Usage:
    # Pretrain on all 21 Ahri tasks
    python scripts/transfer/train_opentslm_sp.py \\
        --dataset ahri --ahri_data data/ahri \\
        --llm HuggingFaceTB/SmolLM2-360M \\
        --out results/transfer/pretrain-ahri \\
        --steps 10000 --batch_size 4

    # Downstream control: train from scratch on TSQA
    python scripts/transfer/train_opentslm_sp.py \\
        --dataset tsqa \\
        --llm HuggingFaceTB/SmolLM2-360M \\
        --out results/transfer/tsqa-control \\
        --epochs 5

    # Downstream treatment: load AHRI-pretrained encoder+projector, fine-tune on TSQA
    python scripts/transfer/train_opentslm_sp.py \\
        --dataset tsqa \\
        --llm HuggingFaceTB/SmolLM2-360M \\
        --load_checkpoint results/transfer/pretrain-ahri/best.pt \\
        --out results/transfer/tsqa-treatment \\
        --epochs 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


# --------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------

def build_dataset(args, eos_token: str):
    """Returns (train_ds, val_ds, test_ds). `val_ds` and `test_ds` may be None."""
    name = args.dataset.lower()
    if name == "ahri":
        from opentslm.ahri.opentslm_adapter import ahri_all_tasks, ahri_subset
        from opentslm.ahri.tasks import list_task_ids
        task_ids = list_task_ids() if args.ahri_tasks == ["all"] else args.ahri_tasks
        if task_ids == list_task_ids():
            train = ahri_all_tasks(args.ahri_data, "train", EOS_TOKEN=eos_token)
            val = ahri_all_tasks(args.ahri_data, "val", EOS_TOKEN=eos_token)
            test = ahri_all_tasks(args.ahri_data, "test", EOS_TOKEN=eos_token)
        else:
            train = ahri_subset(args.ahri_data, task_ids, "train", EOS_TOKEN=eos_token)
            val = ahri_subset(args.ahri_data, task_ids, "val", EOS_TOKEN=eos_token)
            test = ahri_subset(args.ahri_data, task_ids, "test", EOS_TOKEN=eos_token)
        return train, val, test

    if name == "tsqa":
        from opentslm.time_series_datasets.TSQADataset import TSQADataset
        return (TSQADataset("train", eos_token),
                TSQADataset("validation", eos_token),
                TSQADataset("test", eos_token))

    if name == "har_cot":
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
        return (HARCoTQADataset("train", eos_token),
                HARCoTQADataset("validation", eos_token),
                HARCoTQADataset("test", eos_token))

    if name == "sleep":
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
        return (SleepEDFCoTQADataset("train", eos_token),
                SleepEDFCoTQADataset("validation", eos_token),
                SleepEDFCoTQADataset("test", eos_token))

    raise ValueError(f"Unknown dataset: {name}")


# --------------------------------------------------------------------
# Checkpoint I/O — encoder + projector + (optional) LoRA on the LLM
#
# This mirrors the upstream OpenTSLM-SP curriculum: stages 1-2 train only
# the encoder + projector; stage 3+ enable LoRA and the LLM adapts via
# those adapters. We always enable_lora() before training (matching the
# full SP recipe), so checkpoints carry LoRA state too.
# --------------------------------------------------------------------

def save_checkpoint(model: OpenTSLMSP, path: Path, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "encoder": model.encoder.state_dict(),
        "projector": model.projector.state_dict(),
        "extra": extra or {},
    }
    # adds 'lora_enabled' + 'lora_state' if LoRA is on; no-op otherwise
    model.save_lora_state_to_checkpoint(ckpt)
    torch.save(ckpt, path)


def load_checkpoint(model: OpenTSLMSP, path: Path):
    state = torch.load(path, map_location=model.device, weights_only=False)
    model.encoder.load_state_dict(state["encoder"])
    model.projector.load_state_dict(state["projector"])
    # if the checkpoint has LoRA, load it; require LoRA to be enabled on model
    if state.get("lora_enabled", False):
        model.load_lora_state_from_checkpoint(state, allow_missing=False)
    return state.get("extra", {})


# --------------------------------------------------------------------
# Eval (exact-match accuracy on the test set)
# --------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(model: OpenTSLMSP, loader: DataLoader, max_new_tokens: int = 16) -> dict:
    model.eval()
    correct = 0
    total = 0
    sample_predictions = []
    for batch in tqdm(loader, desc="eval", leave=False, mininterval=2.0):
        preds = model.generate(batch, max_new_tokens=max_new_tokens,
                                pad_token_id=model.tokenizer.pad_token_id)
        for p, ex in zip(preds, batch):
            gt = ex["answer"].strip().lower().rstrip("</s>").strip()
            pred = p.strip().lower()
            # exact-substring match: pred contains gt as a substring
            is_correct = gt and (gt in pred or pred in gt)
            correct += int(bool(is_correct))
            total += 1
            if len(sample_predictions) < 10:
                sample_predictions.append({"gt": gt, "pred": pred[:80]})
    return {
        "n": total,
        "accuracy": correct / max(total, 1),
        "samples": sample_predictions,
    }


@torch.no_grad()
def evaluate_loss(model: OpenTSLMSP, loader: DataLoader) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        loss = model.compute_loss(batch)
        total += loss.item() * len(batch)
        n += len(batch)
    return total / max(n, 1)


# --------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[setup] dataset={args.dataset} llm={args.llm}")
    model = OpenTSLMSP(llm_id=args.llm, device=str(device))
    eos = model.get_eos_token()

    # Enable LoRA BEFORE loading any checkpoint (so the LoRA params exist to
    # be populated). This matches Stanford's stage-3+ curriculum: encoder +
    # projector + LoRA on LLM all train together.
    if args.enable_lora:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout)

    if args.load_checkpoint:
        extra = load_checkpoint(model, Path(args.load_checkpoint))
        print(f"[setup] warm-started encoder+projector"
              + (" + LoRA" if args.enable_lora else "")
              + f" from {args.load_checkpoint}")
        print(f"        original train extra: {extra}")

    train_ds, val_ds, test_ds = build_dataset(args, eos_token=eos)
    print(f"[setup] dataset sizes: train={len(train_ds)} val={len(val_ds) if val_ds else None} test={len(test_ds) if test_ds else None}")

    collate = lambda b: extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers) if test_ds else None

    # only encoder + projector are trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[setup] trainable params: {sum(p.numel() for p in trainable_params):,}")

    # step budget: epochs *or* explicit --steps
    if args.steps is not None:
        total_steps = args.steps
    else:
        total_steps = len(train_loader) * args.epochs

    opt = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=int(total_steps * 0.05),
                                             num_training_steps=total_steps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    history: list[dict] = []
    step = 0
    epoch = 0

    print(f"[train] starting; total_steps={total_steps}")
    t0 = time.time()
    while step < total_steps:
        epoch += 1
        model.train()
        running, n_running = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False, mininterval=2.0)
        for batch in pbar:
            loss = model.compute_loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n_running += 1
            step += 1
            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{running / max(n_running, 1):.3f}", step=step)
            if step >= total_steps:
                break

        train_loss = running / max(n_running, 1)
        entry = {"epoch": epoch, "step": step, "train_loss": train_loss}
        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader)
            entry["val_loss"] = val_loss
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                save_checkpoint(model, out_dir / "best.pt", extra={"step": step, "val_loss": val_loss})
                entry["saved_best"] = True
        else:
            save_checkpoint(model, out_dir / "best.pt", extra={"step": step})
        history.append(entry)
        elapsed = (time.time() - t0) / 60
        print(f"[epoch {epoch:3d}] step={step} train_loss={train_loss:.4f}"
              + (f" val_loss={entry.get('val_loss'):.4f}" if 'val_loss' in entry else "")
              + f" elapsed={elapsed:.1f}m")

    # final eval on test set
    test_metrics = {}
    if test_loader is not None:
        load_checkpoint(model, out_dir / "best.pt")
        test_metrics = evaluate_accuracy(model, test_loader, max_new_tokens=args.max_new_tokens)
        print(f"[test] accuracy = {test_metrics['accuracy']:.4f}  (n={test_metrics['n']})")
        for s in test_metrics["samples"][:5]:
            print(f"  gt={s['gt']!r:30s}  pred={s['pred']!r}")

    summary = {
        "dataset": args.dataset,
        "llm": args.llm,
        "load_checkpoint": args.load_checkpoint,
        "total_steps": total_steps,
        "best_val_loss": best_val if best_val != math.inf else None,
        "test_metrics": test_metrics,
        "history": history,
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] wrote {out_dir/'summary.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["ahri", "tsqa", "har_cot", "sleep"])
    ap.add_argument("--ahri_data", default="data/ahri", help="Local path or HF repo id for Ahri")
    ap.add_argument("--ahri_tasks", nargs="+", default=["all"],
                    help="Subset of Ahri task ids; default = all 21")
    ap.add_argument("--llm", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--load_checkpoint", default=None,
                    help="Warm-start encoder+projector from this .pt path (treatment condition)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=None, help="If set, overrides --epochs")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--enable_lora", action="store_true", default=True,
                    help="Enable LoRA on the LLM (matches full upstream SP recipe). Default on.")
    ap.add_argument("--no-enable_lora", dest="enable_lora", action="store_false",
                    help="Disable LoRA (encoder+projector only; LLM frozen throughout).")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
