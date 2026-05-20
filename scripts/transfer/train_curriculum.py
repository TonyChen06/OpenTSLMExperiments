#!/usr/bin/env python3
"""
Sequential curriculum trainer for the AHRI-as-pretraining transfer
experiment.

Runs an ordered list of training stages on a single OpenTSLM-SP model.
Between stages the model is NOT reset — the encoder, projector, and LoRA
state carry forward, matching Stanford's curriculum_learning.py recipe.

After each stage, runs eval on that stage's test set and writes a
per-stage summary; everything lands in one directory.

Two canonical curricula:

  control:    HAR -> Sleep -> ECG
  treatment:  AHRI -> HAR -> Sleep -> ECG

For the treatment, AHRI is just another stage at position 0; once it's
done the rest is identical to control.

Usage:
    # Control: no AHRI
    PYTHONPATH=src python scripts/transfer/train_curriculum.py \\
        --stages har_cot sleep ecg_qa \\
        --llm meta-llama/Llama-3.2-1B-Instruct \\
        --out results/transfer/control --epochs 3

    # Treatment: AHRI first, then the same downstream order
    PYTHONPATH=src python scripts/transfer/train_curriculum.py \\
        --stages ahri har_cot sleep ecg_qa \\
        --ahri_data data/ahri \\
        --llm meta-llama/Llama-3.2-1B-Instruct \\
        --out results/transfer/treatment --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
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
# Stage definitions
# --------------------------------------------------------------------

STAGE_NAMES = {"ahri", "tsqa", "har_cot", "sleep", "ecg_qa"}


@dataclass
class Stage:
    name: str
    train_size: int
    val_size: int | None
    test_size: int | None

    def __str__(self):
        return f"<Stage {self.name}: train={self.train_size}>"


def build_stage_loaders(stage_name: str, eos_token: str, args, collate) -> tuple[DataLoader, DataLoader | None, DataLoader | None, Stage]:
    from torch.utils.data import Subset
    if stage_name == "ahri":
        from opentslm.ahri.opentslm_adapter import ahri_all_tasks
        # Use the new `pretrain` split for AHRI: 3k/task from full parameter range
        train = ahri_all_tasks(args.ahri_data, "pretrain", EOS_TOKEN=eos_token)
        val = ahri_all_tasks(args.ahri_data, "val", EOS_TOKEN=eos_token)
        # for the AHRI stage we don't need a downstream-style test
        test = None
    elif stage_name == "tsqa":
        from opentslm.time_series_datasets.TSQADataset import TSQADataset
        train = TSQADataset("train", eos_token)
        val = TSQADataset("validation", eos_token)
        test = TSQADataset("test", eos_token)
    elif stage_name == "har_cot":
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
        train = HARCoTQADataset("train", eos_token)
        val = HARCoTQADataset("validation", eos_token)
        test = HARCoTQADataset("test", eos_token)
    elif stage_name == "sleep":
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
        train = SleepEDFCoTQADataset("train", eos_token)
        val = SleepEDFCoTQADataset("validation", eos_token)
        test = SleepEDFCoTQADataset("test", eos_token)
    elif stage_name == "ecg_qa":
        from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
        # ECG-QA-CoT is 12-lead x 1000 samples = 3000 patch tokens/example, which
        # would be 22-68x more compute than the other stages at full scale.
        # Cap train+eval to keep wall time comparable; document the cap in the paper.
        train = ECGQACoTQADataset("train", eos_token, max_samples=args.ecg_max_train)
        val = ECGQACoTQADataset("validation", eos_token, max_samples=args.ecg_max_train)
        test = ECGQACoTQADataset("test", eos_token, max_samples=args.ecg_max_train)
        if args.ecg_max_eval and args.ecg_max_eval < len(val):
            val = Subset(val, list(range(args.ecg_max_eval)))
        if args.ecg_max_eval and args.ecg_max_eval < len(test):
            test = Subset(test, list(range(args.ecg_max_eval)))
    else:
        raise ValueError(f"Unknown stage: {stage_name}")

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers) if val else None
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=args.num_workers) if test else None

    return train_loader, val_loader, test_loader, Stage(
        name=stage_name, train_size=len(train),
        val_size=len(val) if val else None,
        test_size=len(test) if test else None,
    )


# --------------------------------------------------------------------
# Checkpoint I/O (encoder + projector + LoRA)
# --------------------------------------------------------------------

def save_checkpoint(model: OpenTSLMSP, path: Path, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "encoder": model.encoder.state_dict(),
        "projector": model.projector.state_dict(),
        "extra": extra or {},
    }
    model.save_lora_state_to_checkpoint(ckpt)
    torch.save(ckpt, path)


def load_checkpoint(model: OpenTSLMSP, path: Path):
    state = torch.load(path, map_location=model.device, weights_only=False)
    model.encoder.load_state_dict(state["encoder"])
    model.projector.load_state_dict(state["projector"])
    if state.get("lora_enabled", False):
        model.load_lora_state_from_checkpoint(state, allow_missing=False)
    return state.get("extra", {})


# --------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(model: OpenTSLMSP, loader: DataLoader) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        loss = model.compute_loss(batch)
        total += loss.item() * len(batch)
        n += len(batch)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_accuracy(model: OpenTSLMSP, loader: DataLoader, max_new_tokens: int = 32) -> dict:
    model.eval()
    correct = 0
    total = 0
    samples = []
    for batch in tqdm(loader, desc="eval", leave=False, mininterval=2.0):
        preds = model.generate(batch, max_new_tokens=max_new_tokens,
                                pad_token_id=model.tokenizer.pad_token_id)
        for p, ex in zip(preds, batch):
            gt = ex["answer"].strip().lower().rstrip("</s>").strip()
            pred = p.strip().lower()
            ok = bool(gt) and (gt in pred or pred in gt)
            correct += int(ok)
            total += 1
            if len(samples) < 6:
                samples.append({"gt": gt[:80], "pred": pred[:80]})
    return {"n": total, "accuracy": correct / max(total, 1), "samples": samples}


# --------------------------------------------------------------------
# Stage training loop
# --------------------------------------------------------------------

def run_stage(model: OpenTSLMSP, stage_name: str, stage_idx: int, args, out_dir: Path) -> dict:
    print(f"\n{'='*70}\n[stage {stage_idx + 1}] {stage_name}\n{'='*70}")
    collate = lambda b: extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=4)

    train_loader, val_loader, test_loader, stage = build_stage_loaders(
        stage_name, model.get_eos_token(), args, collate
    )
    print(f"[stage {stage_name}] sizes: {stage}")

    total_steps = args.epochs * len(train_loader)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * 0.05), total_steps)

    best_val = math.inf
    history: list[dict] = []
    t0 = time.time()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_running = 0.0, 0
        pbar = tqdm(train_loader, desc=f"  ep {epoch}", leave=False, mininterval=2.0)
        for batch in pbar:
            loss = model.compute_loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n_running += 1
            global_step += 1
            if global_step % args.log_every == 0:
                pbar.set_postfix(loss=f"{running / max(n_running, 1):.3f}", step=global_step)

        train_loss = running / max(n_running, 1)
        entry = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader)
            entry["val_loss"] = val_loss
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                save_checkpoint(model, out_dir / f"{stage_name}_best.pt",
                                 extra={"stage": stage_name, "epoch": epoch, "val_loss": val_loss})
                entry["saved_best"] = True
        history.append(entry)
        elapsed = (time.time() - t0) / 60
        print(f"  [stage {stage_name} epoch {epoch}/{args.epochs}] train={train_loss:.4f}"
              + (f" val={entry.get('val_loss'):.4f}" if 'val_loss' in entry else "")
              + f" elapsed={elapsed:.1f}m")

    # Always save end-of-stage checkpoint (used by the next stage)
    save_checkpoint(model, out_dir / f"{stage_name}_final.pt",
                     extra={"stage": stage_name, "epoch": args.epochs})

    # Eval test set
    test_metrics = {}
    if test_loader is not None:
        # If we saved a best by val_loss, reload it for test eval
        best_path = out_dir / f"{stage_name}_best.pt"
        if best_path.exists():
            load_checkpoint(model, best_path)
        test_metrics = evaluate_accuracy(model, test_loader, max_new_tokens=args.max_new_tokens)
        print(f"  [stage {stage_name}] test acc={test_metrics['accuracy']:.4f}  n={test_metrics['n']}")

    return {
        "stage": stage_name,
        "stage_idx": stage_idx,
        "sizes": asdict(stage),
        "history": history,
        "test_metrics": test_metrics,
        "best_val_loss": best_val if best_val != math.inf else None,
        "stage_wall_minutes": (time.time() - t0) / 60,
    }


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", required=True,
                    help="Ordered list of stages; choices: " + ", ".join(sorted(STAGE_NAMES)))
    ap.add_argument("--ahri_data", default="data/ahri")
    ap.add_argument("--llm", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--enable_lora", action="store_true", default=True)
    ap.add_argument("--no-enable_lora", dest="enable_lora", action="store_false")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--ecg_max_train", type=int, default=50_000,
                    help="Cap on the ECG-QA-CoT training set (full is 159k, 22-68x heavier per example than the other stages).")
    ap.add_argument("--ecg_max_eval", type=int, default=5_000,
                    help="Cap on ECG val/test for fast eval at end of stage.")
    args = ap.parse_args()

    for s in args.stages:
        if s not in STAGE_NAMES:
            raise SystemExit(f"Unknown stage '{s}'. Choices: {sorted(STAGE_NAMES)}")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] llm={args.llm} stages={args.stages} device={device}")
    model = OpenTSLMSP(llm_id=args.llm, device=str(device))
    if args.enable_lora:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    # Resume support: if a `_final.pt` from an earlier stage exists in the
    # output dir, we load it and skip that stage.
    completed = []
    for s in args.stages:
        ck = out_dir / f"{s}_final.pt"
        if ck.exists():
            load_checkpoint(model, ck)
            completed.append(s)
        else:
            break
    if completed:
        print(f"[resume] Skipping completed stages: {completed}")

    overall = {
        "llm": args.llm,
        "stages": [],
        "args": vars(args),
    }
    summary_path = out_dir / "curriculum_summary.json"
    if summary_path.exists():
        try:
            overall = json.loads(summary_path.read_text())
        except Exception:
            pass
    t_start = time.time()

    for idx, stage_name in enumerate(args.stages):
        if stage_name in completed:
            continue
        result = run_stage(model, stage_name, idx, args, out_dir)
        overall["stages"].append(result)
        overall["total_wall_minutes"] = (time.time() - t_start) / 60
        summary_path.write_text(json.dumps(overall, indent=2, default=str))
        print(f"\n[curriculum] wrote {summary_path}  (after stage {stage_name})")

    print(f"\n[done] all stages complete; summary: {summary_path}")


if __name__ == "__main__":
    main()
