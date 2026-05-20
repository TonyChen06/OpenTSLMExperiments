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
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from opentslm.ahri.distributed import (
    barrier,
    cleanup_distributed,
    is_main_rank,
    rank,
    setup_distributed,
    world_size,
)
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


def _log(msg: str):
    """Print only on the main rank (no-op on other ranks under DDP)."""
    if is_main_rank():
        print(msg, flush=True)


class _LossModule(nn.Module):
    """Wrap OpenTSLMSP so DDP can intercept the loss computation.

    DDP only synchronises gradients for work done inside the wrapped
    module's ``forward``. OpenTSLMSP exposes ``compute_loss``/``generate``
    rather than ``forward``, so we route the training-step loss through
    ``forward`` here. Eval/generate still call the inner model directly
    (rank 0 only), bypassing DDP.
    """

    def __init__(self, model: OpenTSLMSP):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.compute_loss(batch)


# --------------------------------------------------------------------
# AHRI pretrain curriculum ordering
# --------------------------------------------------------------------

# Default fraction of each tier's examples that "leak" forward past their own
# block (a small, sustained replay of earlier tiers); the rest stay in-block.
DEFAULT_CURRICULUM_LEAK = 0.15


def _ahri_tier_per_index(concat_dataset) -> np.ndarray:
    """Tier (1-5) for each global index of the AHRI pretrain ConcatDataset.

    ahri_all_tasks builds a ConcatDataset of one AhriQADataset per task in
    list_task_ids() order, so tiers fall in contiguous blocks we read off
    each sub-dataset's task."""
    tiers: list[int] = []
    for d in concat_dataset.datasets:
        tiers.extend([d.task.tier] * len(d))
    return np.asarray(tiers, dtype=np.int64)


class CurriculumSampler(Sampler):
    """Single-pass, block-structured easy->hard ordering for AHRI pretraining.

    The epoch is split into one block per tier in ascending order (tier 1
    first ... tier 5 last). Each example stays inside its own tier's block
    with probability (1 - leak); with probability `leak` it jumps forward
    uniformly into the remaining timeline. So each block is dominated by its
    own tier, but a small, sustained tail of every earlier tier keeps showing
    up afterward (soft replay) -- "most of tier 1 first, then a bit of tier 1
    from then on". Later tiers never leak backward.

    Every example appears exactly once per epoch (total unchanged); subtasks
    within a tier are mixed by the random in-block positions.

    Shards across ranks in lockstep (each rank takes every num_replicas-th
    element of the shared order) so both GPUs move through the curriculum
    together. Reseeded per epoch via set_epoch."""

    def __init__(self, tiers: np.ndarray, num_replicas: int = 1, rank: int = 0,
                 base_seed: int = 0, leak: float = DEFAULT_CURRICULUM_LEAK):
        self.tiers = tiers
        self.N = len(tiers)
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.base_seed = base_seed
        self.leak = float(leak)
        self.num_tiers = int(tiers.max())
        self.epoch = 0
        self.per_rank = math.ceil(self.N / self.num_replicas)
        self.total = self.per_rank * self.num_replicas
        # per-example block boundaries in [0, 1)
        b = tiers - 1
        self._block_lo = (b / self.num_tiers).astype(np.float64)
        self._block_hi = ((b + 1) / self.num_tiers).astype(np.float64)
        self._width = 1.0 / self.num_tiers

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _ordered_indices(self) -> np.ndarray:
        rng = np.random.default_rng(self.base_seed + self.epoch)
        # in-block position (most examples stay here)
        in_block = self._block_lo + rng.random(self.N) * self._width
        # forward-leaked position, uniform in [block_hi, 1)
        span = 1.0 - self._block_hi
        leaked = self._block_hi + rng.random(self.N) * span
        # leak forward with prob `leak`, but only where there is room ahead
        do_leak = (rng.random(self.N) < self.leak) & (span > 1e-9)
        pos = np.where(do_leak, leaked, in_block)
        order = np.argsort(pos, kind="stable")
        if self.total > self.N:  # pad so every rank gets an equal count
            order = np.concatenate([order, order[: self.total - self.N]])
        return order

    def __iter__(self):
        order = self._ordered_indices()
        shard = order[self.rank :: self.num_replicas]
        return iter(int(i) for i in shard)

    def __len__(self) -> int:
        return self.per_rank


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


def build_stage_loaders(stage_name: str, eos_token: str, args, collate, distributed: bool = False) -> tuple[DataLoader, DistributedSampler | None, DataLoader | None, DataLoader | None, Stage]:
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

    # Training loader. The AHRI stage can use a blended easy->hard curriculum
    # ordering (single pass, total unchanged, subtasks mixed, earlier tiers
    # replayed); other stages use plain shuffle (or a DistributedSampler under
    # DDP). The curriculum sampler shards across ranks itself.
    use_curriculum = stage_name == "ahri" and getattr(args, "ahri_curriculum", "none") != "none"
    if use_curriculum:
        train_sampler = CurriculumSampler(_ahri_tier_per_index(train),
                                          num_replicas=world_size(), rank=rank(),
                                          base_seed=args.seed,
                                          leak=getattr(args, "curriculum_leak", DEFAULT_CURRICULUM_LEAK))
    else:
        train_sampler = DistributedSampler(train, shuffle=True) if distributed else None
    train_loader = DataLoader(train, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              collate_fn=collate, num_workers=args.num_workers)
    # Val is sharded too under DDP (shuffled so a capped slice is representative);
    # eval reduces across ranks. Test stays unsharded (rank-0 generate eval).
    val_sampler = DistributedSampler(val, shuffle=True) if (distributed and val) else None
    val_loader = DataLoader(val, batch_size=args.batch_size,
                            shuffle=False, sampler=val_sampler,
                            collate_fn=collate, num_workers=args.num_workers) if val else None
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=args.num_workers) if test else None

    return train_loader, train_sampler, val_loader, val_sampler, test_loader, Stage(
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
def evaluate_loss(model: OpenTSLMSP, loader: DataLoader, max_batches: int = 0,
                  distributed: bool = False) -> float:
    """Mean loss over (a capped slice of) the loader.

    Under DDP the loader is sharded per rank; we sum the per-rank totals
    with an all-reduce so every rank ends up with the same global mean
    (this also acts as the cross-rank sync point, so no separate barrier
    is needed and no rank sits idle waiting on rank 0)."""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        loss = model.compute_loss(batch)
        total += loss.item() * len(batch)
        n += len(batch)
    if distributed:
        import torch.distributed as dist
        t = torch.tensor([total, n], device=model.device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, n = t[0].item(), t[1].item()
    return total / max(n, 1)


@torch.no_grad()
def evaluate_accuracy(model: OpenTSLMSP, loader: DataLoader, max_new_tokens: int = 32,
                      max_batches: int = 0) -> dict:
    model.eval()
    correct = 0
    total = 0
    samples = []
    for i, batch in enumerate(tqdm(loader, desc="eval", leave=False, mininterval=2.0)):
        if max_batches and i >= max_batches:
            break
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

def run_stage(train_module: nn.Module, inner: OpenTSLMSP, stage_name: str, stage_idx: int,
              args, out_dir: Path, distributed: bool = False) -> dict:
    _log(f"\n{'='*70}\n[stage {stage_idx + 1}] {stage_name}\n{'='*70}")
    collate = lambda b: extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=4)

    train_loader, train_sampler, val_loader, val_sampler, test_loader, stage = build_stage_loaders(
        stage_name, inner.get_eos_token(), args, collate, distributed=distributed
    )
    _log(f"[stage {stage_name}] sizes: {stage}  (steps/epoch/rank={len(train_loader)})")

    total_steps = args.epochs * len(train_loader)
    trainable = [p for p in train_module.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * 0.05), total_steps)

    best_val = math.inf
    history: list[dict] = []
    t0 = time.time()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        train_module.train()
        running, n_running = 0.0, 0
        pbar = tqdm(train_loader, desc=f"  ep {epoch}", leave=False, mininterval=2.0,
                    disable=not is_main_rank())
        for batch in pbar:
            loss = train_module(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n_running += 1
            global_step += 1
            if is_main_rank() and global_step % args.log_every == 0:
                pbar.set_postfix(loss=f"{running / max(n_running, 1):.3f}", step=global_step)

        train_loss = running / max(n_running, 1)
        entry = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None:
            # Distributed eval reduces across ranks (the all-reduce also syncs
            # ranks), so val_loss is identical everywhere and the "improved?"
            # decision below is consistent; only rank 0 writes the file.
            val_loss = evaluate_loss(inner, val_loader,
                                     max_batches=args.eval_max_batches, distributed=distributed)
            entry["val_loss"] = val_loss
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                if is_main_rank():
                    save_checkpoint(inner, out_dir / f"{stage_name}_best.pt",
                                     extra={"stage": stage_name, "epoch": epoch, "val_loss": val_loss})
                entry["saved_best"] = True
        history.append(entry)
        elapsed = (time.time() - t0) / 60
        _log(f"  [stage {stage_name} epoch {epoch}/{args.epochs}] train={train_loss:.4f}"
              + (f" val={entry.get('val_loss'):.4f}" if 'val_loss' in entry else "")
              + f" elapsed={elapsed:.1f}m")

    # Always save end-of-stage checkpoint (used by the next stage), rank 0 only.
    if is_main_rank():
        save_checkpoint(inner, out_dir / f"{stage_name}_final.pt",
                         extra={"stage": stage_name, "epoch": args.epochs})
    barrier()

    # If a best-by-val checkpoint was written, every rank reloads it so weights
    # stay consistent across ranks before the next stage (the barrier above
    # guarantees rank 0 finished writing it).
    best_path = out_dir / f"{stage_name}_best.pt"
    if best_path.exists():
        load_checkpoint(inner, best_path)

    # Eval test set (generate-based; rank 0 only, capped to avoid long idle).
    test_metrics = {}
    if is_main_rank() and test_loader is not None:
        test_metrics = evaluate_accuracy(inner, test_loader, max_new_tokens=args.max_new_tokens,
                                          max_batches=args.eval_max_batches)
        _log(f"  [stage {stage_name}] test acc={test_metrics['accuracy']:.4f}  n={test_metrics['n']}")
    barrier()

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
    ap.add_argument("--eval_max_batches", type=int, default=0,
                    help="Cap per-stage val/test eval to this many batches per rank (0 = full). "
                         "Keeps per-epoch eval cheap during pretraining and avoids long idles under DDP.")
    ap.add_argument("--ahri_curriculum", choices=["none", "easy2hard"], default="none",
                    help="AHRI pretrain ordering: 'none' = plain shuffle; 'easy2hard' = block-structured "
                         "ascending-by-tier curriculum (single pass, subtasks mixed within a tier, "
                         "earlier tiers leak forward as soft replay).")
    ap.add_argument("--curriculum_leak", type=float, default=DEFAULT_CURRICULUM_LEAK,
                    help="Fraction of each tier's examples that leak forward past their block "
                         "(soft replay). 0 = hard tier blocks; ~0.15 = a little leaky.")
    args = ap.parse_args()

    # DDP is opt-in via torchrun (sets WORLD_SIZE); plain `python ...` stays
    # single-process and behaves exactly as before.
    distributed = world_size() > 1

    for s in args.stages:
        if s not in STAGE_NAMES:
            raise SystemExit(f"Unknown stage '{s}'. Choices: {sorted(STAGE_NAMES)}")

    device = setup_distributed() if distributed else torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    if is_main_rank():
        out_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    _log(f"[setup] llm={args.llm} stages={args.stages} device={device} "
         f"world_size={world_size()} distributed={distributed}")
    model = OpenTSLMSP(llm_id=args.llm, device=str(device))
    if args.enable_lora:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    # Resume support: if a `_final.pt` from an earlier stage exists in the
    # output dir, we load it and skip that stage. All ranks load the same
    # checkpoint so they start consistent.
    completed = []
    for s in args.stages:
        ck = out_dir / f"{s}_final.pt"
        if ck.exists():
            load_checkpoint(model, ck)
            completed.append(s)
        else:
            break
    if completed:
        _log(f"[resume] Skipping completed stages: {completed}")

    # Wrap for DDP. The loss-module routes compute_loss through forward so
    # DDP can sync gradients; eval/generate still call `model` directly.
    # find_unused_parameters=False: every trainable param (LoRA + encoder +
    # projector) participates in each step — verified via the 2-proc smoke,
    # which reported no unused params. Avoids an extra autograd traversal.
    train_module: nn.Module = _LossModule(model)
    if distributed:
        train_module = DDP(train_module,
                           device_ids=[device.index] if device.type == "cuda" else None,
                           find_unused_parameters=False)

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
        result = run_stage(train_module, model, stage_name, idx, args, out_dir, distributed=distributed)
        overall["stages"].append(result)
        overall["total_wall_minutes"] = (time.time() - t_start) / 60
        if is_main_rank():
            summary_path.write_text(json.dumps(overall, indent=2, default=str))
            _log(f"\n[curriculum] wrote {summary_path}  (after stage {stage_name})")

    _log(f"\n[done] all stages complete; summary: {summary_path}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
