"""
Multi-task dataset + curriculum schedulers for RQ2.

A `MultiTaskMixture` wraps N AhriParquetDataset(train) and yields batches
according to a `Schedule` that decides which task to sample from at each
training step. The trainer evaluates on each task's val + test_in periodically.

Curricula (paper Section 6, Experiment 6.2):
  - simultaneous: uniform sampling from all tasks
  - easy_to_hard: Tier 1 first 20%, add Tier 2 at 20%, ..., Tier 5 at 80%
  - hard_to_easy: reversed
  - random_introduction: one task at a time, evenly spaced
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from opentslm.ahri.dataset import AhriParquetDataset
from opentslm.ahri.tasks import get_task


class MultiTaskMixture:
    """Not a torch Dataset — the trainer iterates over step indices and pulls
    a fresh batch from the active task at each step. This decouples 'epoch'
    from the multi-task setting, which makes curriculum scheduling clean."""

    def __init__(self, root: str, task_ids: Sequence[str], split: str = "train", seed: int = 0):
        self.task_ids = list(task_ids)
        self.datasets = {tid: AhriParquetDataset(root, tid, split) for tid in self.task_ids}
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, task_id: str, batch_size: int) -> dict:
        ds = self.datasets[task_id]
        idx = self.rng.integers(0, len(ds), size=batch_size)
        items = [ds[int(i)] for i in idx]
        return {
            "signals": torch.stack([it["signals"] for it in items]),
            "prompts": [it["prompt"] for it in items],
            "answers": [it["answer"] for it in items],
        }


# ---------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------

@dataclass
class Schedule:
    """A schedule decides, at training step `step`, which task to draw from."""
    name: str
    task_ids: list[str]

    def active_tasks(self, step: int, total_steps: int) -> list[str]:
        raise NotImplementedError

    def pick(self, step: int, total_steps: int, rng: np.random.Generator) -> str:
        active = self.active_tasks(step, total_steps)
        return active[int(rng.integers(0, len(active)))]


class Simultaneous(Schedule):
    def active_tasks(self, step, total_steps): return self.task_ids


class EasyToHard(Schedule):
    """Tier 1 active in first 20%, Tier 2 added at 20%, ..., Tier 5 added at 80%."""
    def active_tasks(self, step, total_steps):
        progress = step / max(total_steps, 1)
        tier_thresholds = [0.0, 0.2, 0.4, 0.6, 0.8]  # tier 1..5 introduce times
        active = []
        for tid in self.task_ids:
            tier = get_task(tid).tier
            if progress >= tier_thresholds[tier - 1]:
                active.append(tid)
        return active or [self.task_ids[0]]


class HardToEasy(Schedule):
    def active_tasks(self, step, total_steps):
        progress = step / max(total_steps, 1)
        tier_thresholds = [0.8, 0.6, 0.4, 0.2, 0.0]  # introduce tier 5 first
        active = []
        for tid in self.task_ids:
            tier = get_task(tid).tier
            if progress >= tier_thresholds[tier - 1]:
                active.append(tid)
        return active or [self.task_ids[0]]


class RandomIntroduction(Schedule):
    """Tasks introduced one at a time in a random order, evenly spaced
    across training. After introduction, a task stays in the active set."""
    def __init__(self, name: str, task_ids: Sequence[str], seed: int = 0):
        super().__init__(name=name, task_ids=list(task_ids))
        rng = np.random.default_rng(seed)
        order = list(self.task_ids)
        rng.shuffle(order)
        self._order = order

    def active_tasks(self, step, total_steps):
        progress = step / max(total_steps, 1)
        n = len(self._order)
        n_active = min(n, max(1, int(math.ceil(progress * n))))
        return self._order[:n_active]


SCHEDULES: dict[str, Callable[[Sequence[str]], Schedule]] = {
    "simultaneous": lambda tids: Simultaneous("simultaneous", list(tids)),
    "easy_to_hard": lambda tids: EasyToHard("easy_to_hard", list(tids)),
    "hard_to_easy": lambda tids: HardToEasy("hard_to_easy", list(tids)),
    "random_introduction": lambda tids: RandomIntroduction("random_introduction", list(tids)),
}
