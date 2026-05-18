"""All 21 tasks: sample + self-grade across all four splits."""

from __future__ import annotations

import numpy as np
import pytest

from opentslm.ahri.tasks import TASK_REGISTRY, list_task_ids


@pytest.mark.parametrize("task_id", list_task_ids())
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_task_sample_self_grade(task_id: str, split: str):
    task = TASK_REGISTRY[task_id]()
    rng = np.random.default_rng(42)
    for _ in range(5):
        ex = task.sample(rng, split)
        assert len(ex.signals) == task.num_signals, f"{task_id} expected {task.num_signals} signals"
        for s in ex.signals:
            assert s.shape == (1024,)
            assert s.dtype == np.float32
            assert np.all(np.isfinite(s)), f"{task_id} produced non-finite signal"
        result = task.grade(ex.answer, ex)
        assert result["correct"], f"{task_id}/{split}: self-grade failed for answer {ex.answer!r}"


@pytest.mark.parametrize("task_id", list_task_ids())
def test_train_val_never_held(task_id: str):
    """Train and val splits exclude the held-out region: every sample
    must have held=False."""
    task = TASK_REGISTRY[task_id]()
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for _ in range(30):
            ex = task.sample(rng, split)
            assert ex.held is False, f"{task_id}/{split} produced held=True"


@pytest.mark.parametrize("task_id", list_task_ids())
def test_test_split_contains_held(task_id: str):
    """The `test` split samples from the full range, so at least some examples
    should land in the held-out region (except 4.5 which has none)."""
    task = TASK_REGISTRY[task_id]()
    rng = np.random.default_rng(0)
    held_count = 0
    for _ in range(200):
        ex = task.sample(rng, "test")
        if ex.held:
            held_count += 1
    if task_id == "4.5":
        assert held_count == 0
    else:
        assert held_count >= 1, f"{task_id}: test split never produced held=True over 200 samples"


@pytest.mark.parametrize("task_id", ["1.1", "1.2", "1.3", "1.4", "4.1", "4.3", "4.5", "5.1"])
def test_class_balance(task_id: str):
    """Classification tasks should have roughly balanced classes in their
    train split (paper specifies 1/K for each class)."""
    task = TASK_REGISTRY[task_id]()
    rng = np.random.default_rng(0)
    counts: dict[str, int] = {}
    for _ in range(400):
        ex = task.sample(rng, "train")
        counts[str(ex.gt)] = counts.get(str(ex.gt), 0) + 1
    # each class should be at least 1/(2K) of samples (allow factor-of-2 slack
    # for stochasticity at n=400)
    k = len(counts)
    threshold = 400 / (2 * k)
    for cls, c in counts.items():
        assert c >= threshold, f"{task_id}: class {cls} only {c}/400 (k={k})"


def test_registry_contains_21_tasks():
    assert len(TASK_REGISTRY) == 21
    expected = {f"{tier}.{i}" for tier in [1] for i in range(1, 5)}
    expected |= {f"2.{i}" for i in range(1, 5)}
    expected |= {f"3.{i}" for i in range(1, 5)}
    expected |= {f"4.{i}" for i in range(1, 6)}
    expected |= {f"5.{i}" for i in range(1, 5)}
    assert set(TASK_REGISTRY.keys()) == expected


def test_list_task_ids_sorted_numerically():
    """1.10 > 1.2 lexicographically but we want numeric ordering. We don't
    have a .10 task currently but the sort logic should still be correct."""
    ids = list_task_ids()
    assert ids == sorted(ids, key=lambda s: tuple(int(x) for x in s.split(".")))
