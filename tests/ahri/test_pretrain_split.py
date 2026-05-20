"""
Tests for the `pretrain` split added for the transfer experiment.

Verifies:
  - all 21 tasks support the pretrain split
  - pretrain samples from the FULL parameter range (held fraction matches
    test, not train); train still excludes held-out
  - self-grade still works on pretrain samples (no regression)
  - generator can write a pretrain.parquet
  - AhriQADataset can read it back
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opentslm.ahri.tasks import TASK_REGISTRY, get_task, list_task_ids


@pytest.mark.parametrize("task_id", list_task_ids())
def test_pretrain_split_samples(task_id: str):
    """Sampler returns valid examples for split='pretrain'."""
    task = get_task(task_id)
    rng = np.random.default_rng(0)
    for _ in range(5):
        ex = task.sample(rng, "pretrain")
        assert len(ex.signals) == task.num_signals
        for s in ex.signals:
            assert s.shape == (1024,)
            assert np.all(np.isfinite(s))
        # self-grade
        r = task.grade(ex.answer, ex)
        assert r["correct"]


@pytest.mark.parametrize("task_id", list_task_ids())
def test_pretrain_includes_heldout_examples(task_id: str):
    """Pretrain samples from the full range, so some examples should be
    held=True (except 4.5 which has no held-out region)."""
    if task_id == "4.5":
        return  # no held-out region by design
    task = get_task(task_id)
    rng = np.random.default_rng(0)
    held = sum(1 for _ in range(200) if task.sample(rng, "pretrain").held)
    assert held >= 1, f"{task_id}: pretrain never produced held=True"


def test_pretrain_held_distribution_matches_test():
    """Pretrain and test should both sample from the full parameter range,
    so their held-fraction must agree closely."""
    for tid in list_task_ids():
        task = get_task(tid)
        n = 300
        rng = np.random.default_rng(42)
        pre = sum(1 for _ in range(n) if task.sample(rng, "pretrain").held) / n
        rng = np.random.default_rng(42)
        tst = sum(1 for _ in range(n) if task.sample(rng, "test").held) / n
        # identical RNG seed and identical sampling logic -> equal
        assert abs(pre - tst) < 0.01, f"{tid}: pretrain={pre:.3f} test={tst:.3f}"


def test_train_still_excludes_heldout_after_refactor():
    """Make sure the pretrain refactor didn't accidentally let train see held-out."""
    for tid in list_task_ids():
        task = get_task(tid)
        rng = np.random.default_rng(0)
        for _ in range(50):
            ex = task.sample(rng, "train")
            assert ex.held is False, f"{tid}: train produced held=True"


def test_generator_supports_pretrain_split(tmp_path):
    from opentslm.ahri.generate import SPLIT_SIZES, generate_task
    # build a tiny pretrain shard for one task
    original_sizes = dict(SPLIT_SIZES)
    try:
        SPLIT_SIZES.clear()
        SPLIT_SIZES["pretrain"] = 8
        generate_task("1.2", tmp_path, base_seed=0, overwrite=True)
    finally:
        SPLIT_SIZES.clear()
        SPLIT_SIZES.update(original_sizes)

    p = tmp_path / "1.2" / "pretrain.parquet"
    assert p.exists()


def test_adapter_reads_pretrain_split(tmp_path):
    """End-to-end: generate a pretrain shard, then load via AhriQADataset."""
    from opentslm.ahri.generate import SPLIT_SIZES, generate_task
    from opentslm.ahri.opentslm_adapter import AhriQADataset

    original_sizes = dict(SPLIT_SIZES)
    try:
        SPLIT_SIZES.clear()
        SPLIT_SIZES["pretrain"] = 6
        generate_task("1.2", tmp_path, base_seed=0, overwrite=True)
    finally:
        SPLIT_SIZES.clear()
        SPLIT_SIZES.update(original_sizes)

    ds = AhriQADataset(tmp_path, "1.2", "pretrain", EOS_TOKEN="")
    assert len(ds) == 6
    ex = ds[0]
    assert ex["answer"]
    assert ex["time_series"][0].shape == (1024,)
