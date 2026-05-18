"""Multi-task schedules and mixture."""

from __future__ import annotations

import numpy as np

from opentslm.ahri.multitask import SCHEDULES, MultiTaskMixture
from opentslm.ahri.tasks import list_task_ids


def test_simultaneous_always_active():
    s = SCHEDULES["simultaneous"](["1.1", "1.2", "3.1"])
    assert s.active_tasks(0, 100) == ["1.1", "1.2", "3.1"]
    assert s.active_tasks(99, 100) == ["1.1", "1.2", "3.1"]


def test_easy_to_hard_introduces_in_tier_order():
    # tasks at every tier
    tids = ["1.1", "2.1", "3.1", "4.1", "5.1"]
    s = SCHEDULES["easy_to_hard"](tids)
    # at progress=0, only tier 1 is active
    active_start = s.active_tasks(0, 100)
    assert active_start == ["1.1"]
    # at progress=0.5, tiers 1-3
    active_mid = s.active_tasks(50, 100)
    assert "1.1" in active_mid and "2.1" in active_mid and "3.1" in active_mid
    assert "5.1" not in active_mid
    # at progress=1.0, all
    active_end = s.active_tasks(100, 100)
    assert set(active_end) == set(tids)


def test_hard_to_easy_introduces_top_tier_first():
    tids = ["1.1", "5.1"]
    s = SCHEDULES["hard_to_easy"](tids)
    active_start = s.active_tasks(0, 100)
    assert "5.1" in active_start and "1.1" not in active_start
    active_end = s.active_tasks(100, 100)
    assert "1.1" in active_end and "5.1" in active_end


def test_random_introduction_monotonic():
    tids = list_task_ids()
    s = SCHEDULES["random_introduction"](tids)
    sizes = [len(s.active_tasks(step, 1000)) for step in range(0, 1001, 100)]
    # non-decreasing
    for a, b in zip(sizes, sizes[1:]):
        assert a <= b


def test_random_introduction_deterministic_with_seed():
    tids = list_task_ids()
    from opentslm.ahri.multitask import RandomIntroduction
    s1 = RandomIntroduction("a", tids, seed=42)
    s2 = RandomIntroduction("b", tids, seed=42)
    assert s1._order == s2._order


def test_mixture_sample_batch(tiny_data_root):
    mix = MultiTaskMixture(tiny_data_root, ["1.2", "3.1"], split="train")
    b = mix.sample_batch("1.2", 3)
    assert b["signals"].shape == (3, 1, 1024)
    assert len(b["prompts"]) == 3
    b2 = mix.sample_batch("3.1", 3)
    assert b2["signals"].shape == (3, 2, 1024)
