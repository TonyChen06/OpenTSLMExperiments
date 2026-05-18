"""
Shared pytest fixtures for Ahri tests. Generates a tiny on-disk dataset
once per test session so tests of dataset/model/trainer don't pay generation
cost repeatedly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opentslm.ahri.generate import SPLIT_SIZES, generate_task


@pytest.fixture(scope="session")
def tiny_data_root(tmp_path_factory) -> Path:
    """Generate small shards per task for one task from each tier.
    Smoke-test scope only — paper runs use the full 6k/2k/2k."""
    root = tmp_path_factory.mktemp("ahri_data")
    SPLIT_SIZES["train"] = 8
    SPLIT_SIZES["val"] = 3
    SPLIT_SIZES["test"] = 6
    for tid in ["1.2", "2.1", "3.1", "4.4", "5.2"]:
        generate_task(tid, root, overwrite=True)
    return root


@pytest.fixture(scope="session")
def all_tasks_data_root(tmp_path_factory) -> Path:
    """Tiny shard for ALL 21 tasks (used by the regression test that loads
    every task through the parquet path)."""
    root = tmp_path_factory.mktemp("ahri_all_tasks")
    SPLIT_SIZES["train"] = 4
    SPLIT_SIZES["val"] = 2
    SPLIT_SIZES["test"] = 4
    from opentslm.ahri.tasks import list_task_ids
    for tid in list_task_ids():
        generate_task(tid, root, overwrite=True)
    return root
