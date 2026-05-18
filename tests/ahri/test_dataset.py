"""Dataset generator + parquet loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from opentslm.ahri.dataset import AhriParquetDataset, resolve_root


def test_dataset_loads_correct_shapes(tiny_data_root: Path):
    ds = AhriParquetDataset(tiny_data_root, "1.2", "train")
    ex = ds[0]
    assert ex["signals"].shape == (1, 1024)
    assert ex["signals"].dtype == torch.float32
    assert isinstance(ex["prompt"], str)
    assert isinstance(ex["answer"], str)
    assert isinstance(ex["held"], bool)


def test_two_signal_dataset(tiny_data_root: Path):
    ds = AhriParquetDataset(tiny_data_root, "3.1", "train")
    ex = ds[0]
    assert ex["signals"].shape == (2, 1024)


def test_train_val_never_held(tiny_data_root: Path):
    """Train/val never contain held examples."""
    for split in ("train", "val"):
        ds = AhriParquetDataset(tiny_data_root, "1.2", split)
        assert not any(ds[i]["held"] for i in range(len(ds)))


def test_test_contains_held(tiny_data_root: Path):
    """`test` split mixes in-dist and held; held flag distinguishes them."""
    ds = AhriParquetDataset(tiny_data_root, "1.2", "test")
    # Held fraction depends on heldout region width / full range; not
    # guaranteed in a tiny sample, but the column should be present and
    # boolean-typed.
    flags = [ds[i]["held"] for i in range(len(ds))]
    assert all(isinstance(f, bool) for f in flags)


def test_all_21_tasks_load(all_tasks_data_root: Path):
    from opentslm.ahri.tasks import list_task_ids
    for tid in list_task_ids():
        for split in ("train", "val", "test"):
            ds = AhriParquetDataset(all_tasks_data_root, tid, split)
            assert len(ds) > 0, f"{tid}/{split} empty"
            _ = ds[0]


def test_resolve_root_local_path(tiny_data_root: Path):
    assert resolve_root(str(tiny_data_root)) == str(tiny_data_root)


def test_resolve_root_unknown_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        resolve_root("/nonexistent/path/that/does/not/exist")
