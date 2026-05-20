"""
Tests for the Ahri -> OpenTSLM-SP data adapter.

These verify that AhriQADataset emits the exact dict shape OpenTSLM-SP's
trainer (and curriculum_learning.py) expect, with arrays correctly shaped.
No GPU and no model loading — these run fast in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opentslm.ahri.opentslm_adapter import (
    AhriQADataset,
    ahri_all_tasks,
    ahri_subset,
)
from opentslm.ahri.tasks import list_task_ids


EXPECTED_KEYS = {"answer", "post_prompt", "pre_prompt", "time_series", "time_series_text"}


def test_single_signal_task_shape(all_tasks_data_root: Path):
    ds = AhriQADataset(all_tasks_data_root, "1.2", "train", EOS_TOKEN="")
    assert len(ds) > 0
    ex = ds[0]
    assert set(ex.keys()) == EXPECTED_KEYS
    assert isinstance(ex["answer"], str)
    assert isinstance(ex["pre_prompt"], str)
    assert isinstance(ex["post_prompt"], str)
    assert isinstance(ex["time_series"], list)
    assert isinstance(ex["time_series_text"], list)
    assert len(ex["time_series"]) == 1
    assert len(ex["time_series_text"]) == 1
    assert isinstance(ex["time_series"][0], np.ndarray)
    assert ex["time_series"][0].shape == (1024,)


def test_two_signal_task_shape(all_tasks_data_root: Path):
    ds = AhriQADataset(all_tasks_data_root, "3.1", "train", EOS_TOKEN="")
    ex = ds[0]
    assert len(ex["time_series"]) == 2
    assert len(ex["time_series_text"]) == 2
    assert ex["time_series_text"] == ["Signal 1:", "Signal 2:"]
    for ts in ex["time_series"]:
        assert ts.shape == (1024,)


def test_eos_token_appended_to_answer(all_tasks_data_root: Path):
    ds = AhriQADataset(all_tasks_data_root, "1.2", "train", EOS_TOKEN="<eos>")
    assert ds[0]["answer"].endswith("<eos>")


def test_question_appears_in_post_prompt(all_tasks_data_root: Path):
    """post_prompt must contain the task's question, not just \"Answer:\"."""
    from opentslm.ahri.tasks import get_task
    for tid in ["1.2", "2.1", "4.4"]:
        ds = AhriQADataset(all_tasks_data_root, tid, "train")
        ex = ds[0]
        assert get_task(tid).question in ex["post_prompt"]


def test_pre_prompt_is_empty(all_tasks_data_root: Path):
    """Ahri intentionally has no pre-prompt (no stats in text)."""
    ds = AhriQADataset(all_tasks_data_root, "1.2", "train")
    assert ds[0]["pre_prompt"] == ""


def test_split_test_distinguishes_held(all_tasks_data_root: Path):
    """`test` split should contain a mix of held and in-distribution; the
    adapter doesn't expose held flag (OpenTSLM doesn't use it), but the
    parquet file itself does. This test just ensures we load `test` cleanly."""
    ds = AhriQADataset(all_tasks_data_root, "1.2", "test")
    assert len(ds) > 0
    assert ds[0]["answer"]


def test_ahri_all_tasks_concats_21(all_tasks_data_root: Path):
    """ahri_all_tasks() should return a ConcatDataset of all 21 tasks."""
    ds = ahri_all_tasks(all_tasks_data_root, "train")
    expected_n = sum(
        len(AhriQADataset(all_tasks_data_root, tid, "train")) for tid in list_task_ids()
    )
    assert len(ds) == expected_n


def test_ahri_subset_works(all_tasks_data_root: Path):
    ds = ahri_subset(all_tasks_data_root, ["1.1", "1.2"], "train")
    n1 = len(AhriQADataset(all_tasks_data_root, "1.1", "train"))
    n2 = len(AhriQADataset(all_tasks_data_root, "1.2", "train"))
    assert len(ds) == n1 + n2


def test_missing_split_raises_clearly(tmp_path):
    """A missing parquet should raise FileNotFoundError with a hint."""
    with pytest.raises(FileNotFoundError) as exc:
        AhriQADataset(tmp_path, "1.2", "train")
    assert "generate" in str(exc.value)


def test_adapter_can_collate_with_opentslm_util(all_tasks_data_root: Path):
    """End-to-end: build a small batch via the OpenTSLM collate fn — same one
    the trainer uses. Verifies the dict shape is accepted by upstream code."""
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )
    ds = AhriQADataset(all_tasks_data_root, "1.2", "train")
    batch = extend_time_series_to_match_patch_size_and_aggregate(
        [ds[0], ds[1]], patch_size=4
    )
    # collate returns the list of dicts, just with time_series promoted to padded tensors
    assert isinstance(batch, list)
    assert len(batch) == 2
    for ex in batch:
        assert set(ex.keys()) >= EXPECTED_KEYS  # collate may add fields
        # time_series should now be a list of padded tensors
        import torch
        for ts in ex["time_series"]:
            assert isinstance(ts, torch.Tensor)
            # padded length is a multiple of patch_size
            assert ts.size(0) % 4 == 0


def test_two_signal_collate(all_tasks_data_root: Path):
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )
    ds = AhriQADataset(all_tasks_data_root, "3.1", "train")
    batch = extend_time_series_to_match_patch_size_and_aggregate(
        [ds[0], ds[1]], patch_size=4
    )
    for ex in batch:
        assert len(ex["time_series"]) == 2
