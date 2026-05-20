"""
Adapter: Ahri parquet -> OpenTSLM-SP / curriculum_learning trainer format.

Lets you train OpenTSLM-SP (TransformerCNNEncoder + MLPProjector + LLM) on
Ahri data without rewriting the trainer. Each Ahri example becomes a
PromptWithAnswer-shaped dict matching the keys OpenTSLM-SP's `compute_loss`
and `pad_and_apply_batch` expect:

    {
        "answer":         "<answer string>",
        "pre_prompt":     "",                 # Ahri has no pre-prompt
        "post_prompt":    "Question: <q>\\nAnswer:",
        "time_series":    [np.ndarray, ...],  # 1 or 2 signals
        "time_series_text": ["Signal:", ...] # minimal — no stats in text
    }

Used together with `opentslm.time_series_datasets.util.extend_time_series_to_match_patch_size_and_aggregate`
as the DataLoader collate_fn.

This adapter is *the* contract between the Ahri benchmark and the
OpenTSLM-SP transfer experiment. If OpenTSLM-SP's training data format
changes, update here only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import ConcatDataset, Dataset

from opentslm.ahri.tasks import get_task, list_task_ids
from opentslm.prompt.prompt_with_answer import PromptWithAnswer
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt


Split = Literal["train", "val", "test"]


class AhriQADataset(Dataset):
    """One Ahri task's parquet shard, exposed as the dict format OpenTSLM-SP wants."""

    def __init__(self, root: str | Path, task_id: str, split: Split, EOS_TOKEN: str = ""):
        self.task_id = task_id
        self.split = split
        self.EOS_TOKEN = EOS_TOKEN
        self.task = get_task(task_id)

        path = Path(root) / task_id / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} — did you download from HF or run `python -m opentslm.ahri.generate`?"
            )

        table = pq.read_table(path)
        # signals: list[list[list[float]]] -> (N, num_signals, signal_len)
        self._signals = [np.asarray(s, dtype=np.float32) for s in table.column("signals").to_pylist()]
        self._answers = table.column("answer").to_pylist()
        self._held = table.column("held").to_pylist()

        # cache the prompt strings (same for every example in a task)
        self._pre_prompt = ""
        self._post_prompt = f"Question: {self.task.question}\nAnswer:"

    def __len__(self) -> int:
        return len(self._signals)

    def __getitem__(self, idx: int) -> dict:
        signals = self._signals[idx]  # (num_signals, 1024) float32
        answer = self._answers[idx]
        if self.EOS_TOKEN and not answer.endswith(self.EOS_TOKEN):
            answer = answer + self.EOS_TOKEN

        if signals.shape[0] == 1:
            ts_prompts = [TextTimeSeriesPrompt("Signal:", signals[0])]
        else:
            ts_prompts = [
                TextTimeSeriesPrompt(f"Signal {i + 1}:", signals[i])
                for i in range(signals.shape[0])
            ]

        return PromptWithAnswer(
            TextPrompt(self._pre_prompt),
            ts_prompts,
            TextPrompt(self._post_prompt.strip()),
            answer.strip(),
        ).to_dict()


def ahri_all_tasks(root: str | Path, split: Split, EOS_TOKEN: str = "") -> ConcatDataset:
    """ConcatDataset of all 21 Ahri tasks for a given split. Use this as the
    pretraining corpus."""
    return ConcatDataset([
        AhriQADataset(root, tid, split, EOS_TOKEN) for tid in list_task_ids()
    ])


def ahri_subset(root: str | Path, task_ids: list[str], split: Split, EOS_TOKEN: str = "") -> ConcatDataset:
    """ConcatDataset of a chosen subset of Ahri tasks."""
    return ConcatDataset([
        AhriQADataset(root, tid, split, EOS_TOKEN) for tid in task_ids
    ])


__all__ = ["AhriQADataset", "ahri_all_tasks", "ahri_subset"]
