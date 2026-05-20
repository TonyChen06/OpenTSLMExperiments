"""
AhriTask base class.

A task knows how to:
 - sample a single training/val/test example (the (signals, prompt, answer, gt) tuple)
 - place its held-out region in parameter space (paper Table 1)
 - grade a model's generated string against the ground truth

Generation is one-shot: tasks are sampled offline by `ahri.generate`, written
to parquet, and re-used across all runs. Graders run online at eval time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Split = Literal["train", "val", "test", "pretrain"]


@dataclass
class Example:
    """A single Ahri example.

    `signals` is a list of float32 arrays. Tier 1, 2, 4, 5 tasks have one signal;
    Tier 3 tasks have two. Held flag marks whether this test example falls in
    the held-out parameter region (only meaningful for test splits).
    """
    signals: list[np.ndarray]
    prompt: str
    answer: str
    gt: Any
    held: bool = False
    params: dict = field(default_factory=dict)


class AhriTask:
    """Abstract base. Each concrete subclass sets `task_id`, `tier`, `question`
    and overrides `sample()` + `grade()`."""

    task_id: str = ""           # e.g. "1.2"
    tier: int = 0
    question: str = ""           # the natural-language question text
    output_format: str = ""      # "classification", "regression", ...
    num_signals: int = 1
    labels: tuple[str, ...] = () # for classification tasks

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        raise NotImplementedError

    def grade(self, prediction: str, example: Example) -> dict:
        """Returns at minimum {'correct': bool}. May include task-specific
        fields ('mae', 'tolerance_hit', etc.). Failure to parse counts as
        incorrect, per paper Section 4.3."""
        raise NotImplementedError

    # ------------- helpers shared across subclasses -------------

    @staticmethod
    def _format_signal_placeholder(n_patches: int) -> str:
        """Where the projected patch tokens are injected at the embedding
        level. The text never sees them — they are replaced before the LLM
        sees the sequence."""
        return "<|signal|>" * n_patches
