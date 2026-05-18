"""
Standardized prompt format. Paper Section 3.4: one template, no CoT, no stats.

Single-signal:
    <|signal|>...<|signal|>
    Question: {q}
    Answer:

Two-signal:
    Signal 1:
    <|signal|>...<|signal|>
    Signal 2:
    <|signal|>...<|signal|>
    Question: {q}
    Answer:

The signal placeholder count equals the number of patch tokens the model
will inject — fixed at N // PATCH_SIZE = 1024 // 32 = 32 for single-signal,
and 32 + 1 (separator) + 32 = 65 for two-signal.
"""

from __future__ import annotations

from opentslm.ahri.waveforms import N

SIGNAL_TOKEN = "<|signal|>"
SEP_TOKEN = "<|signal_sep|>"
PATCH_SIZE = 32
N_PATCHES = N // PATCH_SIZE  # 32


def format_single(question: str) -> str:
    placeholders = SIGNAL_TOKEN * N_PATCHES
    return f"{placeholders}\nQuestion: {question}\nAnswer:"


def format_double(question: str) -> str:
    p = SIGNAL_TOKEN * N_PATCHES
    return f"Signal 1:\n{p}\n{SEP_TOKEN}\nSignal 2:\n{p}\nQuestion: {question}\nAnswer:"


__all__ = ["SIGNAL_TOKEN", "SEP_TOKEN", "PATCH_SIZE", "N_PATCHES", "format_single", "format_double"]
