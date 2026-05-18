"""
Ahri: Ascending Harmonic Reasoning Instruction.

A controlled evaluation framework for Time-Series Language Models.
21 synthetic tasks in 5 tiers, parametrically generated signals, exact ground
truth, held-out parameter regions per task.

See paper: 'Physics of Time-Series Language Models: Part 1, The Capability
Frontier'.
"""

from opentslm.ahri.tasks import TASK_REGISTRY, get_task, list_task_ids
from opentslm.ahri.waveforms import FS, N, DURATION

__all__ = ["TASK_REGISTRY", "get_task", "list_task_ids", "FS", "N", "DURATION"]
