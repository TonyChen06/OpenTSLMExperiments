"""
Ahri task registry. `TASK_REGISTRY[task_id]` returns the task class.
"""

from __future__ import annotations

from opentslm.ahri.tasks.base import AhriTask, Example, Split
from opentslm.ahri.tasks.tier1_detection import TIER1_TASKS
from opentslm.ahri.tasks.tier2_measurement import TIER2_TASKS
from opentslm.ahri.tasks.tier3_comparison import TIER3_TASKS
from opentslm.ahri.tasks.tier4_temporal import TIER4_TASKS
from opentslm.ahri.tasks.tier5_compositional import TIER5_TASKS

ALL_TASK_CLASSES = TIER1_TASKS + TIER2_TASKS + TIER3_TASKS + TIER4_TASKS + TIER5_TASKS

TASK_REGISTRY: dict[str, type[AhriTask]] = {cls.task_id: cls for cls in ALL_TASK_CLASSES}


def get_task(task_id: str) -> AhriTask:
    return TASK_REGISTRY[task_id]()


def list_task_ids() -> list[str]:
    return sorted(TASK_REGISTRY.keys(), key=lambda s: tuple(int(x) for x in s.split(".")))


__all__ = ["AhriTask", "Example", "Split", "TASK_REGISTRY", "get_task", "list_task_ids"]
