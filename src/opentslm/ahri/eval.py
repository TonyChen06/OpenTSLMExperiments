"""
Eval harness for Ahri.

Given a model, a task, and one or more split datasets, run greedy generation
and aggregate per-example grades into a metrics dict with bootstrap 95% CIs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from opentslm.ahri.dataset import AhriParquetDataset
from opentslm.ahri.tasks import get_task
from opentslm.model.llm.PhysicsTSLM import PhysicsTSLM


def _collate(batch: list[dict]) -> dict:
    return {
        "signals": torch.stack([b["signals"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "held": [b["held"] for b in batch],
    }


@dataclass
class EvalResult:
    task_id: str
    split: str
    n: int
    accuracy: float                       # over all examples (test = in_dist + held)
    accuracy_ci: tuple[float, float]
    # Per-subset breakdown. For the `test` split, the eval harness partitions
    # examples by the `held` flag and reports accuracy on each subset
    # separately. `n_in` + `n_held` == n. For train/val these are still
    # populated (typically n_held == 0).
    n_in: int
    accuracy_in: float
    accuracy_in_ci: tuple[float, float]
    n_held: int
    accuracy_held: float
    accuracy_held_ci: tuple[float, float]
    per_example: list[dict]
    extras: dict   # aggregated regression metrics, multi-segment accuracy, etc.


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05, rng: np.random.Generator | None = None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = values[idx].mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


@torch.no_grad()
def evaluate(
    model: PhysicsTSLM,
    dataset: AhriParquetDataset,
    batch_size: int = 16,
    max_new_tokens: int = 32,
    device: str | torch.device = "cuda",
    desc: str | None = None,
) -> EvalResult:
    model.eval()
    task = get_task(dataset.task_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    per_example: list[dict] = []
    desc = desc or f"eval {dataset.task_id}/{dataset.split}"

    for batch in tqdm(loader, desc=desc, leave=False, mininterval=2.0):
        sigs = batch["signals"].to(device)
        tok = model.tokenize(batch["prompts"], None, device=device)
        gen = model.generate(sigs, **tok, max_new_tokens=max_new_tokens)
        # `inputs_embeds` path: generate returns only the new tokens
        decoded = model.tokenizer.batch_decode(gen, skip_special_tokens=True)
        for ans_pred, ans_gt, held in zip(decoded, batch["answers"], batch["held"]):
            # Build a minimal Example with the right gt; grade() only uses .gt
            class _E:
                pass
            ex = _E()
            ex.gt = _decode_gt_for_grading(task, ans_gt)
            result = task.grade(ans_pred, ex)
            result["prediction"] = ans_pred
            result["answer_gt"] = ans_gt
            result["held"] = bool(held)
            per_example.append(result)

    arr = np.array([float(r["correct"]) for r in per_example])
    in_arr = np.array([float(r["correct"]) for r in per_example if not r["held"]])
    held_arr = np.array([float(r["correct"]) for r in per_example if r["held"]])

    acc = float(arr.mean()) if len(arr) else 0.0
    acc_in = float(in_arr.mean()) if len(in_arr) else 0.0
    acc_held = float(held_arr.mean()) if len(held_arr) else 0.0

    extras = _aggregate_extras(task, per_example)
    return EvalResult(
        task_id=dataset.task_id,
        split=dataset.split,
        n=len(per_example),
        accuracy=acc,
        accuracy_ci=bootstrap_ci(arr),
        n_in=len(in_arr),
        accuracy_in=acc_in,
        accuracy_in_ci=bootstrap_ci(in_arr) if len(in_arr) else (0.0, 0.0),
        n_held=len(held_arr),
        accuracy_held=acc_held,
        accuracy_held_ci=bootstrap_ci(held_arr) if len(held_arr) else (0.0, 0.0),
        per_example=per_example,
        extras=extras,
    )


def _decode_gt_for_grading(task, answer_string: str):
    """Reconstruct task.gt from the stored answer string. For most tasks
    answer_string IS the gt; for regression / multi-output we parse it back
    out of the string."""
    fmt = task.output_format
    if fmt == "regression" and not isinstance(task, type) and getattr(task, "task_id", None) == "4.2":
        # two numbers separated by comma
        parts = [float(x.strip()) for x in answer_string.split(",")]
        return parts
    if fmt == "regression":
        try:
            return float(answer_string)
        except ValueError:
            return None
    if fmt == "multilabel":
        if answer_string == "none":
            return set()
        return {x.strip() for x in answer_string.split(",")}
    if fmt == "anomaly":
        if answer_string == "no":
            return {"has_anomaly": False, "idx": None}
        # "yes, cycle K"
        import re
        m = re.search(r"cycle\s+(\d+)", answer_string.lower())
        return {"has_anomaly": True, "idx": int(m.group(1)) if m else None}
    if fmt == "spectral":
        # "N components: f1, f2, ..."
        import re
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", answer_string)]
        if not nums:
            return {"K": 0, "freqs": []}
        K = int(nums[0])
        return {"K": K, "freqs": sorted(nums[1 : 1 + K])}
    if fmt == "per_segment":
        # "Segment 1: X. Segment 2: Y. Segment 3: Z."
        import re
        matches = re.findall(r"segment\s+\d+:\s*(\w+)", answer_string.lower())
        return matches
    # classification: gt = answer string
    return answer_string


def _aggregate_extras(task, per_example: list[dict]) -> dict:
    extras: dict = {}
    fmt = task.output_format
    if fmt == "regression":
        mae_vals = [r["mae"] for r in per_example if "mae" in r and np.isfinite(r["mae"])]
        if mae_vals:
            extras["mae_mean"] = float(np.mean(mae_vals))
            extras["mae_median"] = float(np.median(mae_vals))
        for tol in getattr(task, "tol_bands", ()):
            key = f"hit_{tol}"
            extras[f"acc_within_{tol}"] = float(np.mean([float(r.get(key, False)) for r in per_example]))
    if fmt == "per_segment":
        per_seg = [r.get("per_seg_accuracy", 0.0) for r in per_example]
        extras["per_seg_acc"] = float(np.mean(per_seg))
    if fmt == "multilabel":
        jacc = [r.get("jaccard", 0.0) for r in per_example]
        extras["jaccard"] = float(np.mean(jacc))
    if fmt == "spectral":
        K_correct = [r.get("K_correct", False) for r in per_example]
        extras["K_correct_rate"] = float(np.mean(K_correct))
    return extras


def evaluate_test(
    model: PhysicsTSLM,
    root: str,
    task_id: str,
    batch_size: int = 16,
    device: str | torch.device = "cuda",
) -> EvalResult:
    """Run a single pass over the `test` parquet; the returned EvalResult
    has both `accuracy_in` and `accuracy_held` populated."""
    ds = AhriParquetDataset(root, task_id, "test")
    return evaluate(model, ds, batch_size=batch_size, device=device)
