"""
Graders: parse model output, return correctness + task-specific metrics.

Parse failures count as incorrect (paper Section 4.3).
"""

from __future__ import annotations

import re
from typing import Sequence

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_first_number(text: str) -> float | None:
    m = _NUM_RE.search(text)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_all_numbers(text: str) -> list[float]:
    return [float(m.group(0)) for m in _NUM_RE.finditer(text)]


def match_label(text: str, labels: Sequence[str]) -> str | None:
    """First-match label parsing. We lowercase both sides and require word
    boundaries so that 'low' doesn't match 'slow'."""
    t = text.lower()
    for lbl in labels:
        if re.search(rf"\b{re.escape(lbl.lower())}\b", t):
            return lbl
    return None


def grade_classification(prediction: str, gt_label: str, labels: Sequence[str]) -> dict:
    pred = match_label(prediction, labels)
    return {"correct": pred == gt_label, "pred_label": pred}


def grade_regression(prediction: str, gt_value: float, tol_bands: Sequence[float]) -> dict:
    pred = extract_first_number(prediction)
    if pred is None:
        return {
            "correct": False,
            "pred": None,
            "mae": float("inf"),
            **{f"hit_{tol}": False for tol in tol_bands},
        }
    err = abs(pred - gt_value)
    hits = {f"hit_{tol}": bool(err <= tol) for tol in tol_bands}
    # "correct" uses the tightest band
    return {"correct": hits[f"hit_{tol_bands[0]}"], "pred": pred, "mae": err, **hits}


def grade_multi_regression(
    prediction: str, gt_values: Sequence[float], tol_bands: Sequence[float]
) -> dict:
    """Multiple numeric outputs (e.g. Task 4.2 start+end frequency).
    Per-estimate tolerance bands. 'correct' requires all estimates within
    the tightest band."""
    preds = extract_all_numbers(prediction)[: len(gt_values)]
    out = {"preds": preds}
    if len(preds) < len(gt_values):
        out["correct"] = False
        out["mae"] = float("inf")
        for tol in tol_bands:
            out[f"hit_{tol}"] = False
        return out
    errs = [abs(p - g) for p, g in zip(preds, gt_values)]
    out["mae"] = float(sum(errs) / len(errs))
    for tol in tol_bands:
        out[f"hit_{tol}"] = bool(all(e <= tol for e in errs))
    out["correct"] = out[f"hit_{tol_bands[0]}"]
    return out


def grade_multilabel(prediction: str, gt_set: set[str], labels: Sequence[str]) -> dict:
    """Multi-label: a label is predicted if it appears as a substring in the
    response. Correct iff the predicted set equals the ground-truth set."""
    t = prediction.lower()
    pred_set = {lbl for lbl in labels if re.search(rf"\b{re.escape(lbl.lower())}\b", t)}
    return {
        "correct": pred_set == gt_set,
        "pred_set": sorted(pred_set),
        "jaccard": (len(pred_set & gt_set) / len(pred_set | gt_set)) if (pred_set | gt_set) else 1.0,
    }
