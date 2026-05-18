"""
Tier 5: Compositional reasoning — combine multiple capabilities.

5.1 2-feature conjunction (binary: oscillation AND trend?)  — stratified: yes/no class first
5.2 3-feature conjunction (multi-label: which of {oscillation, trend, transient}?)
5.3 Anomaly identification (binary + int)
5.4 Spectral decomposition  (int + list of frequencies)
"""

from __future__ import annotations

import re

import numpy as np

from opentslm.ahri import waveforms as wf
from opentslm.ahri.graders import (
    extract_all_numbers,
    grade_classification,
    grade_multilabel,
    match_label,
)
from opentslm.ahri.prompts import format_single
from opentslm.ahri.tasks.base import AhriTask, Example, Split
from opentslm.ahri.tasks.tier1_detection import _sample_param


# ----------------------------------------------------------------------
# 5.1 2-feature conjunction  (stratified yes/no)
# ----------------------------------------------------------------------

class T5_1_TwoFeatureConjunction(AhriTask):
    task_id = "5.1"
    tier = 5
    question = "Does this signal have both an oscillation and a trend?"
    output_format = "classification"
    labels = ("yes", "no")
    heldout_freq = (10.0, 15.0)
    full_range = (1.0, 20.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        freq, held = _sample_param(rng, self.full_range, self.heldout_freq, split)

        # stratify: pick yes/no first, then conditionally pick has_osc/has_trend
        is_yes = bool(rng.integers(0, 2))
        if is_yes:
            has_osc = True
            has_trend = True
        else:
            # "no" class is osc-only, trend-only, or neither — sample uniformly
            kind = int(rng.integers(0, 3))
            has_osc = kind == 0
            has_trend = kind == 1
            # kind == 2 -> neither

        components: list[np.ndarray] = []
        if has_osc:
            amp = float(rng.uniform(0.5, 2.0))
            phase = float(rng.uniform(0.0, 2 * np.pi))
            components.append(wf.sinusoid(freq, amp, phase))
        if has_trend:
            slope_mag = float(rng.uniform(0.2, 1.5))
            slope = slope_mag if rng.integers(0, 2) else -slope_mag
            components.append(wf.linear_ramp(slope))
        if not components:
            components.append(wf.gaussian_noise(0.05, rng))

        sig = wf.additive(*components)
        cls = "yes" if is_yes else "no"
        # held flag is only meaningful when osc is present (freq used).
        # For non-osc "no" examples, mark held=False (no freq draw to be in held).
        if not has_osc:
            held = False
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=cls,
            gt=cls,
            held=held,
            params={"has_osc": has_osc, "has_trend": has_trend, "freq": freq if has_osc else None, "class": cls},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


# ----------------------------------------------------------------------
# 5.2 3-feature conjunction
# ----------------------------------------------------------------------

class T5_2_ThreeFeatureConjunction(AhriTask):
    task_id = "5.2"
    tier = 5
    question = (
        "Which of these features are present in this signal: "
        "oscillation, trend, transient? List all that apply."
    )
    output_format = "multilabel"
    labels = ("oscillation", "trend", "transient")
    heldout_freq = (10.0, 15.0)
    full_range = (1.0, 20.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        freq, held = _sample_param(rng, self.full_range, self.heldout_freq, split)

        has = {
            "oscillation": bool(rng.integers(0, 2)),
            "trend": bool(rng.integers(0, 2)),
            "transient": bool(rng.integers(0, 2)),
        }
        components: list[np.ndarray] = [wf.gaussian_noise(0.03, rng)]
        if has["oscillation"]:
            amp = float(rng.uniform(0.5, 2.0))
            phase = float(rng.uniform(0.0, 2 * np.pi))
            components.append(wf.sinusoid(freq, amp, phase))
        if has["trend"]:
            slope_mag = float(rng.uniform(0.2, 1.5))
            slope = slope_mag if rng.integers(0, 2) else -slope_mag
            components.append(wf.linear_ramp(slope))
        if has["transient"]:
            A = float(rng.uniform(1.5, 3.0))
            t0 = float(rng.uniform(0.5, 4.5))
            s = float(rng.uniform(0.05, 0.3))
            components.append(wf.gaussian_pulse(A, t0, s))

        sig = wf.additive(*components)
        gt_set = {k for k, v in has.items() if v}
        answer = ", ".join(sorted(gt_set)) if gt_set else "none"
        if not has["oscillation"]:
            held = False
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=answer,
            gt=gt_set,
            held=held,
            params={"features": has, "freq": freq if has["oscillation"] else None},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_multilabel(prediction, example.gt, self.labels)


# ----------------------------------------------------------------------
# 5.3 Anomaly identification
# ----------------------------------------------------------------------

class T5_3_AnomalyID(AhriTask):
    task_id = "5.3"
    tier = 5
    question = (
        "This signal contains repeating cycles. "
        "Is there an anomalous cycle? If yes, which cycle number?"
    )
    output_format = "anomaly"
    labels = ("yes", "no")
    heldout_base_freq = (4.0, 7.0)
    base_freq_range = (1.0, 10.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        freq, held = _sample_param(rng, self.base_freq_range, self.heldout_base_freq, split)

        amp = float(rng.uniform(1.0, 2.0))
        n_cycles = int(np.floor(freq * wf.DURATION))
        n_cycles = max(3, min(n_cycles, 12))
        phase = 0.0
        sig = wf.sinusoid(freq, amp, phase)

        has_anomaly = bool(rng.integers(0, 2))
        anomaly_idx = None
        if has_anomaly and n_cycles >= 3:
            anomaly_idx = int(rng.integers(1, n_cycles + 1))
            cycle_period = 1.0 / freq
            t_start = (anomaly_idx - 1) * cycle_period
            t_end = anomaly_idx * cycle_period
            t = wf.TIME
            mask = (t >= t_start) & (t < t_end)
            kind = rng.choice(["amp_spike", "flip"])
            if kind == "amp_spike":
                sig = np.where(mask, 2.5 * sig, sig)
            else:
                sig = np.where(mask, -sig, sig)

        cls = "yes" if has_anomaly else "no"
        answer = f"yes, cycle {anomaly_idx}" if has_anomaly else "no"
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=answer,
            gt={"has_anomaly": has_anomaly, "idx": anomaly_idx, "n_cycles": n_cycles},
            held=held,
            params={"freq": freq, "amp": amp, "n_cycles": n_cycles, "anomaly_idx": anomaly_idx},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        binary_pred = match_label(prediction, self.labels)
        gt = example.gt
        if not gt["has_anomaly"]:
            return {"correct": binary_pred == "no", "binary": binary_pred}
        if binary_pred != "yes":
            return {"correct": False, "binary": binary_pred, "pred_idx": None}
        m = re.search(r"cycle\s+(\d+)", prediction.lower())
        idx = int(m.group(1)) if m else None
        return {"correct": idx == gt["idx"], "binary": binary_pred, "pred_idx": idx}


# ----------------------------------------------------------------------
# 5.4 Spectral decomposition
# ----------------------------------------------------------------------

class T5_4_SpectralDecomposition(AhriTask):
    task_id = "5.4"
    tier = 5
    question = (
        "How many sinusoidal components are present in this signal, "
        "and what are their frequencies in Hz? Reply as: "
        "'N components: f1, f2, ...'."
    )
    output_format = "spectral"
    tol_band = 1.0
    heldout_freq = (10.0, 15.0)
    full_range = (1.0, 20.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        K = int(rng.choice([1, 2, 3, 4]))

        freqs: list[float] = []
        held_flags: list[bool] = []
        for _ in range(K):
            for _attempt in range(50):
                f, h = _sample_param(rng, self.full_range, self.heldout_freq, split)
                if all(abs(f - fi) >= 1.0 for fi in freqs):
                    freqs.append(f)
                    held_flags.append(h)
                    break
        if len(freqs) < K:
            K = len(freqs)
        # sort by frequency, keep held alignment
        order = np.argsort(freqs)
        freqs = [freqs[i] for i in order]
        held_flags = [held_flags[i] for i in order]
        held = any(held_flags)

        components = []
        for f in freqs:
            amp = float(rng.uniform(0.5, 1.5))
            phase = float(rng.uniform(0.0, 2 * np.pi))
            components.append(wf.sinusoid(f, amp, phase))
        sig = wf.additive(*components)

        answer = f"{K} components: " + ", ".join(f"{f:.2f}" for f in freqs)
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=answer,
            gt={"K": K, "freqs": freqs},
            held=held,
            params={"K": K, "freqs": freqs},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        gt = example.gt
        nums = extract_all_numbers(prediction)
        if not nums:
            return {"correct": False, "pred_K": None, "pred_freqs": []}
        pred_K = int(nums[0])
        pred_freqs = sorted(nums[1 : 1 + pred_K]) if len(nums) > 1 else []
        if pred_K != gt["K"] or len(pred_freqs) != gt["K"]:
            return {
                "correct": False,
                "pred_K": pred_K,
                "pred_freqs": pred_freqs,
                "K_correct": pred_K == gt["K"],
            }
        errs = [abs(p - g) for p, g in zip(pred_freqs, gt["freqs"])]
        all_match = all(e <= self.tol_band for e in errs)
        return {
            "correct": all_match,
            "pred_K": pred_K,
            "pred_freqs": pred_freqs,
            "K_correct": True,
            "freq_mae": float(np.mean(errs)),
        }


TIER5_TASKS = [
    T5_1_TwoFeatureConjunction,
    T5_2_ThreeFeatureConjunction,
    T5_3_AnomalyID,
    T5_4_SpectralDecomposition,
]
