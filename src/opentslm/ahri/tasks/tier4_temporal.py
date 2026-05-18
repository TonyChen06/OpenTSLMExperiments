"""
Tier 4: Temporal reasoning — track changes over time.

4.1 Frequency change direction  (3-class: increase / decrease / constant)
4.2 Frequency change measurement (regression x2: start, end freq)
4.3 Envelope classification     (5-class)
4.4 Segment labeling            (per-segment classification, 3 segments)
4.5 Inter-event regularity      (binary: regular / irregular)

4.5 has no held-out parameter; the paper notes "structural diversity
across regular/irregular; no single parameter."
"""

from __future__ import annotations

import numpy as np

from opentslm.ahri import waveforms as wf
from opentslm.ahri.graders import (
    grade_classification,
    grade_multi_regression,
    grade_regression,
    match_label,
)
from opentslm.ahri.prompts import format_single
from opentslm.ahri.tasks.base import AhriTask, Example, Split
from opentslm.ahri.tasks.tier1_detection import _sample_param


# ----------------------------------------------------------------------
# 4.1 Frequency change direction
# ----------------------------------------------------------------------

class T4_1_FreqChangeDirection(AhriTask):
    task_id = "4.1"
    tier = 4
    question = "Does the frequency increase, decrease, or stay constant over time?"
    output_format = "classification"
    labels = ("increase", "decrease", "constant")
    heldout_freq = (10.0, 15.0)
    full_range = (1.0, 20.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        cls = self.labels[int(rng.integers(0, 3))]
        amp = float(rng.uniform(0.5, 3.0))
        held = False

        if cls == "constant":
            f, h = _sample_param(rng, self.full_range, self.heldout_freq, split)
            sig = wf.sinusoid(f, amp, float(rng.uniform(0, 2 * np.pi)))
            held = h
            params = {"class": "constant", "f": f}
        else:
            for _ in range(100):
                f0, h0 = _sample_param(rng, self.full_range, self.heldout_freq, split)
                f1, h1 = _sample_param(rng, self.full_range, self.heldout_freq, split)
                if cls == "increase" and f1 - f0 >= 1.0:
                    break
                if cls == "decrease" and f0 - f1 >= 1.0:
                    break
            held = bool(h0 or h1)
            sig = wf.chirp(amp, f0, f1)
            params = {"class": cls, "f0": f0, "f1": f1}

        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=cls,
            gt=cls,
            held=held,
            params=params,
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


# ----------------------------------------------------------------------
# 4.2 Frequency change measurement
# ----------------------------------------------------------------------

class T4_2_FreqChangeMeasurement(AhriTask):
    task_id = "4.2"
    tier = 4
    question = "What are the starting and ending frequencies of this signal in Hz?"
    output_format = "regression"
    tol_bands = (0.5, 1.0, 2.0)
    heldout_freq = (10.0, 15.0)
    full_range = (1.0, 20.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        for _ in range(100):
            f0, h0 = _sample_param(rng, self.full_range, self.heldout_freq, split)
            f1, h1 = _sample_param(rng, self.full_range, self.heldout_freq, split)
            if abs(f1 - f0) >= 1.0:
                break
        amp = float(rng.uniform(0.5, 3.0))
        sig = wf.chirp(amp, f0, f1)
        held = bool(h0 or h1)
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=f"{f0:.2f}, {f1:.2f}",
            gt=[f0, f1],
            held=held,
            params={"f0": f0, "f1": f1, "amp": amp},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_multi_regression(prediction, example.gt, self.tol_bands)


# ----------------------------------------------------------------------
# 4.3 Envelope classification
# ----------------------------------------------------------------------

class T4_3_EnvelopeClassification(AhriTask):
    task_id = "4.3"
    tier = 4
    question = "What is the shape of this signal's amplitude envelope?"
    output_format = "classification"
    labels = ("constant", "increasing", "decreasing", "decaying", "oscillating")
    heldout_carrier = (8.0, 12.0)
    carrier_range = (5.0, 15.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        fc, held = _sample_param(rng, self.carrier_range, self.heldout_carrier, split)
        cls = self.labels[int(rng.integers(0, 5))]
        t = wf.TIME
        if cls == "constant":
            A = float(rng.uniform(0.5, 3.0))
            env = np.full_like(t, A)
        elif cls == "increasing":
            A0 = float(rng.uniform(0.5, 1.5))
            alpha = float(rng.choice([0.1, 0.2, 0.4, 0.6]))
            env = A0 + alpha * t
        elif cls == "decreasing":
            A0 = float(rng.uniform(1.5, 3.0))
            alpha = float(rng.choice([0.1, 0.2, 0.4, 0.6]))
            env = A0 - alpha * t
        elif cls == "decaying":
            A0 = float(rng.uniform(1.0, 3.0))
            lam = float(rng.choice([0.3, 0.5, 1.0, 2.0]))
            env = A0 * np.exp(-lam * t)
        else:  # oscillating
            A0 = float(rng.uniform(0.5, 2.0))
            beta = float(rng.choice([0.3, 0.5, 0.7]))
            fm = float(rng.choice([0.5, 1.0, 2.0]))
            env = A0 * (1.0 + beta * np.sin(2 * np.pi * fm * t))

        carrier = np.sin(2 * np.pi * fc * t)
        sig = wf.amplitude_modulate(carrier, env)
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=cls,
            gt=cls,
            held=held,
            params={"class": cls, "carrier_freq": fc},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


# ----------------------------------------------------------------------
# 4.4 Segment labeling
# ----------------------------------------------------------------------

class T4_4_SegmentLabeling(AhriTask):
    task_id = "4.4"
    tier = 4
    question = (
        "What happens in each segment? "
        "Reply in the format 'Segment 1: X. Segment 2: Y. Segment 3: Z.' "
        "where each label is one of: silence, low, medium, high, upward, downward."
    )
    output_format = "per_segment"
    seg_labels = ("silence", "low", "medium", "high", "upward", "downward")
    heldout_boundary = (2.0, 3.0)
    boundary_range = (0.5, 4.5)

    def _segment_signal(self, rng: np.random.Generator, kind: str, mask: np.ndarray) -> np.ndarray:
        if kind == "silence":
            full = wf.gaussian_noise(0.05, rng)
        elif kind == "low":
            f = float(rng.uniform(1.0, 7.0))
            full = wf.sinusoid(f, float(rng.uniform(0.5, 2.0)), float(rng.uniform(0, 2 * np.pi)))
        elif kind == "medium":
            f = float(rng.uniform(8.0, 13.0))
            full = wf.sinusoid(f, float(rng.uniform(0.5, 2.0)), float(rng.uniform(0, 2 * np.pi)))
        elif kind == "high":
            f = float(rng.uniform(14.0, 20.0))
            full = wf.sinusoid(f, float(rng.uniform(0.5, 2.0)), float(rng.uniform(0, 2 * np.pi)))
        elif kind == "upward":
            full = wf.linear_ramp(float(rng.uniform(0.2, 2.0)), float(rng.uniform(-0.5, 0.5)))
        else:  # downward
            full = wf.linear_ramp(-float(rng.uniform(0.2, 2.0)), float(rng.uniform(-0.5, 0.5)))
        return np.where(mask, full, 0.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        t = wf.TIME
        t1, held = _sample_param(rng, self.boundary_range, self.heldout_boundary, split)
        t2_lo = max(t1 + 1.0, self.boundary_range[0] + 1.0)
        t2_hi = min(self.boundary_range[1], wf.DURATION - 0.12)
        if t2_lo >= t2_hi:
            t2 = min(t2_hi, t1 + 1.0)
        else:
            t2 = float(rng.uniform(t2_lo, t2_hi))

        a = self.seg_labels[int(rng.integers(0, len(self.seg_labels)))]
        while True:
            b = self.seg_labels[int(rng.integers(0, len(self.seg_labels)))]
            if b != a:
                break
        while True:
            c = self.seg_labels[int(rng.integers(0, len(self.seg_labels)))]
            if c != b:
                break

        m1 = t < t1
        m2 = (t >= t1) & (t < t2)
        m3 = t >= t2

        sig = (
            self._segment_signal(rng, a, m1)
            + self._segment_signal(rng, b, m2)
            + self._segment_signal(rng, c, m3)
        )

        answer = f"Segment 1: {a}. Segment 2: {b}. Segment 3: {c}."
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=answer,
            gt=[a, b, c],
            held=held,
            params={"t1": t1, "t2": t2, "labels": [a, b, c]},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        gt_labels = example.gt
        chunks = prediction.lower().split("segment")
        per_seg = []
        for i in range(1, 4):
            if i >= len(chunks):
                per_seg.append(None)
            else:
                per_seg.append(match_label(chunks[i], self.seg_labels))
        hits = [pred == gt for pred, gt in zip(per_seg, gt_labels)]
        return {
            "correct": all(hits),
            "per_seg_accuracy": sum(hits) / len(hits),
            "pred_labels": per_seg,
        }


# ----------------------------------------------------------------------
# 4.5 Regularity
# ----------------------------------------------------------------------

class T4_5_Regularity(AhriTask):
    task_id = "4.5"
    tier = 4
    question = "Are the events in this signal regularly or irregularly spaced?"
    output_format = "classification"
    labels = ("regular", "irregular")

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        regular = bool(rng.integers(0, 2))
        n = int(rng.choice([3, 4, 5, 6, 7, 8]))
        sigma = float(rng.choice([0.03, 0.05, 0.08]))
        A = float(rng.uniform(0.8, 2.5))

        if regular:
            base_dt = float(rng.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]))
            t_start = float(rng.uniform(0.2, 1.0))
            centers = [t_start + i * base_dt for i in range(n)]
            centers = [c for c in centers if c < wf.DURATION - 0.1]
            if len(centers) < 2:
                centers = list(np.linspace(0.3, wf.DURATION - 0.3, n))
        else:
            attempts = 0
            while True:
                attempts += 1
                gaps = rng.uniform(0.2, 1.0, size=n - 1)
                t0 = float(rng.uniform(0.2, 1.0))
                centers = [t0]
                for g in gaps:
                    centers.append(centers[-1] + float(g))
                centers = [c for c in centers if c < wf.DURATION - 0.1]
                if len(centers) >= 2:
                    diffs = np.diff(centers)
                    cv = np.std(diffs) / max(np.mean(diffs), 1e-6)
                    if cv > 0.3:
                        break
                if attempts > 50:
                    break

        sig = np.zeros(wf.N, dtype=np.float64)
        for c in centers:
            sig = sig + wf.gaussian_pulse(A, c, sigma)

        cls = "regular" if regular else "irregular"
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=cls,
            gt=cls,
            held=False,  # no held-out region for 4.5
            params={"class": cls, "n": n, "centers": centers},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


TIER4_TASKS = [
    T4_1_FreqChangeDirection,
    T4_2_FreqChangeMeasurement,
    T4_3_EnvelopeClassification,
    T4_4_SegmentLabeling,
    T4_5_Regularity,
]
