"""
Tier 2: Measurement — extract a quantitative value.

2.1 Frequency estimation       (regression, +-0.5 / +-1 / +-2 Hz)
2.2 Peak counting              (10-class, K in {1..10})
2.3 Event localization         (regression, +-0.05 / +-0.1 / +-0.25 / +-0.5 s)
2.4 Change-point localization  (regression, +-0.1 / +-0.25 / +-0.5 s)
"""

from __future__ import annotations

import numpy as np

from opentslm.ahri import waveforms as wf
from opentslm.ahri.graders import grade_classification, grade_regression
from opentslm.ahri.prompts import format_single
from opentslm.ahri.tasks.base import AhriTask, Example, Split
from opentslm.ahri.tasks.tier1_detection import _sample_int, _sample_param


# ----------------------------------------------------------------------
# 2.1 Frequency estimation
# ----------------------------------------------------------------------

class T2_1_FrequencyEstimation(AhriTask):
    task_id = "2.1"
    tier = 2
    question = "What is the frequency of this signal in Hz?"
    output_format = "regression"
    tol_bands = (0.5, 1.0, 2.0)
    heldout_freq = (10.0, 15.0)
    full_range = (1.0, 20.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        freq, held = _sample_param(rng, self.full_range, self.heldout_freq, split)
        amp = float(rng.uniform(0.5, 3.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        sig = wf.sinusoid(freq, amp, phase)
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=f"{freq:.2f}",
            gt=freq,
            held=held,
            params={"freq": freq, "amp": amp, "phase": phase},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_regression(prediction, example.gt, self.tol_bands)


# ----------------------------------------------------------------------
# 2.2 Peak counting
# ----------------------------------------------------------------------

class T2_2_PeakCounting(AhriTask):
    task_id = "2.2"
    tier = 2
    question = "How many distinct peaks are in this signal?"
    output_format = "classification"
    labels = tuple(str(i) for i in range(1, 11))
    heldout_k = (6, 7)
    k_range = list(range(1, 11))

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        K, held = _sample_int(rng, self.k_range, list(self.heldout_k), split)

        # place K non-overlapping pulses with min 0.3 s separation
        centers: list[float] = []
        for _ in range(500):
            if len(centers) == K:
                break
            cand = float(rng.uniform(0.3, wf.DURATION - 0.3))
            if all(abs(cand - c) >= 0.3 for c in centers):
                centers.append(cand)
        if len(centers) < K:
            centers = list(np.linspace(0.3, wf.DURATION - 0.3, K))

        sig = np.zeros(wf.N, dtype=np.float64)
        for c in centers:
            A = float(rng.uniform(0.8, 2.5))
            s = float(rng.choice([0.05, 0.10, 0.15]))
            sig = sig + wf.gaussian_pulse(A, c, s)

        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=str(K),
            gt=str(K),
            held=held,
            params={"K": K, "centers": centers},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


# ----------------------------------------------------------------------
# 2.3 Event localization
# ----------------------------------------------------------------------

class T2_3_EventLocalization(AhriTask):
    task_id = "2.3"
    tier = 2
    question = "At what time (in seconds) does the event occur?"
    output_format = "regression"
    tol_bands = (0.05, 0.1, 0.25, 0.5)
    heldout_t0 = (2.0, 3.0)
    full_range = (0.5, 4.5)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        t0, held = _sample_param(rng, self.full_range, self.heldout_t0, split)
        A = float(rng.uniform(0.5, 3.0))
        s = float(rng.uniform(0.05, 0.5))
        sig = wf.gaussian_pulse(A, t0, s)
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=f"{t0:.3f}",
            gt=t0,
            held=held,
            params={"t0": t0, "A": A, "sigma": s},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_regression(prediction, example.gt, self.tol_bands)


# ----------------------------------------------------------------------
# 2.4 Change-point localization
# ----------------------------------------------------------------------

class T2_4_ChangePoint(AhriTask):
    task_id = "2.4"
    tier = 2
    question = "At what time does the signal's frequency change?"
    output_format = "regression"
    tol_bands = (0.1, 0.25, 0.5)
    heldout_tcp = (2.0, 3.0)
    full_range = (1.0, 4.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        tcp, held = _sample_param(rng, self.full_range, self.heldout_tcp, split)
        f1 = float(rng.uniform(1.0, 20.0))
        for _ in range(100):
            f2 = float(rng.uniform(1.0, 20.0))
            if abs(f1 - f2) >= 1.0:
                break
        amp = float(rng.uniform(0.5, 3.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        s1 = wf.sinusoid(f1, amp, phase)
        s2 = wf.sinusoid(f2, amp, phase)
        sig = wf.temporal_concat(s1, s2, tcp)
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=f"{tcp:.3f}",
            gt=tcp,
            held=held,
            params={"tcp": tcp, "f1": f1, "f2": f2, "amp": amp, "phase": phase},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_regression(prediction, example.gt, self.tol_bands)


TIER2_TASKS = [T2_1_FrequencyEstimation, T2_2_PeakCounting, T2_3_EventLocalization, T2_4_ChangePoint]
