"""
Tier 3: Comparison — relate two signals.

3.1 Frequency comparison    (binary: signal 1 / signal 2)
3.2 Amplitude comparison    (binary)
3.3 Event count comparison  (binary)
3.4 Temporal lag detection  (binary)

held=True iff at least one of the two signals' primary parameter falls in
the held-out region (paper Table 1).
"""

from __future__ import annotations

import numpy as np

from opentslm.ahri import waveforms as wf
from opentslm.ahri.graders import grade_classification
from opentslm.ahri.prompts import format_double
from opentslm.ahri.tasks.base import AhriTask, Example, Split


_LABELS = ("signal 1", "signal 2")


def _swap_if(swap: bool, a, b):
    return (b, a) if swap else (a, b)


def _sample_pair(
    rng: np.random.Generator,
    full_range: tuple[float, float],
    heldout: tuple[float, float],
    split: Split,
    min_gap: float = 0.5,
) -> tuple[float, float, bool]:
    """Sample (x_lo, x_hi) with |x1-x2| >= min_gap. For test, sample both
    from the full range; for train/val, exclude held-out for both."""
    for _ in range(200):
        if split in ("test", "pretrain"):
            x1 = float(rng.uniform(*full_range))
            x2 = float(rng.uniform(*full_range))
        else:
            x1 = wf.sample_uniform_excluding(rng, full_range, heldout)
            x2 = wf.sample_uniform_excluding(rng, full_range, heldout)
        if abs(x1 - x2) >= min_gap:
            break
    lo, hi = sorted([x1, x2])
    held = (split in ("test", "pretrain")) and (wf.in_heldout(x1, heldout) or wf.in_heldout(x2, heldout))
    return lo, hi, held


# ----------------------------------------------------------------------
# 3.1 Frequency comparison
# ----------------------------------------------------------------------

class T3_1_FrequencyComparison(AhriTask):
    task_id = "3.1"
    tier = 3
    num_signals = 2
    question = "Which signal has a higher frequency: signal 1 or signal 2?"
    output_format = "classification"
    labels = _LABELS
    heldout_freq = (10.0, 15.0)
    full_range = (1.0, 20.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        f_lo, f_hi, held = _sample_pair(rng, self.full_range, self.heldout_freq, split, min_gap=0.5)
        amp = float(rng.uniform(0.5, 3.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        s_low = wf.sinusoid(f_lo, amp, phase)
        s_high = wf.sinusoid(f_hi, amp, phase)

        higher_is_first = bool(rng.integers(0, 2))
        sig1, sig2 = _swap_if(not higher_is_first, s_high, s_low)
        answer = "signal 1" if higher_is_first else "signal 2"
        return Example(
            signals=[sig1.astype(np.float32), sig2.astype(np.float32)],
            prompt=format_double(self.question),
            answer=answer,
            gt=answer,
            held=held,
            params={"f_low": f_lo, "f_high": f_hi, "higher_is_first": higher_is_first},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


# ----------------------------------------------------------------------
# 3.2 Amplitude comparison
# ----------------------------------------------------------------------

class T3_2_AmplitudeComparison(AhriTask):
    task_id = "3.2"
    tier = 3
    num_signals = 2
    question = "Which signal has a larger amplitude: signal 1 or signal 2?"
    output_format = "classification"
    labels = _LABELS
    heldout_amp = (1.5, 2.0)
    full_range = (0.5, 3.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        A_lo, A_hi, held = _sample_pair(rng, self.full_range, self.heldout_amp, split, min_gap=0.1)
        freq = float(rng.uniform(1.0, 20.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        s_lo = wf.sinusoid(freq, A_lo, phase)
        s_hi = wf.sinusoid(freq, A_hi, phase)

        higher_is_first = bool(rng.integers(0, 2))
        sig1, sig2 = _swap_if(not higher_is_first, s_hi, s_lo)
        answer = "signal 1" if higher_is_first else "signal 2"
        return Example(
            signals=[sig1.astype(np.float32), sig2.astype(np.float32)],
            prompt=format_double(self.question),
            answer=answer,
            gt=answer,
            held=held,
            params={"A_lo": A_lo, "A_hi": A_hi, "freq": freq, "higher_is_first": higher_is_first},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


# ----------------------------------------------------------------------
# 3.3 Event count comparison
# ----------------------------------------------------------------------

class T3_3_CountComparison(AhriTask):
    task_id = "3.3"
    tier = 3
    num_signals = 2
    question = "Which signal has more peaks: signal 1 or signal 2?"
    output_format = "classification"
    labels = _LABELS
    heldout_k = (6, 7)
    k_pool = list(range(1, 11))

    @staticmethod
    def _pulse_train(rng: np.random.Generator, K: int) -> np.ndarray:
        centers: list[float] = []
        for _ in range(500):
            if len(centers) == K:
                break
            c = float(rng.uniform(0.3, wf.DURATION - 0.3))
            if all(abs(c - x) >= 0.3 for x in centers):
                centers.append(c)
        if len(centers) < K:
            centers = list(np.linspace(0.3, wf.DURATION - 0.3, K))
        sig = np.zeros(wf.N, dtype=np.float64)
        for c in centers:
            A = float(rng.uniform(0.8, 2.5))
            s = float(rng.choice([0.05, 0.10, 0.15]))
            sig = sig + wf.gaussian_pulse(A, c, s)
        return sig

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        if split in ("test", "pretrain"):
            K1 = int(rng.choice(self.k_pool))
            for _ in range(50):
                K2 = int(rng.choice(self.k_pool))
                if K2 != K1:
                    break
            held = (K1 in self.heldout_k) or (K2 in self.heldout_k)
        else:
            K1 = wf.sample_int_excluding(rng, self.k_pool, list(self.heldout_k))
            for _ in range(50):
                K2 = wf.sample_int_excluding(rng, self.k_pool, list(self.heldout_k))
                if K2 != K1:
                    break
            held = False

        s1 = self._pulse_train(rng, K1)
        s2 = self._pulse_train(rng, K2)
        answer = "signal 1" if K1 > K2 else "signal 2"
        return Example(
            signals=[s1.astype(np.float32), s2.astype(np.float32)],
            prompt=format_double(self.question),
            answer=answer,
            gt=answer,
            held=held,
            params={"K1": K1, "K2": K2},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


# ----------------------------------------------------------------------
# 3.4 Temporal lag
# ----------------------------------------------------------------------

class T3_4_TemporalLag(AhriTask):
    task_id = "3.4"
    tier = 3
    num_signals = 2
    question = "Which signal leads (occurs earlier): signal 1 or signal 2?"
    output_format = "classification"
    labels = _LABELS
    heldout_freq = (10.0, 15.0)
    full_range = (1.0, 20.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        if split in ("test", "pretrain"):
            freq = float(rng.uniform(*self.full_range))
            held = wf.in_heldout(freq, self.heldout_freq)
        else:
            freq = wf.sample_uniform_excluding(rng, self.full_range, self.heldout_freq)
            held = False
        amp = float(rng.uniform(0.5, 3.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))

        tau_mag = float(rng.uniform(0.05, 1.0))
        sign = int(rng.choice([-1, 1]))
        tau = sign * tau_mag
        s1 = wf.sinusoid(freq, amp, phase)
        s2 = wf.sinusoid(freq, amp, phase - 2 * np.pi * freq * tau)
        answer = "signal 1" if tau > 0 else "signal 2"
        return Example(
            signals=[s1.astype(np.float32), s2.astype(np.float32)],
            prompt=format_double(self.question),
            answer=answer,
            gt=answer,
            held=held,
            params={"freq": freq, "tau": tau, "amp": amp},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


TIER3_TASKS = [T3_1_FrequencyComparison, T3_2_AmplitudeComparison, T3_3_CountComparison, T3_4_TemporalLag]
