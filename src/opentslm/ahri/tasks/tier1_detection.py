"""
Tier 1: Detection — perceive a categorical property of one signal.

1.1 Trend direction (3-class: upward / downward / none)
1.2 Frequency band  (3-class: low 1-7 / medium 8-13 / high 14-20 Hz)
1.3 Periodicity     (binary: periodic / aperiodic)
1.4 Event presence  (binary: yes / no)

Split semantics:
- "train" / "val": parameters sampled excluding the held-out region.
- "test":          parameters sampled from the FULL range. The `held` flag
                   marks examples whose parameter landed in the held-out
                   region. Eval reports accuracy on the full set, on the
                   in-distribution subset, and on the held subset.

Class stratification is class-first sampling: pick a class label uniformly,
then sample parameters consistent with that label. This guarantees balanced
classes (paper Section 3.2 specifies this for several tasks).
"""

from __future__ import annotations

import numpy as np

from opentslm.ahri import waveforms as wf
from opentslm.ahri.graders import grade_classification
from opentslm.ahri.prompts import format_single
from opentslm.ahri.tasks.base import AhriTask, Example, Split


def _sample_param(rng: np.random.Generator, full_range: tuple[float, float], heldout: tuple[float, float] | None, split: Split) -> tuple[float, bool]:
    """Sample one parameter consistent with the split's distribution.
    Returns (value, held_flag)."""
    if split == "test":
        value = float(rng.uniform(*full_range))
        held = wf.in_heldout(value, heldout) if heldout is not None else False
        return value, held
    # train / val: exclude held-out
    value = wf.sample_uniform_excluding(rng, full_range, heldout)
    return value, False


def _sample_int(rng: np.random.Generator, values: list[int], heldout: list[int] | None, split: Split) -> tuple[int, bool]:
    if split == "test":
        v = int(rng.choice(values))
        held = (heldout is not None) and v in set(heldout)
        return v, held
    v = wf.sample_int_excluding(rng, values, heldout)
    return v, False


# ----------------------------------------------------------------------
# 1.1 Trend direction
# ----------------------------------------------------------------------

class T1_1_TrendDirection(AhriTask):
    task_id = "1.1"
    tier = 1
    question = "Does this signal have an upward trend, a downward trend, or no trend?"
    output_format = "classification"
    labels = ("upward", "downward", "none")
    heldout_freq = (6.0, 8.0)
    freq_range = (3.0, 10.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        freq, held = _sample_param(rng, self.freq_range, self.heldout_freq, split)
        cls_idx = int(rng.integers(0, 3))
        cls = self.labels[cls_idx]
        amp = float(rng.uniform(0.5, 1.5))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        carrier = wf.sinusoid(freq, amp, phase)

        if cls == "none":
            slope = 0.0
            sig = carrier
        else:
            slope_mag = float(rng.uniform(0.2, 2.0))  # avoid near-zero so direction is unambiguous
            slope = slope_mag if cls == "upward" else -slope_mag
            sig = wf.additive(carrier, wf.linear_ramp(slope))

        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=cls,
            gt=cls,
            held=held,
            params={"freq": freq, "amp": amp, "phase": phase, "slope": slope, "class": cls},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


# ----------------------------------------------------------------------
# 1.2 Frequency band  (stratified: class first, then f in that band)
# ----------------------------------------------------------------------

class T1_2_FrequencyBand(AhriTask):
    task_id = "1.2"
    tier = 1
    question = "Is this signal low-frequency (1-7 Hz), medium-frequency (8-13 Hz), or high-frequency (14-20 Hz)?"
    output_format = "classification"
    labels = ("low", "medium", "high")
    bands = {"low": (1.0, 7.0), "medium": (8.0, 13.0), "high": (14.0, 20.0)}
    heldout_freq = (10.0, 15.0)
    full_range = (1.0, 20.0)

    def _sample_freq_in_band(self, rng: np.random.Generator, cls: str, split: Split) -> tuple[float, bool]:
        band = self.bands[cls]
        # intersect band with the held-out exclusion (for train/val) or use the
        # full band (for test, which spans held-out).
        if split == "test":
            f = float(rng.uniform(*band))
            held = wf.in_heldout(f, self.heldout_freq)
            return f, held
        # train/val: sample within band excluding the held-out interval
        for _ in range(1000):
            f = float(rng.uniform(*band))
            if not wf.in_heldout(f, self.heldout_freq):
                return f, False
        raise RuntimeError(f"Could not sample band {cls} excluding {self.heldout_freq}")

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        cls = self.labels[int(rng.integers(0, 3))]
        freq, held = self._sample_freq_in_band(rng, cls, split)
        amp = float(rng.uniform(0.5, 3.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        sig = wf.sinusoid(freq, amp, phase)
        return Example(
            signals=[sig.astype(np.float32)],
            prompt=format_single(self.question),
            answer=cls,
            gt=cls,
            held=held,
            params={"freq": freq, "amp": amp, "phase": phase, "class": cls},
        )

    def grade(self, prediction: str, example: Example) -> dict:
        return grade_classification(prediction, example.gt, self.labels)


# ----------------------------------------------------------------------
# 1.3 Periodicity
# ----------------------------------------------------------------------

class T1_3_Periodicity(AhriTask):
    task_id = "1.3"
    tier = 1
    question = "Is this signal periodic or aperiodic?"
    output_format = "classification"
    labels = ("periodic", "aperiodic")
    heldout_freq = (10.0, 15.0)
    full_range = (1.0, 20.0)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        is_periodic = bool(rng.integers(0, 2))
        if is_periodic:
            freq, held = _sample_param(rng, self.full_range, self.heldout_freq, split)
            amp = float(rng.uniform(0.5, 3.0))
            phase = float(rng.uniform(0.0, 2 * np.pi))
            alpha = float(rng.choice([0.0, 0.3, 0.5]))
            sig = wf.sinusoid(freq, amp, phase, harmonic_strength=alpha)
            params = {"freq": freq, "amp": amp, "phase": phase, "alpha": alpha, "class": "periodic"}
        else:
            # Aperiodic class isn't tied to a held-out parameter (no `f`).
            # We treat aperiodic test examples as in-distribution.
            kind = rng.choice(["ramp", "pulse", "rwalk"])
            if kind == "ramp":
                m = float(rng.uniform(-2.0, 2.0))
                b = float(rng.uniform(-1.0, 1.0))
                sig = wf.linear_ramp(m, b)
            elif kind == "pulse":
                A = float(rng.uniform(0.5, 3.0))
                t0 = float(rng.uniform(0.5, 4.5))
                s = float(rng.uniform(0.05, 0.5))
                sig = wf.gaussian_pulse(A, t0, s)
            else:
                sig = wf.random_walk(0.1, rng)
            held = False
            params = {"kind": kind, "class": "aperiodic"}

        cls = "periodic" if is_periodic else "aperiodic"
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
# 1.4 Event presence
# ----------------------------------------------------------------------

class T1_4_EventPresence(AhriTask):
    task_id = "1.4"
    tier = 1
    question = "Is there a transient event (a pulse or spike) in this signal?"
    output_format = "classification"
    labels = ("yes", "no")
    heldout_t0 = (2.0, 3.0)
    t0_range = (0.5, 4.5)

    def sample(self, rng: np.random.Generator, split: Split) -> Example:
        has_event = bool(rng.integers(0, 2))
        bg_sigma = float(rng.choice([0.05, 0.1, 0.15, 0.2]))
        bg = wf.gaussian_noise(bg_sigma, rng)
        held = False

        if has_event:
            t0, held = _sample_param(rng, self.t0_range, self.heldout_t0, split)
            min_amp = max(0.5, 5.0 * bg_sigma)
            A = float(rng.uniform(min_amp, 3.0))
            s = float(rng.uniform(0.05, 0.5))
            sig = bg + wf.gaussian_pulse(A, t0, s)
            params = {"has_event": True, "t0": t0, "A": A, "sigma": s, "bg_sigma": bg_sigma}
        else:
            sig = bg
            params = {"has_event": False, "bg_sigma": bg_sigma}

        cls = "yes" if has_event else "no"
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


TIER1_TASKS = [T1_1_TrendDirection, T1_2_FrequencyBand, T1_3_Periodicity, T1_4_EventPresence]
