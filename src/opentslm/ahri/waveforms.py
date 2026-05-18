"""
Ahri waveform vocabulary: 6 parametric primitives + 3 composition rules.

Every signal in Ahri is built from these. Generators return numpy arrays
sampled at FS=200 Hz for DURATION=5.12 s (N=1024 samples). All primitive
parameter ranges come straight from the paper, Section 3.2.

This module is pure-numpy and has no side effects beyond consuming
a seeded np.random.Generator passed in by the caller. The generator is
threaded through every sampler so a (task_id, split, seed) triple uniquely
determines the realization of every example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

# ---- global signal grid ----

FS: float = 200.0
N: int = 1024
DURATION: float = N / FS  # 5.12 s
TIME = np.arange(N, dtype=np.float64) / FS  # cached time axis


# ----------------------------------------------------------------------
# Primitive waveforms
# ----------------------------------------------------------------------

def sinusoid(
    freq: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    harmonic_strength: float = 0.0,
    t: np.ndarray = TIME,
) -> np.ndarray:
    """s(t) = A sin(2 pi f t + phi) + alpha A sin(4 pi f t + 2 phi).

    Harmonic phase is locked to 2*phi to prevent time-shift equivalences.
    """
    fund = amplitude * np.sin(2 * np.pi * freq * t + phase)
    if harmonic_strength != 0.0:
        harm = harmonic_strength * amplitude * np.sin(4 * np.pi * freq * t + 2 * phase)
        return fund + harm
    return fund


def linear_ramp(slope: float, offset: float = 0.0, t: np.ndarray = TIME) -> np.ndarray:
    return slope * t + offset


def gaussian_pulse(
    amplitude: float,
    center: float,
    width: float,
    t: np.ndarray = TIME,
) -> np.ndarray:
    return amplitude * np.exp(-((t - center) ** 2) / (2.0 * width**2))


def step(amplitude: float, transition_time: float, t: np.ndarray = TIME) -> np.ndarray:
    return amplitude * (t > transition_time).astype(np.float64)


def chirp(
    amplitude: float,
    f0: float,
    f1: float,
    duration: float = DURATION,
    t: np.ndarray = TIME,
) -> np.ndarray:
    """Linear frequency sweep from f0 to f1 over the full window."""
    return amplitude * np.sin(2 * np.pi * (f0 + (f1 - f0) / (2.0 * duration) * t) * t)


def exp_decay(amplitude: float, rate: float, t: np.ndarray = TIME) -> np.ndarray:
    return amplitude * np.exp(-rate * t)


# ----------------------------------------------------------------------
# Composition rules
# ----------------------------------------------------------------------

def additive(*components: np.ndarray) -> np.ndarray:
    out = np.zeros_like(TIME)
    for c in components:
        out = out + c
    return out


def amplitude_modulate(carrier: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    return carrier * envelope


def temporal_concat(s1: np.ndarray, s2: np.ndarray, split_time: float, t: np.ndarray = TIME) -> np.ndarray:
    """Use s1 for t < split_time, s2 otherwise. Both arrays must be N samples long."""
    mask = (t >= split_time)
    return np.where(mask, s2, s1)


# ----------------------------------------------------------------------
# Convenience: noise backgrounds
# ----------------------------------------------------------------------

def gaussian_noise(sigma: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, sigma, size=N)


def random_walk(sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.cumsum(rng.normal(0.0, sigma, size=N))


# ----------------------------------------------------------------------
# Parameter-range registry (paper Section 3.2)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Ranges:
    """Paper-spec parameter ranges. Held-out regions are imposed per-task,
    not at this layer."""
    sinusoid_freq: tuple[float, float] = (1.0, 20.0)
    sinusoid_amp: tuple[float, float] = (0.5, 3.0)
    sinusoid_phase: tuple[float, float] = (0.0, 2 * np.pi)
    sinusoid_harmonic_strength: tuple[float, ...] = (0.0, 0.3, 0.5)
    ramp_slope: tuple[float, float] = (-2.0, 2.0)
    ramp_offset: tuple[float, float] = (-1.0, 1.0)
    pulse_amp: tuple[float, float] = (0.5, 3.0)
    pulse_center: tuple[float, float] = (0.5, 4.5)
    pulse_width: tuple[float, float] = (0.05, 0.5)
    step_amp: tuple[float, float] = (0.5, 3.0)
    step_t0: tuple[float, float] = (0.5, 4.5)
    chirp_amp: tuple[float, float] = (0.5, 3.0)
    chirp_f0: tuple[float, float] = (1.0, 10.0)
    chirp_f1: tuple[float, float] = (5.0, 20.0)
    decay_amp: tuple[float, float] = (0.5, 3.0)
    decay_rate: tuple[float, float] = (0.2, 3.0)


RANGES = Ranges()


# ----------------------------------------------------------------------
# Held-out helpers
# ----------------------------------------------------------------------

def in_heldout(value: float, region: tuple[float, float]) -> bool:
    """Open interval (lo, hi); inclusive at boundaries would let train sneak in."""
    lo, hi = region
    return lo < value < hi


def sample_uniform_excluding(
    rng: np.random.Generator,
    full_range: tuple[float, float],
    heldout: tuple[float, float] | None,
) -> float:
    """Sample uniformly from `full_range`, excluding the open interval `heldout`
    when given. Rejection sampling is fine here because held-out is always a
    small fraction of full."""
    lo, hi = full_range
    if heldout is None:
        return float(rng.uniform(lo, hi))
    for _ in range(1000):
        x = float(rng.uniform(lo, hi))
        if not in_heldout(x, heldout):
            return x
    raise RuntimeError(f"Could not sample outside heldout {heldout} in {full_range}")


def sample_int_excluding(
    rng: np.random.Generator,
    values: Sequence[int],
    heldout: Sequence[int] | None,
) -> int:
    """Sample one int from `values`, optionally excluding `heldout`."""
    if heldout:
        choices = [v for v in values if v not in set(heldout)]
    else:
        choices = list(values)
    return int(rng.choice(choices))


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

__all__ = [
    "FS",
    "N",
    "DURATION",
    "TIME",
    "RANGES",
    "Ranges",
    "sinusoid",
    "linear_ramp",
    "gaussian_pulse",
    "step",
    "chirp",
    "exp_decay",
    "additive",
    "amplitude_modulate",
    "temporal_concat",
    "gaussian_noise",
    "random_walk",
    "in_heldout",
    "sample_uniform_excluding",
    "sample_int_excluding",
]
