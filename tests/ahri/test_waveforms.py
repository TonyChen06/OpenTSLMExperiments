"""Waveform primitives and composition rules."""

from __future__ import annotations

import numpy as np

from opentslm.ahri import waveforms as wf


def test_constants():
    assert wf.FS == 200.0
    assert wf.N == 1024
    assert abs(wf.DURATION - 5.12) < 1e-6
    assert wf.TIME.shape == (1024,)


def test_sinusoid_shape_and_period():
    s = wf.sinusoid(freq=1.0, amplitude=1.0, phase=0.0)
    assert s.shape == (1024,)
    # 1 Hz over 5.12 s = ~5 cycles; expect zero crossings near integer multiples of 0.5 s
    zero_crossings = np.where(np.diff(np.sign(s)))[0]
    assert len(zero_crossings) >= 10


def test_sinusoid_harmonic_locked():
    """Harmonic phase should track 2*phi. Two settings with different phi
    must produce non-identical signals (no time-shift equivalence)."""
    a = wf.sinusoid(freq=2.0, amplitude=1.0, phase=0.0, harmonic_strength=0.5)
    b = wf.sinusoid(freq=2.0, amplitude=1.0, phase=np.pi / 3, harmonic_strength=0.5)
    assert not np.allclose(a, b)


def test_primitives_return_correct_length():
    for fn, kwargs in [
        (wf.linear_ramp, {"slope": 1.0, "offset": 0.0}),
        (wf.gaussian_pulse, {"amplitude": 1.0, "center": 2.0, "width": 0.1}),
        (wf.step, {"amplitude": 1.0, "transition_time": 2.5}),
        (wf.chirp, {"amplitude": 1.0, "f0": 1.0, "f1": 5.0}),
        (wf.exp_decay, {"amplitude": 1.0, "rate": 1.0}),
    ]:
        s = fn(**kwargs)
        assert s.shape == (1024,), fn.__name__


def test_additive_composition():
    a = wf.sinusoid(1.0)
    b = wf.linear_ramp(0.5)
    s = wf.additive(a, b)
    np.testing.assert_allclose(s, a + b)


def test_temporal_concat_at_boundary():
    s1 = np.ones(wf.N)
    s2 = np.full(wf.N, 2.0)
    s = wf.temporal_concat(s1, s2, split_time=2.56)
    # first half should be 1, second half 2
    assert s[0] == 1.0 and s[-1] == 2.0
    boundary_idx = int(2.56 * wf.FS)
    assert s[boundary_idx - 1] == 1.0
    assert s[boundary_idx] == 2.0


def test_in_heldout_open_interval():
    assert wf.in_heldout(7.5, (6.0, 8.0))
    assert not wf.in_heldout(6.0, (6.0, 8.0))  # open interval at endpoints
    assert not wf.in_heldout(8.0, (6.0, 8.0))
    assert not wf.in_heldout(3.0, (6.0, 8.0))


def test_sample_uniform_excluding_actually_excludes():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = wf.sample_uniform_excluding(rng, (1.0, 20.0), (10.0, 15.0))
        assert 1.0 <= x <= 20.0
        assert not (10.0 < x < 15.0)


def test_sample_int_excluding():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = wf.sample_int_excluding(rng, list(range(1, 11)), [6, 7])
        assert x in {1, 2, 3, 4, 5, 8, 9, 10}
