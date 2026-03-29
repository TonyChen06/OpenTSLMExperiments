# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import numpy as np


class NoiseInjectionMixin:
    """
    Mixin providing signal perturbation for interpretability testing.

    Two independent perturbation modes:

    1. Noise blending: output = (1 - noise_level) * signal + noise_level * noise
    2. Signal blocking: replace random windows with linear interpolation
       between boundary values, erasing diagnostic features while keeping
       the signal visually continuous.

    Both can be used independently or together (blocking is applied first).

    Each concrete subclass gets its OWN class-level state due to Python's
    class attribute resolution.
    """

    # --- Noise blending ---
    _use_noise = False
    _noise_type = "gaussian"
    _noise_level = 1.0
    _noise_seed = None

    # --- Signal blocking ---
    _use_block = False
    _block_total_sec = 0.0   # Total seconds of signal to block out
    _block_avg_sec = 0.5     # Average block length in seconds
    _block_std_sec = 0.1     # Std of block lengths in seconds
    _block_seed = None

    @classmethod
    def set_noise_mode(cls, use_noise: bool, noise_type: str = "gaussian", noise_level: float = 1.0, noise_seed: int = None):
        """Configure noise injection for all dataset instances."""
        cls.clear_caches()
        cls._use_noise = use_noise
        cls._noise_type = noise_type
        cls._noise_level = float(noise_level)
        cls._noise_seed = noise_seed
        if noise_seed is not None:
            np.random.seed(noise_seed)
        if use_noise:
            level_msg = f", level={cls._noise_level}" if cls._noise_level < 1.0 else ""
            print(f"[NOISE MODE] {cls.__name__}: noise_type='{noise_type}'{level_msg}, seed={noise_seed}")

    @classmethod
    def get_noise_mode(cls) -> dict:
        return {
            "use_noise": cls._use_noise,
            "noise_type": cls._noise_type,
            "noise_level": cls._noise_level,
            "noise_seed": cls._noise_seed,
        }

    @classmethod
    def _generate_noise_signal(cls, length: int, noise_type: str, original_signal: np.ndarray = None) -> np.ndarray:
        if noise_type == "gaussian":
            return np.random.randn(length)
        elif noise_type == "shuffle":
            if original_signal is None:
                return np.random.randn(length)
            shuffled = original_signal.copy()
            np.random.shuffle(shuffled)
            return shuffled
        elif noise_type == "zero":
            return np.zeros(length)
        elif noise_type == "uniform":
            return np.random.uniform(-1, 1, length)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}. Options: gaussian, shuffle, zero, uniform")

    @classmethod
    def _blend_with_noise(cls, original_signal: np.ndarray, noise_type: str) -> np.ndarray:
        """output = (1 - noise_level) * original + noise_level * noise"""
        noise = cls._generate_noise_signal(len(original_signal), noise_type, original_signal)
        level = cls._noise_level
        if level >= 1.0:
            return noise
        if level <= 0.0:
            return original_signal.copy()
        return (1.0 - level) * original_signal + level * noise

    # ---- Signal blocking ----

    @classmethod
    def set_block_mode(cls, use_block: bool, block_total_sec: float = 0.0,
                       block_avg_sec: float = 0.5, block_std_sec: float = 0.1,
                       block_seed: int = None):
        """
        Configure signal blocking for all dataset instances.

        Replaces random windows of the signal with straight-line interpolation
        between the window's boundary values, erasing interior features.

        Args:
            use_block: If True, apply signal blocking
            block_total_sec: Total seconds of signal to block out
            block_avg_sec: Mean block duration in seconds (sampled from Normal)
            block_std_sec: Std of block durations in seconds
            block_seed: Random seed for reproducibility
        """
        cls.clear_caches()
        cls._use_block = use_block
        cls._block_total_sec = float(block_total_sec)
        cls._block_avg_sec = float(block_avg_sec)
        cls._block_std_sec = float(block_std_sec)
        cls._block_seed = block_seed
        if block_seed is not None:
            np.random.seed(block_seed)
        if use_block:
            print(f"[BLOCK MODE] {cls.__name__}: total={block_total_sec}s, "
                  f"avg_block={block_avg_sec}s, std={block_std_sec}s, seed={block_seed}")

    @classmethod
    def _apply_signal_blocking(cls, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Replace random windows of the signal with linear interpolation.

        Each blocked window is replaced by a straight line from signal[start]
        to signal[end], preserving continuity at boundaries while erasing
        interior features.

        Args:
            signal: 1D numpy array (the time series).
            sample_rate: Samples per second (e.g. 100 for ECG at 100Hz).

        Returns:
            Signal with blocked regions replaced by linear interpolation.
        """
        n = len(signal)
        total_samples = int(cls._block_total_sec * sample_rate)
        if total_samples <= 0 or total_samples >= n:
            if total_samples >= n:
                # Block everything: straight line from first to last
                return np.linspace(signal[0], signal[-1], n)
            return signal.copy()

        avg_len = max(1, int(cls._block_avg_sec * sample_rate))
        std_len = max(0, cls._block_std_sec * sample_rate)

        # Generate block lengths from Normal(avg, std), clipped to [1, n/2]
        blocks = []
        accumulated = 0
        while accumulated < total_samples:
            blen = int(np.random.normal(avg_len, std_len))
            blen = max(1, min(blen, n // 2))
            remaining = total_samples - accumulated
            if blen > remaining:
                blen = remaining
            blocks.append(blen)
            accumulated += blen

        n_blocks = len(blocks)
        total_gap = n - total_samples  # total non-blocked samples

        if total_gap <= 0:
            return np.linspace(signal[0], signal[-1], n)

        # Place blocks randomly using random gaps between them
        # Generate n_blocks + 1 gap sizes that sum to total_gap
        cuts = np.sort(np.random.randint(0, total_gap + 1, size=n_blocks))
        gaps = np.empty(n_blocks + 1, dtype=int)
        gaps[0] = cuts[0]
        gaps[1:-1] = np.diff(cuts)
        gaps[-1] = total_gap - cuts[-1]

        # Build the output signal
        out = signal.copy()
        pos = 0
        for i, blen in enumerate(blocks):
            pos += gaps[i]  # skip gap
            start = pos
            end = min(pos + blen, n)
            if end <= start:
                break
            # Linear interpolation from boundary to boundary
            out[start:end] = np.linspace(signal[start], signal[min(end, n) - 1], end - start)
            pos = end

        return out
