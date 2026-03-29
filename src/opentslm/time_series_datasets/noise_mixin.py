# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import numpy as np


class NoiseInjectionMixin:
    """
    Mixin providing noise injection for interpretability testing.

    Blending formula: output = (1 - noise_level) * signal + noise_level * noise

    Each concrete subclass gets its OWN class-level state due to Python's
    class attribute resolution.
    """

    _use_noise = False
    _noise_type = "gaussian"
    _noise_level = 1.0
    _noise_seed = None

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
