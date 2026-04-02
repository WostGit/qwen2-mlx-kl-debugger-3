"""Simple student estimator used for toy and Qwen experiment paths."""

from __future__ import annotations

import numpy as np


def make_student_distribution(victim_probs: np.ndarray, budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + budget)
    alpha = max(1.0, budget / 64.0)
    noisy = victim_probs + rng.normal(0.0, 1.0 / (20.0 * alpha), size=victim_probs.shape)
    noisy = np.clip(noisy, 1e-12, None)
    noisy = noisy / noisy.sum()
    return noisy
