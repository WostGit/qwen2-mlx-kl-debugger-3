"""Simple student distribution estimators for extraction experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StudentModel:
    vocab_size: int
    alpha: float = 0.05

    def __post_init__(self) -> None:
        self.counts = np.ones(self.vocab_size, dtype=np.float64)

    def update(self, target_distribution: np.ndarray) -> None:
        if target_distribution.shape[0] != self.vocab_size:
            raise ValueError("target_distribution vocab mismatch")
        self.counts += self.alpha * target_distribution

    def predict(self) -> np.ndarray:
        probs = self.counts / self.counts.sum()
        if np.isnan(probs).any() or np.isinf(probs).any():
            raise ValueError("Student produced invalid probabilities")
        return probs


def to_argmax_target(full_probs: np.ndarray) -> np.ndarray:
    out = np.zeros_like(full_probs)
    out[int(np.argmax(full_probs))] = 1.0
    return out


def to_topk_target(full_probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    idx = np.argsort(full_probs)[-k:]
    out = np.zeros_like(full_probs)
    out[idx] = full_probs[idx]
    s = out.sum()
    if s <= 0:
        raise ValueError("top-k target has zero mass")
    return out / s
