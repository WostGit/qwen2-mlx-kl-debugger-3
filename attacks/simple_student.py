"""A tiny student that estimates next-token distributions from queried targets."""

from __future__ import annotations

import numpy as np


class SimpleStudent:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.mass = np.ones(vocab_size, dtype=np.float64) * 1e-6

    def update(self, target_distribution: np.ndarray) -> None:
        if target_distribution.shape[0] != self.vocab_size:
            raise ValueError("Student/victim vocabulary mismatch during update.")
        self.mass += target_distribution

    def predict(self) -> np.ndarray:
        probs = self.mass / self.mass.sum()
        return probs.astype(np.float64)
