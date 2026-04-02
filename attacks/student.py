from __future__ import annotations

import numpy as np

from experiments.metrics import softmax


class StudentModel:
    """Simple student that learns from target vectors by averaging."""

    def __init__(self, vocab_size: int, seed: int) -> None:
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)
        self.prototype = softmax(self.rng.normal(size=vocab_size))

    def train_step(self, target_probs: np.ndarray, lr: float = 0.2) -> None:
        self.prototype = (1 - lr) * self.prototype + lr * target_probs
        self.prototype = self.prototype / self.prototype.sum()

    def predict(self, _: str) -> np.ndarray:
        noise = self.rng.normal(scale=0.02, size=self.vocab_size)
        return softmax(np.log(np.clip(self.prototype, 1e-12, 1.0)) + noise)
