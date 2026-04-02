"""Toy victim that always returns valid dense probability vectors."""

from __future__ import annotations

import hashlib
import numpy as np


class ToyVictim:
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size

    def next_token_probs(self, prompt: str) -> np.ndarray:
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        x = rng.random(self.vocab_size) + 1e-9
        probs = x / x.sum()
        return probs.astype(np.float64)
