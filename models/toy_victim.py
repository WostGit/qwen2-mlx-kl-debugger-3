"""Toy victim that always returns valid dense probabilities."""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np


class ToyVictim:
    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size

    def next_token_probs(self, prompt: str) -> np.ndarray:
        h = hashlib.sha256(prompt.encode("utf-8")).digest()
        rng_seed = int.from_bytes(h[:8], "little", signed=False)
        rng = np.random.default_rng(rng_seed)
        logits = rng.normal(0.0, 1.0, size=self.vocab_size)
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        return probs.astype(np.float64)


def make_toy_prompts(n: int = 64) -> List[str]:
    return [f"toy prompt {i}" for i in range(n)]
