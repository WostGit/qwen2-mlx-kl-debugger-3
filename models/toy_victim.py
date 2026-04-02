from __future__ import annotations

import numpy as np

from experiments.metrics import softmax


class ToyVictim:
    """Toy victim with deterministic dense distributions for each prompt index."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size

    def next_token_probs(self, prompt: str, prompt_idx: int) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((prompt, prompt_idx))) % (2**32))
        logits = rng.normal(size=self.vocab_size)
        return softmax(logits)
