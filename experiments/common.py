from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EvalRow:
    source: str
    interface: str
    budget: int
    seed: int
    agreement: float
    kl_divergence: float | None
    kl_valid: bool
    invalid_reason: str


def top1_agreement(victim_probs: np.ndarray, student_probs: np.ndarray) -> float:
    return float(int(np.argmax(victim_probs) == np.argmax(student_probs)))


def kl_divergence(victim_probs: np.ndarray, student_probs: np.ndarray) -> float:
    if victim_probs.shape != student_probs.shape:
        raise ValueError(f"KL shape mismatch: victim={victim_probs.shape} student={student_probs.shape}")
    if np.any(np.isnan(victim_probs)) or np.any(np.isnan(student_probs)):
        raise ValueError("KL input contains NaN.")
    if np.any(np.isinf(victim_probs)) or np.any(np.isinf(student_probs)):
        raise ValueError("KL input contains inf.")
    if not np.isclose(victim_probs.sum(), 1.0, atol=1e-6):
        raise ValueError("Victim probabilities do not sum to 1.")
    if not np.isclose(student_probs.sum(), 1.0, atol=1e-6):
        raise ValueError("Student probabilities do not sum to 1.")

    eps = 1e-12
    v = np.clip(victim_probs, eps, 1.0)
    s = np.clip(student_probs, eps, 1.0)
    return float(np.sum(v * np.log(v / s)))


def derive_target(full_probs: np.ndarray, interface: str) -> np.ndarray:
    if interface == "argmax":
        t = np.zeros_like(full_probs)
        t[int(np.argmax(full_probs))] = 1.0
        return t
    if interface == "probs":
        return full_probs.copy()
    if interface.startswith("top"):
        k = int(interface.replace("top", ""))
        idx = np.argpartition(full_probs, -k)[-k:]
        t = np.zeros_like(full_probs)
        t[idx] = full_probs[idx]
        denom = t.sum()
        if denom <= 0:
            raise ValueError(f"Top-k renormalization failed for interface={interface}")
        t /= denom
        return t
    raise ValueError(f"Unknown interface: {interface}")


def build_prompt_set(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        val = int(rng.integers(0, 10_000_000))
        out.append(f"Prompt #{i} token {val}: The next word is")
    return out


def safe_invalid_row(source: str, interface: str, budget: int, seed: int, reason: str) -> EvalRow:
    return EvalRow(
        source=source,
        interface=interface,
        budget=budget,
        seed=seed,
        agreement=float("nan"),
        kl_divergence=None,
        kl_valid=False,
        invalid_reason=reason,
    )
