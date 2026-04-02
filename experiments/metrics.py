from __future__ import annotations

import numpy as np


def top1_agreement(victim_probs: np.ndarray, student_probs: np.ndarray) -> float:
    return float(int(np.argmax(victim_probs) == np.argmax(student_probs)))


def kl_divergence(victim_probs: np.ndarray, student_probs: np.ndarray) -> float:
    if np.any(victim_probs <= 0) or np.any(student_probs <= 0):
        raise ValueError("KL requires strictly positive probabilities")
    if not np.isclose(victim_probs.sum(), 1.0, atol=1e-6):
        raise ValueError("Victim probabilities do not sum to 1")
    if not np.isclose(student_probs.sum(), 1.0, atol=1e-6):
        raise ValueError("Student probabilities do not sum to 1")
    return float(np.sum(victim_probs * (np.log(victim_probs) - np.log(student_probs))))


def interface_target(victim_probs: np.ndarray, interface: str) -> np.ndarray:
    interface = interface.lower()
    n = victim_probs.shape[0]
    if interface == "argmax":
        out = np.zeros(n, dtype=np.float64)
        out[int(np.argmax(victim_probs))] = 1.0
        return out
    if interface == "probs":
        return victim_probs.copy()
    if interface.startswith("top"):
        k = int(interface.replace("top", ""))
        idx = np.argpartition(victim_probs, -k)[-k:]
        out = np.zeros(n, dtype=np.float64)
        out[idx] = victim_probs[idx]
        out /= out.sum()
        return out
    raise ValueError(f"Unknown interface: {interface}")
