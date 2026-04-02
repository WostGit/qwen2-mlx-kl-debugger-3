from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    denom = exps.sum()
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Invalid softmax denominator: {denom}")
    probs = exps / denom
    if not np.isfinite(probs).all():
        raise ValueError("Softmax produced non-finite values")
    return probs


def top1_agreement(victim_probs: np.ndarray, student_probs: np.ndarray) -> float:
    v = int(np.argmax(victim_probs))
    s = int(np.argmax(student_probs))
    return float(v == s)


def kl_divergence(p_victim: np.ndarray, q_student: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p_victim, dtype=np.float64)
    q = np.asarray(q_student, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch for KL: {p.shape} vs {q.shape}")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Non-finite probability vector in KL input")
    p_sum = p.sum()
    q_sum = q.sum()
    if abs(p_sum - 1.0) > 1e-6 or abs(q_sum - 1.0) > 1e-6:
        raise ValueError(f"KL requires normalized vectors; got sums p={p_sum}, q={q_sum}")
    if (p < 0).any() or (q < 0).any():
        raise ValueError("KL requires non-negative vectors")
    q_safe = np.clip(q, eps, 1.0)
    p_safe = np.clip(p, eps, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def is_valid_distribution(v: np.ndarray, atol: float = 1e-6) -> bool:
    v = np.asarray(v, dtype=np.float64)
    return bool(np.isfinite(v).all() and (v >= 0).all() and abs(v.sum() - 1.0) <= atol)
