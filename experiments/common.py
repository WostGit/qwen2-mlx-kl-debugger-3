from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

EPS = 1e-12


def safe_kl(p_student: np.ndarray, p_victim: np.ndarray) -> float:
    if p_student.shape != p_victim.shape:
        raise ValueError("Cannot compute KL on mismatched vocabularies")
    if np.isnan(p_student).any() or np.isnan(p_victim).any():
        return float("nan")
    if np.isinf(p_student).any() or np.isinf(p_victim).any():
        return float("nan")
    if abs(p_student.sum() - 1.0) > 1e-6 or abs(p_victim.sum() - 1.0) > 1e-6:
        return float("nan")
    p = np.clip(p_student, EPS, 1.0)
    q = np.clip(p_victim, EPS, 1.0)
    return float(np.sum(p * np.log(p / q)))


def agreement_top1(p_student: np.ndarray, p_victim: np.ndarray) -> float:
    if p_student.shape != p_victim.shape:
        return float("nan")
    return float(np.argmax(p_student) == np.argmax(p_victim))


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
