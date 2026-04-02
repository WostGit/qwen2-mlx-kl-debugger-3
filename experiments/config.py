from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = "Qwen/Qwen2-0.5B-Instruct"
    budgets: tuple[int, ...] = (64, 128, 256, 512, 1024)
    seeds: tuple[int, ...] = (0, 1, 2)
    qwen_interfaces: tuple[str, ...] = ("argmax", "topk", "probs")
    qwen_topk_sweep: tuple[str, ...] = ("argmax", "top2", "top3", "top5", "probs")
    fixed_budget_topk_sweep: int = 512
    topk_default: int = 5
    debug_examples: int = 10
    kl_eps: float = 1e-12
    invalid_row_failure_threshold: float = 0.01


def default_prompts() -> List[str]:
    return [
        "The capital of France is",
        "In one sentence, define entropy:",
        "Python is a programming language that",
        "The moon orbits the",
        "Machine learning models often",
        "A robust experiment should",
        "The opposite of hot is",
        "Data science combines",
        "The next word in this phrase is",
        "Open-source software enables",
        "A small model can",
        "Deterministic seeding helps",
    ]


def ensure_results_dir() -> Path:
    p = Path("results")
    p.mkdir(parents=True, exist_ok=True)
    return p


def as_yaml_dict(config: ExperimentConfig) -> dict:
    return asdict(config)
