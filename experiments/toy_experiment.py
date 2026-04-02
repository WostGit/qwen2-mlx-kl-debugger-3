from __future__ import annotations

import numpy as np
import pandas as pd

from attacks.student import make_student_distribution
from experiments.metrics import kl_divergence, top1_agreement


TOY_VOCAB = 64


def run_toy_budget_sweep(budgets: list[int], seeds: list[int]) -> pd.DataFrame:
    rows = []
    base_rng = np.random.default_rng(2026)
    for budget in budgets:
        for seed in seeds:
            victim = base_rng.dirichlet(np.ones(TOY_VOCAB))
            student = make_student_distribution(victim, budget=budget, seed=seed)
            kl = kl_divergence(victim, student)
            agree = top1_agreement(victim, student)
            rows.append(
                {
                    "source": "toy",
                    "interface": "dense",
                    "budget": budget,
                    "seed": seed,
                    "agreement": agree,
                    "kl_divergence": kl,
                    "kl_valid": True,
                    "invalid_reason": "",
                }
            )
    return pd.DataFrame(rows)
