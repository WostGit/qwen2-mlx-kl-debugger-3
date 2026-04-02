from __future__ import annotations

import numpy as np
import pandas as pd

from attacks.simple_student import SimpleStudent
from experiments.common import EvalRow, build_prompt_set, derive_target, kl_divergence, safe_invalid_row, top1_agreement
from models.toy_victim import ToyVictim


TOY_INTERFACES = ["argmax", "top2", "top3", "top5", "probs"]


def run_toy_budget_sweep(budgets: list[int], seeds: list[int]) -> pd.DataFrame:
    rows: list[EvalRow] = []
    victim = ToyVictim(vocab_size=64)
    eval_prompts = build_prompt_set(n=32, seed=999)

    for seed in seeds:
        for budget in budgets:
            for interface in ["argmax", "top2", "probs"]:
                student = SimpleStudent(vocab_size=victim.vocab_size)
                query_prompts = build_prompt_set(n=budget, seed=seed + budget)
                for p in query_prompts:
                    full = victim.next_token_probs(p)
                    target = derive_target(full, interface)
                    student.update(target)

                agreements = []
                kls = []
                try:
                    for p in eval_prompts:
                        victim_full = victim.next_token_probs(p)
                        student_full = student.predict()
                        agreements.append(top1_agreement(victim_full, student_full))
                        kls.append(kl_divergence(victim_full, student_full))
                    rows.append(
                        EvalRow(
                            source="toy",
                            interface=interface,
                            budget=budget,
                            seed=seed,
                            agreement=float(np.mean(agreements)),
                            kl_divergence=float(np.mean(kls)),
                            kl_valid=True,
                            invalid_reason="",
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(safe_invalid_row("toy", interface, budget, seed, f"toy eval failed: {exc}"))

    return pd.DataFrame([r.__dict__ for r in rows])
