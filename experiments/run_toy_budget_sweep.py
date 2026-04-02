from __future__ import annotations

import argparse
from typing import Dict, List

from attacks.student import StudentModel, to_argmax_target, to_topk_target
from experiments.common import agreement_top1, safe_kl, write_csv
from models.toy_victim import ToyVictim, make_toy_prompts


BUDGETS = [64, 128, 256, 512, 1024]


def run(seed: int, output_csv: str) -> None:
    victim = ToyVictim(vocab_size=64)
    prompts = make_toy_prompts(1200)
    rows: List[Dict] = []

    for interface in ["argmax", "topk", "probs"]:
        for budget in BUDGETS:
            student = StudentModel(vocab_size=victim.vocab_size, alpha=0.2)
            for i in range(budget):
                p = victim.next_token_probs(prompts[i])
                if interface == "argmax":
                    t = to_argmax_target(p)
                elif interface == "topk":
                    t = to_topk_target(p, 5)
                else:
                    t = p
                student.update(t)

            eval_prompt = prompts[budget + seed % 10]
            v = victim.next_token_probs(eval_prompt)
            s = student.predict()
            kl = safe_kl(s, v)
            rows.append(
                {
                    "source": "toy",
                    "interface": interface,
                    "budget": budget,
                    "seed": seed,
                    "agreement": agreement_top1(s, v),
                    "kl_divergence": kl,
                    "kl_valid": int(kl == kl),
                }
            )

    write_csv(output_csv, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/toy_budget_sweep.csv")
    args = parser.parse_args()
    run(args.seed, args.output)
