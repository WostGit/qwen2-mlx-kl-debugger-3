from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.toy_experiment import run_toy_budget_sweep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run toy + Qwen extraction experiments")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--sanity-check-llm", action="store_true")
    p.add_argument("--debug-llm", action="store_true")
    p.add_argument("--qwen-startup-selfcheck", action="store_true")
    p.add_argument("--skip-qwen", action="store_true")
    return p.parse_args()


def ensure_seed(seed: int = 1234) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    ensure_seed(1234)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.skip_qwen:
        toy_df = run_toy_budget_sweep([64, 128, 256, 512, 1024], seeds=[0, 1, 2])
        toy_df.to_csv(out / "results_raw.csv", index=False)
        print("Wrote toy-only results")
        return

    from models.qwen_victim import load_qwen, run_sanity_check, write_startup_selfcheck
    from experiments.qwen_experiment import run_qwen_experiment

    ctx = load_qwen()

    if args.qwen_startup_selfcheck:
        write_startup_selfcheck(ctx, out / "qwen_startup_selfcheck.txt")

    if args.sanity_check_llm:
        sanity = run_sanity_check(ctx)
        print("[SANITY-CHECK-LLM]", sanity)
        if not sanity["ok"]:
            raise RuntimeError("--sanity-check-llm failed")
        return

    budgets = [64, 128, 256, 512, 1024]
    seeds = [0, 1, 2]

    toy_df = run_toy_budget_sweep(budgets=budgets, seeds=seeds)

    qwen_budget_df, _, _ = run_qwen_experiment(
        ctx=ctx,
        budgets=budgets,
        seeds=seeds,
        interfaces=["argmax", "top2", "probs"],
        output_dir=out,
        debug_llm=args.debug_llm,
    )
    qwen_topk_df, _, _ = run_qwen_experiment(
        ctx=ctx,
        budgets=[256],
        seeds=seeds,
        interfaces=["argmax", "top2", "top3", "top5", "probs"],
        output_dir=out,
        debug_llm=args.debug_llm,
    )

    raw = pd.concat([toy_df, qwen_budget_df, qwen_topk_df], ignore_index=True)
    raw.to_csv(out / "results_raw.csv", index=False)
    print(f"Wrote raw results to {out / 'results_raw.csv'}")

    invalid_qwen = raw[(raw["source"] == "qwen") & (~raw["kl_valid"]) ]
    invalid_pct = (len(invalid_qwen) / max(len(raw[raw["source"] == "qwen"]), 1)) * 100.0
    print(f"Qwen invalid KL run-level rows: {len(invalid_qwen)} ({invalid_pct:.2f}%)")
    if invalid_pct > 1.0:
        raise RuntimeError("More than 1 percent of Qwen KL rows are invalid.")


if __name__ == "__main__":
    main()
