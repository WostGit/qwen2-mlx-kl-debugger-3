from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.qwen_experiment import run_qwen
from experiments.toy_experiment import run_toy_budget_sweep
from models.qwen_victim import qwen_startup_selfcheck, sanity_check_llm
from scripts.make_plots import make_plots
from scripts.summarize_results import summarize

BUDGETS = [64, 128, 256, 512, 1024]
SEEDS = [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy + Qwen extraction experiments")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--sanity-check-llm", action="store_true")
    parser.add_argument("--debug-llm", action="store_true")
    parser.add_argument("--qwen-selfcheck", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.qwen_selfcheck:
        qwen_startup_selfcheck(results_dir / "qwen_startup_selfcheck.txt")
        return

    if args.sanity_check_llm:
        sanity_check_llm(results_dir / "qwen_sanity_check.jsonl")
        return

    toy_df = run_toy_budget_sweep(BUDGETS, SEEDS)

    qwen_budget_df = run_qwen(
        budgets=BUDGETS,
        seeds=SEEDS,
        interfaces=["argmax", "top3", "probs"],
        results_dir=results_dir,
        debug_llm=args.debug_llm,
    )
    qwen_topk_df = run_qwen(
        budgets=[256],
        seeds=SEEDS,
        interfaces=["argmax", "top2", "top3", "top5", "probs"],
        results_dir=results_dir,
        debug_llm=args.debug_llm,
    )

    raw = pd.concat([toy_df, qwen_budget_df, qwen_topk_df], ignore_index=True)
    raw.to_csv(results_dir / "results_raw.csv", index=False)
    summary = summarize(results_dir / "results_raw.csv", results_dir / "results_summary.csv")

    qwen_rows = raw[raw["source"] == "qwen"]
    invalid_rate = 1.0 - float(qwen_rows["kl_valid"].mean())
    print(f"[SUMMARY] Qwen invalid KL rate={invalid_rate:.4f}")
    if invalid_rate > 0.01:
        raise SystemExit(f"Qwen invalid KL rate exceeded 1%: {invalid_rate:.4f}")

    if raw["kl_divergence"].fillna(0).equals(raw["kl_divergence"]):
        print("[SUMMARY] No raw KL NaNs detected or conversion not applied.")

    make_plots(results_dir / "results_summary.csv", results_dir)
    print("[DONE] Generated raw/summary CSVs and plots.")
    print(summary.head())


if __name__ == "__main__":
    np.random.seed(42)
    main()
