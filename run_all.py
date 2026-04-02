from __future__ import annotations

import argparse
import random

import numpy as np

from experiments.run_qwen_budget_sweep import run as run_qwen_budget
from experiments.run_qwen_topk_sweep import run_topk
from experiments.run_toy_budget_sweep import run as run_toy_budget


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Day 1 extraction experiments")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--multi-seed", default="0,1,2")
    p.add_argument("--debug-llm", action="store_true")
    p.add_argument("--sanity-check-llm", action="store_true")
    args = p.parse_args()

    if args.sanity_check_llm:
        from scripts.sanity_check_llm import run

        run()
        raise SystemExit(0)

    for s in [int(x.strip()) for x in args.multi_seed.split(",") if x.strip()]:
        set_seed(s)
        run_toy_budget(seed=s, output_csv=f"results/toy_budget_sweep_seed{s}.csv")
        run_qwen_budget(seed=s, output_csv=f"results/qwen_budget_sweep_seed{s}.csv", debug_llm=args.debug_llm)

    # canonical merged names expected by summary script
    import pandas as pd

    pd.concat(
        [pd.read_csv(f"results/toy_budget_sweep_seed{s}.csv") for s in [int(x) for x in args.multi_seed.split(",")]],
        ignore_index=True,
    ).to_csv("results/toy_budget_sweep.csv", index=False)
    pd.concat(
        [pd.read_csv(f"results/qwen_budget_sweep_seed{s}.csv") for s in [int(x) for x in args.multi_seed.split(",")]],
        ignore_index=True,
    ).to_csv("results/qwen_budget_sweep.csv", index=False)

    run_topk(seed=args.seed, budget=256, output="results/qwen_topk_sweep.csv", debug_llm=args.debug_llm)

    from scripts.summarize_results import __name__ as _unused  # noqa: F401
    import subprocess

    subprocess.check_call(["python", "scripts/summarize_results.py"])
    subprocess.check_call(["python", "scripts/plot_results.py"])
    print("Completed all experiments")
