from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.run_qwen_budget_sweep import run


INTERFACES = {
    "argmax": ("argmax", 1),
    "top2": ("topk", 2),
    "top3": ("topk", 3),
    "top5": ("topk", 5),
    "probs": ("probs", 0),
}


def run_topk(seed: int, budget: int, output: str, debug_llm: bool) -> None:
    rows = []
    for name, (iface, k) in INTERFACES.items():
        tmp = Path(f"results/tmp_qwen_{name}.csv")
        run(seed=seed, output_csv=str(tmp), debug_llm=debug_llm, topk_k=max(1, k))
        df = pd.read_csv(tmp)
        df = df[df["budget"] == budget].copy()
        if iface == "topk":
            df = df[df["interface"] == "topk"]
        else:
            df = df[df["interface"] == iface]
        df["topk_label"] = name
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(output, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--output", default="results/qwen_topk_sweep.csv")
    p.add_argument("--debug-llm", action="store_true")
    args = p.parse_args()
    run_topk(args.seed, args.budget, args.output, args.debug_llm)
