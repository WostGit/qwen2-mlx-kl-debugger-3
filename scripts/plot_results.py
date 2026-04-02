from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="results/results_raw.csv")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.raw)

    budget_df = df[df["interface"].isin(["argmax", "top2", "probs"])]
    qwen = budget_df[budget_df["source"] == "qwen"]

    plt.figure(figsize=(8, 5))
    for interface, g in qwen.groupby("interface"):
        agg = g.groupby("budget")["agreement"].mean().reset_index()
        plt.plot(agg["budget"], agg["agreement"], marker="o", label=interface)
    plt.title("Qwen agreement vs budget")
    plt.xlabel("budget")
    plt.ylabel("agreement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "agreement_vs_budget.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for interface, g in qwen.groupby("interface"):
        agg = g.groupby("budget")["kl_divergence"].mean().reset_index()
        plt.plot(agg["budget"], agg["kl_divergence"], marker="o", label=interface)
    plt.title("Qwen KL vs budget")
    plt.xlabel("budget")
    plt.ylabel("KL divergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "kl_vs_budget.png", dpi=160)
    plt.close()

    topk_df = df[(df["source"] == "qwen") & (df["budget"] == 256)]
    order = ["argmax", "top2", "top3", "top5", "probs"]
    vals = [topk_df[topk_df["interface"] == k]["kl_divergence"].mean() for k in order]
    plt.figure(figsize=(8, 5))
    plt.bar(order, vals)
    plt.title("Qwen top-k KL at fixed budget=256")
    plt.xlabel("interface")
    plt.ylabel("KL divergence")
    plt.tight_layout()
    plt.savefig(outdir / "qwen_topk_kl.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
