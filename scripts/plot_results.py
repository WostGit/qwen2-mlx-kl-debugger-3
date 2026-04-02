from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    rdir = Path(args.results_dir)
    summary = pd.read_csv(rdir / "results_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for (source, interface), g in summary.groupby(["source", "interface"]):
        ax.plot(g["budget"], g["agreement_mean"], marker="o", label=f"{source}:{interface}")
    ax.set_xlabel("budget")
    ax.set_ylabel("agreement")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(rdir / "agreement_vs_budget.png", dpi=160)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (source, interface), g in summary.groupby(["source", "interface"]):
        ax.plot(g["budget"], g["kl_mean"], marker="o", label=f"{source}:{interface}")
    ax.set_xlabel("budget")
    ax.set_ylabel("kl_divergence")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(rdir / "kl_vs_budget.png", dpi=160)

    topk = pd.read_csv(rdir / "qwen_topk_sweep.csv")
    topk_sum = topk.groupby("topk_label", dropna=False)["kl_divergence"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(topk_sum["topk_label"], topk_sum["kl_divergence"])
    ax.set_xlabel("interface")
    ax.set_ylabel("mean kl_divergence")
    fig.tight_layout()
    fig.savefig(rdir / "qwen_topk_kl.png", dpi=160)
