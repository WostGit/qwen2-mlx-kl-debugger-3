from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_plots(results_summary: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_summary)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (source, interface), g in df.groupby(["source", "interface"]):
        ax.plot(g["budget"], g["agreement_mean"], marker="o", label=f"{source}:{interface}")
    ax.set_title("Agreement vs Budget")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Agreement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "agreement_vs_budget.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (source, interface), g in df.groupby(["source", "interface"]):
        ax.plot(g["budget"], g["kl_mean"], marker="o", label=f"{source}:{interface}")
    ax.set_title("KL vs Budget")
    ax.set_xlabel("Budget")
    ax.set_ylabel("KL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "kl_vs_budget.png", dpi=150)
    plt.close(fig)

    qwen = df[(df["source"] == "qwen") & (df["budget"] == 256)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(qwen["interface"], qwen["kl_mean"])
    ax.set_title("Qwen Top-k KL (Budget=256)")
    ax.set_ylabel("KL")
    fig.tight_layout()
    fig.savefig(out_dir / "qwen_topk_kl.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    make_plots(Path("results/results_summary.csv"), Path("results"))
