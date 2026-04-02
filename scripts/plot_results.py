import matplotlib.pyplot as plt
import pandas as pd


def line_plot(df: pd.DataFrame, y_col: str, output: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for (source, interface), sdf in df.groupby(["source", "interface"]):
        if sdf["budget"].nunique() < 2:
            continue
        sdf = sdf.sort_values("budget")
        plt.plot(sdf["budget"], sdf[y_col], marker="o", label=f"{source}:{interface}")
    plt.xscale("log", base=2)
    plt.xlabel("Query budget")
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("results/results_summary.csv")
    line_plot(df, "agreement_mean", "results/agreement_vs_budget.png", "Agreement vs Budget")
    line_plot(df, "kl_mean", "results/kl_vs_budget.png", "KL vs Budget")

    topk = df[df["source"] == "qwen_topk_sweep"].copy()
    plt.figure(figsize=(7, 4))
    if not topk.empty:
        topk = topk.sort_values("interface")
        plt.bar(topk["interface"], topk["kl_mean"])
    plt.ylabel("KL mean")
    plt.title("Qwen top-k interface sweep")
    plt.tight_layout()
    plt.savefig("results/qwen_topk_kl.png", dpi=150)
