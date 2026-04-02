from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    p = Path("results/results_raw.csv")
    if not p.exists():
        raise FileNotFoundError("results/results_raw.csv missing")
    df = pd.read_csv(p)
    summary = (
        df.groupby(["source", "interface", "budget"], dropna=False)
        .agg(
            agreement_mean=("agreement", "mean"),
            agreement_std=("agreement", "std"),
            kl_mean=("kl_divergence", "mean"),
            kl_std=("kl_divergence", "std"),
            valid_kl_rows=("kl_divergence", lambda s: int(s.notna().sum())),
            invalid_kl_rows=("kl_divergence", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    summary.to_csv("results/results_summary.csv", index=False)
    print("Wrote results/results_summary.csv")
