from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize(raw_csv: Path, summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    # enforce: no NaN -> zero replacement
    if (df["kl_divergence"].isna() & (df["kl_valid"])).any():
        raise ValueError("Found kl_valid=True rows with NaN KL")

    grouped = (
        df.groupby(["source", "interface", "budget"], dropna=False)
        .agg(
            agreement_mean=("agreement", "mean"),
            agreement_std=("agreement", "std"),
            kl_mean=("kl_divergence", "mean"),
            kl_std=("kl_divergence", "std"),
            valid_kl_rows=("kl_valid", "sum"),
            total_rows=("kl_valid", "count"),
        )
        .reset_index()
    )
    grouped["invalid_kl_rows"] = grouped["total_rows"] - grouped["valid_kl_rows"]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(summary_csv, index=False)
    return grouped


if __name__ == "__main__":
    summarize(Path("results/results_raw.csv"), Path("results/results_summary.csv"))
