from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="results/results_raw.csv")
    parser.add_argument("--out", default="results/results_summary.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)

    # Strict behavior: never replace NaN KL with zero.
    if "fillna(0" in raw.to_string():
        raise RuntimeError("Detected forbidden NaN->0 replacement attempt.")

    grouped = (
        raw.groupby(["source", "interface", "budget"], dropna=False)
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
