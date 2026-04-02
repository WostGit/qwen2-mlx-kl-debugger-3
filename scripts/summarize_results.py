from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    rdir = Path(args.results_dir)
    frames = []
    for path in [
        rdir / "toy_budget_sweep.csv",
        rdir / "qwen_budget_sweep.csv",
        rdir / "qwen_topk_sweep.csv",
    ]:
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        raise RuntimeError("No result files found")

    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(rdir / "results_raw.csv", index=False)

    if (raw["kl_divergence"] == 0).sum() > 0 and raw["kl_divergence"].isna().sum() > 0:
        print("Note: KL zeros exist, but NaNs are preserved in raw output.")

    summary = (
        raw.groupby(["source", "interface", "budget"], dropna=False)
        .agg(
            agreement_mean=("agreement", "mean"),
            agreement_std=("agreement", "std"),
            kl_mean=("kl_divergence", "mean"),
            kl_std=("kl_divergence", "std"),
            kl_valid_count=("kl_valid", "sum"),
            row_count=("kl_valid", "count"),
        )
        .reset_index()
    )
    summary["kl_invalid_count"] = summary["row_count"] - summary["kl_valid_count"]
    summary.to_csv(rdir / "results_summary.csv", index=False)

    if raw["kl_divergence"].isna().any() and not (rdir / "qwen_nan_report.csv").exists():
        raise RuntimeError("NaN KL detected but qwen_nan_report.csv missing")
