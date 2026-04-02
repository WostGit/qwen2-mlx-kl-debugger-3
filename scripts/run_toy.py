from experiments.config import ExperimentConfig, ensure_results_dir
from experiments.run_experiments import run_toy_family


if __name__ == "__main__":
    df = run_toy_family(ExperimentConfig())
    out = ensure_results_dir() / "toy_results.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
