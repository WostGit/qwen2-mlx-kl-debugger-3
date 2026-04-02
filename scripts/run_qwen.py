import argparse

from experiments.config import ExperimentConfig, ensure_results_dir
from experiments.run_experiments import run_qwen_family


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-llm", action="store_true")
    args = parser.parse_args()
    df = run_qwen_family(ExperimentConfig(), debug_llm=args.debug_llm)
    out = ensure_results_dir() / "qwen_results.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
