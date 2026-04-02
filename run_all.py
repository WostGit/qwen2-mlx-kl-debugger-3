from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n=== Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check-llm", action="store_true")
    parser.add_argument("--debug-llm", action="store_true")
    args = parser.parse_args()

    if args.sanity_check_llm:
        run([sys.executable, "-m", "experiments.run_experiments", "--sanity-check-llm"])
        return

    run([sys.executable, "-m", "experiments.run_experiments"] + (["--debug-llm"] if args.debug_llm else []))
    run([sys.executable, "scripts/summarize_results.py"])
    run([sys.executable, "scripts/plot_results.py"])


if __name__ == "__main__":
    main()
