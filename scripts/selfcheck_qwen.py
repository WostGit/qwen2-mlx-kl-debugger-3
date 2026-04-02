from __future__ import annotations

import argparse
import os

from models.qwen_victim import QwenConfig, QwenVictim


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/qwen_startup_selfcheck.txt")
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "~/.cache/huggingface"))
    args = p.parse_args()

    cfg = QwenConfig(cache_dir=args.cache_dir)
    victim = QwenVictim(cfg)
    victim.startup_selfcheck(args.output)
    print(f"Wrote startup self-check to {args.output}")
