from __future__ import annotations

import argparse

import numpy as np

from models.qwen_victim import QwenConfig, QwenVictim

PROMPTS = ["Hello", "The weather is", "One plus one equals"]


def run() -> None:
    victim = QwenVictim(QwenConfig())
    for i, prompt in enumerate(PROMPTS):
        probs, info = victim.forward_logits(prompt)
        print(f"[{i}] prompt={prompt!r}")
        print(f"  token list python type: {info['token_ids_python_type']}")
        print(f"  token list length: {len(info['token_ids'])}")
        print(f"  mlx dtype after conversion: {info['mlx_dtype']}")
        print(f"  input shape: {info['input_shape']}")
        print(f"  logits shape: {info['logits_shape']}")
        print(f"  vocab size: {info['vocab_size']}")
        print(f"  prob sum: {probs.sum():.8f}")
        if abs(probs.sum() - 1.0) > 1e-4:
            raise RuntimeError("Softmax probability sum check failed")
        if np.isnan(probs).any() or np.isinf(probs).any():
            raise RuntimeError("Found NaN/Inf in probabilities")
    print("Sanity check passed. Exiting before full experiment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check-llm", action="store_true")
    args = parser.parse_args()
    if args.sanity_check_llm:
        run()
