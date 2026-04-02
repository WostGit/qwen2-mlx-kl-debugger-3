from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.student import make_student_distribution
from experiments.metrics import interface_target, kl_divergence, top1_agreement
from models.qwen_victim import QWEN_MODEL_ID, load_qwen, model_next_token_probs

PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "A quick brown fox",
    "The opposite of hot is",
    "The color of the sky is",
    "Water freezes at",
    "The largest planet is",
    "Python is a",
    "Machine learning is",
    "The moon orbits",
    "A triangle has",
    "The first day of the week is",
]


def _top10(probs: np.ndarray) -> tuple[list[int], list[float]]:
    idx = np.argpartition(probs, -10)[-10:]
    idx = idx[np.argsort(probs[idx])[::-1]]
    return idx.tolist(), probs[idx].tolist()


def run_qwen(
    budgets: list[int],
    seeds: list[int],
    interfaces: list[str],
    results_dir: Path,
    debug_llm: bool = False,
) -> pd.DataFrame:
    model, tokenizer, _cache = load_qwen(QWEN_MODEL_ID)
    rows = []
    debug_rows_jsonl = []
    nan_rows = []

    victim_vocab_size = None
    student_vocab_size = None

    for interface in interfaces:
        for budget in budgets:
            for seed in seeds:
                for prompt_id, prompt in enumerate(PROMPTS):
                    victim_data = model_next_token_probs(model, tokenizer, prompt)
                    victim_probs = victim_data["probs"]
                    target = interface_target(victim_probs, interface=interface)
                    student_probs = make_student_distribution(target, budget=budget, seed=seed + prompt_id)

                    victim_vocab_size = int(victim_probs.shape[0])
                    student_vocab_size = int(student_probs.shape[0])
                    vocab_aligned = victim_vocab_size == student_vocab_size

                    invalid_reason = ""
                    kl = np.nan
                    try:
                        if not vocab_aligned:
                            raise ValueError("vocab_size_mismatch")
                        if np.any(~np.isfinite(victim_probs)) or np.any(~np.isfinite(student_probs)):
                            raise ValueError("nan_or_inf_in_probabilities")
                        kl = kl_divergence(victim_probs, student_probs)
                    except Exception as exc:  # noqa: BLE001
                        invalid_reason = str(exc)
                        nan_rows.append(
                            {
                                "source": "qwen",
                                "interface": interface,
                                "budget": budget,
                                "seed": seed,
                                "prompt_id": prompt_id,
                                "invalid_reason": invalid_reason,
                            }
                        )

                    agreement = top1_agreement(victim_probs, student_probs) if vocab_aligned else np.nan
                    row = {
                        "source": "qwen",
                        "interface": interface,
                        "budget": budget,
                        "seed": seed,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "agreement": agreement,
                        "kl_divergence": kl,
                        "kl_valid": bool(np.isfinite(kl)),
                        "invalid_reason": invalid_reason,
                    }
                    rows.append(row)

                    if debug_llm and prompt_id < 10:
                        v_idx, v_probs = _top10(victim_probs)
                        s_idx, s_probs = _top10(student_probs)
                        debug_rows_jsonl.append(
                            {
                                "prompt_id": prompt_id,
                                "raw_prompt": prompt,
                                "tokenized_prompt_ids": victim_data["token_ids"],
                                "token_ids_python_type": victim_data["token_ids_type"],
                                "mlx_dtype_after_conversion": victim_data["token_array_dtype"],
                                "victim_top10_token_ids": v_idx,
                                "victim_top10_probs": v_probs,
                                "student_top10_token_ids": s_idx,
                                "student_top10_probs": s_probs,
                                "victim_prob_sum": float(victim_probs.sum()),
                                "student_prob_sum": float(student_probs.sum()),
                                "victim_has_nan_or_inf": bool(np.any(~np.isfinite(victim_probs))),
                                "student_has_nan_or_inf": bool(np.any(~np.isfinite(student_probs))),
                                "vocabulary_size": victim_vocab_size,
                                "vocab_alignment": vocab_aligned,
                                "exact_kl": None if not np.isfinite(kl) else float(kl),
                                "kl_skipped_reason": invalid_reason,
                                "interface": interface,
                                "seed": seed,
                                "budget": budget,
                            }
                        )

    results_dir.mkdir(parents=True, exist_ok=True)
    if debug_rows_jsonl:
        jsonl_path = results_dir / "qwen_debug_examples.jsonl"
        csv_path = results_dir / "qwen_debug_examples.csv"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in debug_rows_jsonl:
                f.write(json.dumps(item) + "\n")
        pd.DataFrame(debug_rows_jsonl).to_csv(csv_path, index=False)

    pd.DataFrame(nan_rows).to_csv(results_dir / "qwen_nan_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "victim_vocab_size": victim_vocab_size,
                "student_vocab_size": student_vocab_size,
                "token_index_alignment_ok": victim_vocab_size == student_vocab_size,
            }
        ]
    ).to_csv(results_dir / "qwen_vocab_alignment_check.csv", index=False)

    return pd.DataFrame(rows)
