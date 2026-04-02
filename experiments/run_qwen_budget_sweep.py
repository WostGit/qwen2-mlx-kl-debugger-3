from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from attacks.student import StudentModel, to_argmax_target, to_topk_target
from experiments.common import append_jsonl, agreement_top1, safe_kl, write_csv
from models.qwen_victim import QwenConfig, QwenVictim

BUDGETS = [64, 128, 256, 512, 1024]
PROMPTS = [
    "Write a single sentence about apples.",
    "What is 2 + 2?",
    "Name one ocean on Earth.",
    "Give a short greeting.",
    "What color is the sky on a clear day?",
    "Say one word that means happy.",
    "Complete: The capital of France is",
    "Write one animal name.",
    "Translate to Spanish: hello",
    "Finish: Machine learning is",
    "What comes after Monday?",
    "Give one programming language.",
] * 120


def run(seed: int, output_csv: str, debug_llm: bool = False, topk_k: int = 5) -> None:
    victim = QwenVictim(QwenConfig())

    rows: List[Dict] = []
    nan_rows: List[Dict] = []
    vocab_rows: List[Dict] = []

    for interface in ["argmax", "topk", "probs"]:
        for budget in BUDGETS:
            student = None
            for i in range(budget):
                p_victim, info = victim.forward_logits(PROMPTS[i])
                if student is None:
                    student = StudentModel(vocab_size=len(p_victim), alpha=0.3)

                if interface == "argmax":
                    t = to_argmax_target(p_victim)
                elif interface == "topk":
                    t = to_topk_target(p_victim, topk_k)
                else:
                    t = p_victim
                student.update(t)

            for eval_i in range(10):
                prompt_id = budget + eval_i + seed
                prompt = PROMPTS[prompt_id]
                p_victim, info = victim.forward_logits(prompt)
                p_student = student.predict()
                vocab_aligned = int(len(p_victim) == len(p_student))
                vocab_rows.append(
                    {
                        "source": "qwen",
                        "interface": interface,
                        "seed": seed,
                        "budget": budget,
                        "victim_vocab_size": len(p_victim),
                        "student_vocab_size": len(p_student),
                        "token_index_alignment": vocab_aligned,
                    }
                )
                kl = safe_kl(p_student, p_victim)
                agreement = agreement_top1(p_student, p_victim)
                invalid_reason = ""
                if np.isnan(kl):
                    invalid_reason = "KL invalid due to NaN/Inf/probability-sum/vocab mismatch"
                    nan_rows.append(
                        {
                            "source": "qwen",
                            "interface": interface,
                            "seed": seed,
                            "budget": budget,
                            "prompt_id": prompt_id,
                            "reason": invalid_reason,
                        }
                    )

                rows.append(
                    {
                        "source": "qwen",
                        "interface": interface,
                        "budget": budget,
                        "seed": seed,
                        "agreement": agreement,
                        "kl_divergence": kl,
                        "kl_valid": int(not np.isnan(kl)),
                    }
                )

                if debug_llm:
                    top_v = np.argsort(p_victim)[-10:][::-1]
                    top_s = np.argsort(p_student)[-10:][::-1]
                    append_jsonl(
                        f"results/debug_qwen_{interface}_seed{seed}.jsonl",
                        [
                            {
                                "prompt_id": prompt_id,
                                "prompt_text": prompt,
                                "tokenized_prompt_ids": info["token_ids"],
                                "token_ids_python_type": info["token_ids_python_type"],
                                "mlx_dtype": info["mlx_dtype"],
                                "victim_top10_token_ids": top_v.tolist(),
                                "victim_top10_probs": p_victim[top_v].tolist(),
                                "student_top10_token_ids": top_s.tolist(),
                                "student_top10_probs": p_student[top_s].tolist(),
                                "victim_prob_sum": float(p_victim.sum()),
                                "student_prob_sum": float(p_student.sum()),
                                "nan_flag": bool(np.isnan(p_victim).any() or np.isnan(p_student).any()),
                                "inf_flag": bool(np.isinf(p_victim).any() or np.isinf(p_student).any()),
                                "vocab_size": len(p_victim),
                                "vocab_alignment": bool(vocab_aligned),
                                "exact_kl": kl,
                                "kl_skip_reason": invalid_reason,
                            }
                        ],
                    )

    write_csv(output_csv, rows)
    write_csv("results/qwen_nan_report.csv", nan_rows)
    write_csv("results/qwen_vocab_alignment_check.csv", vocab_rows)

    invalid_rate = 1.0 if not rows else 1 - (sum(r["kl_valid"] for r in rows) / len(rows))
    if invalid_rate > 0.01:
        raise RuntimeError(f"Invalid Qwen KL rate is {invalid_rate:.2%}, exceeding 1% threshold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/qwen_budget_sweep.csv")
    parser.add_argument("--debug-llm", action="store_true")
    parser.add_argument("--topk-k", type=int, default=5)
    args = parser.parse_args()
    run(seed=args.seed, output_csv=args.output, debug_llm=args.debug_llm, topk_k=args.topk_k)
