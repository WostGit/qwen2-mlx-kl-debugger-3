from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.simple_student import SimpleStudent
from experiments.common import EvalRow, build_prompt_set, derive_target, kl_divergence, safe_invalid_row, top1_agreement
from models.qwen_victim import QwenContext, qwen_next_token_probs


def _top10(probs: np.ndarray) -> tuple[list[int], list[float]]:
    idx = np.argpartition(probs, -10)[-10:]
    idx = idx[np.argsort(probs[idx])[::-1]]
    return [int(i) for i in idx], [float(probs[i]) for i in idx]


def run_qwen_experiment(
    ctx: QwenContext,
    budgets: list[int],
    seeds: list[int],
    interfaces: list[str],
    output_dir: str | Path,
    debug_llm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[EvalRow] = []
    nan_rows: list[dict] = []
    align_rows: list[dict] = []

    eval_prompts = build_prompt_set(n=32, seed=555)

    for seed in seeds:
        for budget in budgets:
            for interface in interfaces:
                query_prompts = build_prompt_set(n=budget, seed=seed + budget)
                victim_first = qwen_next_token_probs(ctx, query_prompts[0])
                vocab_size = victim_first["vocab_size"]
                student = SimpleStudent(vocab_size=vocab_size)

                for p in query_prompts:
                    r = qwen_next_token_probs(ctx, p)
                    full_probs = r["probs"]
                    target = derive_target(full_probs, interface)
                    student.update(target)

                agreements = []
                kls = []
                debug_examples = []

                for i, p in enumerate(eval_prompts):
                    r = qwen_next_token_probs(ctx, p)
                    victim_probs = r["probs"]
                    student_probs = student.predict()

                    vocab_ok = bool(victim_probs.shape[0] == student_probs.shape[0])
                    align_rows.append(
                        {
                            "source": "qwen",
                            "interface": interface,
                            "budget": budget,
                            "seed": seed,
                            "victim_vocab_size": int(victim_probs.shape[0]),
                            "student_vocab_size": int(student_probs.shape[0]),
                            "vocab_alignment_ok": vocab_ok,
                            "token_index_alignment": "assumed_shared_index_space",
                        }
                    )

                    try:
                        agr = top1_agreement(victim_probs, student_probs)
                        kl = kl_divergence(victim_probs, student_probs)
                        agreements.append(agr)
                        kls.append(kl)
                        invalid_reason = ""
                        valid = True
                    except Exception as exc:  # noqa: BLE001
                        invalid_reason = str(exc)
                        valid = False
                        nan_rows.append(
                            {
                                "source": "qwen",
                                "interface": interface,
                                "budget": budget,
                                "seed": seed,
                                "prompt_id": i,
                                "prompt": p,
                                "reason": invalid_reason,
                            }
                        )

                    if debug_llm and i < 10:
                        vt_i, vt_p = _top10(victim_probs)
                        st_i, st_p = _top10(student_probs)
                        debug_examples.append(
                            {
                                "prompt_id": i,
                                "prompt": p,
                                "tokenized_prompt_ids": r["token_ids"],
                                "token_ids_python_type": r["token_ids_python_type"],
                                "mlx_dtype": r["mlx_dtype"],
                                "victim_top10_token_ids": vt_i,
                                "victim_top10_probs": vt_p,
                                "student_top10_token_ids": st_i,
                                "student_top10_probs": st_p,
                                "victim_prob_sum": float(victim_probs.sum()),
                                "student_prob_sum": float(student_probs.sum()),
                                "victim_has_nan_or_inf": bool(np.any(~np.isfinite(victim_probs))),
                                "student_has_nan_or_inf": bool(np.any(~np.isfinite(student_probs))),
                                "vocabulary_size": int(victim_probs.shape[0]),
                                "vocab_alignment_ok": vocab_ok,
                                "kl": None if not valid else float(kl),
                                "kl_skipped_reason": invalid_reason,
                            }
                        )

                if debug_llm:
                    jsonl_path = output_dir / f"qwen_debug_{interface}_b{budget}_s{seed}.jsonl"
                    csv_path = output_dir / f"qwen_debug_{interface}_b{budget}_s{seed}.csv"
                    with jsonl_path.open("w", encoding="utf-8") as f:
                        for row in debug_examples:
                            f.write(json.dumps(row) + "\n")
                    pd.DataFrame(debug_examples).to_csv(csv_path, index=False)

                valid_ratio = (len(kls) / len(eval_prompts)) if eval_prompts else 0.0
                if len(kls) == 0:
                    rows.append(
                        safe_invalid_row("qwen", interface, budget, seed, "No valid KL rows for this run.")
                    )
                else:
                    rows.append(
                        EvalRow(
                            source="qwen",
                            interface=interface,
                            budget=budget,
                            seed=seed,
                            agreement=float(np.mean(agreements)),
                            kl_divergence=float(np.mean(kls)),
                            kl_valid=valid_ratio >= 0.99,
                            invalid_reason="" if valid_ratio >= 0.99 else "More than 1% invalid KL rows.",
                        )
                    )

    df = pd.DataFrame([r.__dict__ for r in rows])
    nan_df = pd.DataFrame(nan_rows)
    align_df = pd.DataFrame(align_rows)
    if not nan_df.empty:
        nan_df.to_csv(output_dir / "qwen_nan_report.csv", index=False)
    else:
        pd.DataFrame(columns=["source", "interface", "budget", "seed", "prompt_id", "prompt", "reason"]).to_csv(
            output_dir / "qwen_nan_report.csv", index=False
        )
    align_df.to_csv(output_dir / "qwen_vocab_alignment_check.csv", index=False)
    return df, nan_df, align_df
