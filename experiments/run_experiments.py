from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from attacks.student import StudentModel
from experiments.config import ExperimentConfig, default_prompts, ensure_results_dir
from experiments.metrics import is_valid_distribution, kl_divergence, top1_agreement
from models.qwen_victim import QwenVictim, interface_target_from_dense, write_jsonl
from models.toy_victim import ToyVictim


def _init_rng(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def run_toy_family(config: ExperimentConfig) -> pd.DataFrame:
    prompts = default_prompts()
    victim = ToyVictim(vocab_size=128)
    rows: List[Dict[str, Any]] = []

    for seed in config.seeds:
        _init_rng(seed)
        for budget in config.budgets:
            student = StudentModel(vocab_size=victim.vocab_size, seed=seed)
            for q in range(budget):
                idx = q % len(prompts)
                target = victim.next_token_probs(prompts[idx], idx)
                student.train_step(target)

            eval_kl, eval_agree = [], []
            for idx, prompt in enumerate(prompts):
                v = victim.next_token_probs(prompt, idx)
                s = student.predict(prompt)
                eval_kl.append(kl_divergence(v, s, eps=config.kl_eps))
                eval_agree.append(top1_agreement(v, s))

            rows.append(
                {
                    "source": "toy",
                    "interface": "dense_probs",
                    "budget": budget,
                    "seed": seed,
                    "agreement": float(np.mean(eval_agree)),
                    "kl_divergence": float(np.mean(eval_kl)),
                    "kl_valid": True,
                    "invalid_reason": "",
                }
            )
    return pd.DataFrame(rows)


def _qwen_eval_one(
    prompt_id: int,
    prompt: str,
    interface: str,
    victim_probs: np.ndarray,
    student_probs: np.ndarray,
    token_ids: List[int],
    token_python_type: str,
    mlx_dtype: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "tokenized_prompt_ids": token_ids,
        "token_ids_python_type": token_python_type,
        "mlx_dtype": mlx_dtype,
        "interface": interface,
        "vocab_size": int(victim_probs.shape[0]),
        "victim_prob_sum": float(victim_probs.sum()),
        "student_prob_sum": float(student_probs.sum()),
        "victim_has_nan_or_inf": bool((~np.isfinite(victim_probs)).any()),
        "student_has_nan_or_inf": bool((~np.isfinite(student_probs)).any()),
    }
    aligned = victim_probs.shape == student_probs.shape
    row["vocab_aligned"] = aligned
    if not aligned:
        row["kl_valid"] = False
        row["invalid_reason"] = "vocabulary mismatch"
        row["kl_divergence"] = np.nan
        row["agreement"] = np.nan
        return row

    row["victim_top10_token_ids"] = np.argsort(-victim_probs)[:10].tolist()
    row["victim_top10_probs"] = np.sort(victim_probs)[-10:][::-1].tolist()
    row["student_top10_token_ids"] = np.argsort(-student_probs)[:10].tolist()
    row["student_top10_probs"] = np.sort(student_probs)[-10:][::-1].tolist()

    try:
        if not is_valid_distribution(victim_probs) or not is_valid_distribution(student_probs):
            raise ValueError("invalid dense distribution")
        row["kl_divergence"] = kl_divergence(victim_probs, student_probs)
        row["agreement"] = top1_agreement(victim_probs, student_probs)
        row["kl_valid"] = True
        row["invalid_reason"] = ""
    except Exception as exc:
        row["kl_divergence"] = np.nan
        row["agreement"] = np.nan
        row["kl_valid"] = False
        row["invalid_reason"] = str(exc)
    return row


def run_qwen_family(config: ExperimentConfig, debug_llm: bool = False) -> pd.DataFrame:
    results_dir = ensure_results_dir()
    prompts = default_prompts()
    victim = QwenVictim(model_id=config.model_id)
    startup_path = results_dir / "qwen_startup_selfcheck.txt"
    runtime = victim.initialize(startup_path)

    vocab_size = victim._tokenizer_vocab_size(runtime.tokenizer)
    vocab_alignment_rows = [
        {
            "model_id": config.model_id,
            "victim_vocab_size": vocab_size,
            "student_vocab_size": vocab_size,
            "token_index_alignment": True,
            "notes": "Student vectors are created directly on victim token index support.",
        }
    ]

    rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    for seed in config.seeds:
        _init_rng(seed)
        for budget in config.budgets:
            for interface in config.qwen_interfaces:
                student = StudentModel(vocab_size=vocab_size, seed=seed)
                for q in range(budget):
                    p = prompts[q % len(prompts)]
                    victim_dense = victim.next_token_probs(p)
                    target = interface_target_from_dense(victim_dense, interface, topk_default=config.topk_default)
                    student.train_step(target)

                eval_rows_for_combo: List[Dict[str, Any]] = []
                for pid, prompt in enumerate(prompts):
                    token_ids = victim.tokenize_to_python_int_list(runtime.tokenizer, prompt)
                    mx_ids = victim.token_ids_to_mx(token_ids)
                    victim_dense = victim.next_token_probs(prompt)
                    student_dense = student.predict(prompt)
                    row = _qwen_eval_one(
                        prompt_id=pid,
                        prompt=prompt,
                        interface=interface,
                        victim_probs=victim_dense,
                        student_probs=student_dense,
                        token_ids=token_ids,
                        token_python_type=str(type(token_ids)),
                        mlx_dtype=str(mx_ids.dtype),
                    )
                    row["source"] = "qwen"
                    row["seed"] = seed
                    row["budget"] = budget
                    eval_rows_for_combo.append(row)

                df_combo = pd.DataFrame(eval_rows_for_combo)
                invalid_ratio = 1.0 - float(df_combo["kl_valid"].mean())
                rows.append(
                    {
                        "source": "qwen",
                        "interface": interface,
                        "budget": budget,
                        "seed": seed,
                        "agreement": float(df_combo["agreement"].dropna().mean()) if df_combo["agreement"].notna().any() else np.nan,
                        "kl_divergence": float(df_combo["kl_divergence"].dropna().mean())
                        if df_combo["kl_divergence"].notna().any()
                        else np.nan,
                        "kl_valid": bool(invalid_ratio <= config.invalid_row_failure_threshold),
                        "invalid_reason": "too_many_invalid_rows" if invalid_ratio > config.invalid_row_failure_threshold else "",
                        "invalid_ratio": invalid_ratio,
                    }
                )

                if debug_llm:
                    debug_rows.extend(eval_rows_for_combo[: config.debug_examples])

    # Fixed-budget top-k sweep
    for seed in config.seeds:
        _init_rng(seed)
        budget = config.fixed_budget_topk_sweep
        for interface in config.qwen_topk_sweep:
            student = StudentModel(vocab_size=vocab_size, seed=seed + 99)
            for q in range(budget):
                prompt = prompts[q % len(prompts)]
                victim_dense = victim.next_token_probs(prompt)
                target = interface_target_from_dense(victim_dense, interface, topk_default=config.topk_default)
                student.train_step(target)
            kls, agrees, invalid = [], [], 0
            for pid, prompt in enumerate(prompts):
                token_ids = victim.tokenize_to_python_int_list(runtime.tokenizer, prompt)
                mx_ids = victim.token_ids_to_mx(token_ids)
                victim_dense = victim.next_token_probs(prompt)
                student_dense = student.predict(prompt)
                rr = _qwen_eval_one(
                    pid,
                    prompt,
                    interface,
                    victim_dense,
                    student_dense,
                    token_ids,
                    str(type(token_ids)),
                    str(mx_ids.dtype),
                )
                if rr["kl_valid"]:
                    kls.append(rr["kl_divergence"])
                    agrees.append(rr["agreement"])
                else:
                    invalid += 1
                if debug_llm and pid < config.debug_examples:
                    rr["source"] = "qwen_topk_sweep"
                    rr["seed"] = seed
                    rr["budget"] = budget
                    debug_rows.append(rr)

            rows.append(
                {
                    "source": "qwen_topk_sweep",
                    "interface": interface,
                    "budget": budget,
                    "seed": seed,
                    "agreement": float(np.mean(agrees)) if agrees else np.nan,
                    "kl_divergence": float(np.mean(kls)) if kls else np.nan,
                    "kl_valid": invalid / len(prompts) <= config.invalid_row_failure_threshold,
                    "invalid_reason": "too_many_invalid_rows" if invalid / len(prompts) > config.invalid_row_failure_threshold else "",
                    "invalid_ratio": invalid / len(prompts),
                }
            )

    pd.DataFrame(vocab_alignment_rows).to_csv(results_dir / "qwen_vocab_alignment_check.csv", index=False)
    if debug_llm:
        debug_df = pd.DataFrame(debug_rows)
        debug_df.to_csv(results_dir / "qwen_debug_examples.csv", index=False)
        write_jsonl(results_dir / "qwen_debug_examples.jsonl", debug_rows)
        nan_df = debug_df[~debug_df["kl_valid"]].copy()
    else:
        nan_df = pd.DataFrame(columns=["prompt_id", "prompt", "interface", "invalid_reason"])
    nan_df.to_csv(results_dir / "qwen_nan_report.csv", index=False)

    out = pd.DataFrame(rows)
    # fail early if summary logic would coerce NaN to 0; we intentionally preserve NaN
    if (out["kl_divergence"] == 0).any() and out["kl_divergence"].isna().any():
        raise RuntimeError("Unexpected KL zero values mixed with NaN; refusing silent coercion")
    return out


def run_sanity_check_llm(config: ExperimentConfig) -> None:
    results_dir = ensure_results_dir()
    victim = QwenVictim(model_id=config.model_id)
    victim.initialize(results_dir / "qwen_startup_selfcheck.txt")
    rows = victim.sanity_check_llm(default_prompts()[:3])
    with (results_dir / "qwen_sanity_check.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    for r in rows:
        print(json.dumps(r, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check-llm", action="store_true")
    parser.add_argument("--debug-llm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    if args.sanity_check_llm:
        run_sanity_check_llm(config)
        return

    results_dir = ensure_results_dir()
    toy_df = run_toy_family(config)
    qwen_df = run_qwen_family(config, debug_llm=args.debug_llm)
    raw_df = pd.concat([toy_df, qwen_df], ignore_index=True)
    raw_df.to_csv(results_dir / "results_raw.csv", index=False)

    grouped = (
        raw_df.groupby(["source", "interface", "budget"], dropna=False)
        .agg(
            agreement_mean=("agreement", "mean"),
            agreement_std=("agreement", "std"),
            kl_mean=("kl_divergence", "mean"),
            kl_std=("kl_divergence", "std"),
            valid_kl_rows=("kl_divergence", lambda s: int(s.notna().sum())),
            invalid_kl_rows=("kl_divergence", lambda s: int(s.isna().sum())),
            rows=("kl_divergence", "size"),
        )
        .reset_index()
    )

    if (grouped["valid_kl_rows"] + grouped["invalid_kl_rows"] != grouped["rows"]).any():
        raise RuntimeError("Summary accounting mismatch")
    grouped.to_csv(results_dir / "results_summary.csv", index=False)


if __name__ == "__main__":
    main()
