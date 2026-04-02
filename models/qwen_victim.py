"""Qwen2-0.5B victim wrapper with strict token dtype handling and verbose diagnostics."""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import mlx.core as mx
import mlx_lm
from huggingface_hub import snapshot_download
from mlx_lm import load

QWEN_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"


class QwenStartupError(RuntimeError):
    """Raised when Qwen startup checks fail."""


@dataclass
class QwenArtifacts:
    model_id: str
    cache_dir: str
    tokenizer_class: str
    vocab_size: int


SHORT_PROMPTS = [
    "Hello world.",
    "What is two plus two?",
    "Name a color.",
]


def _ensure_cache_and_download(model_id: str, cache_dir: Path, token: str | None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[QWEN] Ensuring model exists in cache: model_id={model_id} cache_dir={cache_dir}")
    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            token=token,
            local_files_only=False,
            resume_download=True,
        )
    except Exception as exc:  # noqa: BLE001
        auth_state = "enabled" if token else "disabled"
        raise QwenStartupError(
            "Failed to download Qwen from Hugging Face. "
            f"Authentication is {auth_state}. "
            "Set HF_TOKEN in GitHub Actions secrets if this is a rate-limit/auth issue. "
            f"Original error: {exc}"
        ) from exc


def safe_tokenize(tokenizer, prompt: str) -> list[int]:
    tokens = tokenizer.encode(prompt)
    if not isinstance(tokens, list):
        tokens = list(tokens)
    if not all(isinstance(tok, int) for tok in tokens):
        raise QwenStartupError(f"Tokenizer returned non-int token ids: {tokens!r}")
    return tokens


def tokens_to_mx_array(tokens: list[int]) -> tuple[mx.array, str]:
    # Must remain python ints until this conversion.
    arr = mx.array([tokens], dtype=mx.int32)
    dtype_name = str(arr.dtype)
    return arr, dtype_name


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    denom = np.sum(exp)
    if not np.isfinite(denom) or denom <= 0:
        raise QwenStartupError("Invalid softmax denominator while converting logits.")
    probs = exp / denom
    return probs


def model_next_token_probs(model, tokenizer, prompt: str) -> dict:
    token_ids = safe_tokenize(tokenizer, prompt)
    token_array, dtype_name = tokens_to_mx_array(token_ids)
    logits = model(token_array)
    logits_np = np.array(logits)
    if logits_np.ndim != 3:
        raise QwenStartupError(f"Unexpected logits ndim={logits_np.ndim}, expected 3")
    next_logits = logits_np[0, -1, :]
    probs = softmax_np(next_logits)
    return {
        "prompt": prompt,
        "token_ids": token_ids,
        "token_ids_type": type(token_ids).__name__,
        "token_array_dtype": dtype_name,
        "logits_shape": list(logits_np.shape),
        "vocab_size": int(probs.shape[0]),
        "probs": probs,
    }


def load_qwen(model_id: str = QWEN_MODEL_ID, cache_dir: str | None = None):
    cache_root = Path(cache_dir or os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    token = os.environ.get("HF_TOKEN")
    _ensure_cache_and_download(model_id=model_id, cache_dir=cache_root, token=token)
    print("[QWEN] Loading model via mlx_lm.load ...")
    try:
        model, tokenizer = load(model_id, tokenizer_config={"trust_remote_code": True})
    except Exception as exc:  # noqa: BLE001
        raise QwenStartupError(f"mlx_lm.load failed for {model_id}: {exc}") from exc
    return model, tokenizer, cache_root


def qwen_startup_selfcheck(output_file: Path, model_id: str = QWEN_MODEL_ID, cache_dir: str | None = None) -> QwenArtifacts:
    token = os.environ.get("HF_TOKEN")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer, cache_root = load_qwen(model_id=model_id, cache_dir=cache_dir)

    sanity = model_next_token_probs(model, tokenizer, SHORT_PROMPTS[0])
    vocab_size = int(sanity["vocab_size"])

    text = {
        "python_version": sys.version,
        "macos_version": platform.platform(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
        "model_id": model_id,
        "cache_dir": str(cache_root),
        "hf_token_present": bool(token),
        "hf_authenticated_download_enabled": bool(token),
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocabulary_size": vocab_size,
        "tiny_forward_pass": {
            "prompt": SHORT_PROMPTS[0],
            "token_count": len(sanity["token_ids"]),
            "token_ids_type": sanity["token_ids_type"],
            "mlx_dtype": sanity["token_array_dtype"],
            "logits_shape": sanity["logits_shape"],
            "probs_sum": float(np.sum(sanity["probs"])),
        },
    }
    output_file.write_text(json.dumps(text, indent=2), encoding="utf-8")
    print(f"[QWEN] Wrote startup self-check to {output_file}")

    return QwenArtifacts(
        model_id=model_id,
        cache_dir=str(cache_root),
        tokenizer_class=tokenizer.__class__.__name__,
        vocab_size=vocab_size,
    )


def sanity_check_llm(output_file: Path, prompts: Iterable[str] | None = None) -> None:
    prompts = list(prompts or SHORT_PROMPTS)
    model, tokenizer, _cache_root = load_qwen()

    lines = []
    for i, prompt in enumerate(prompts):
        data = model_next_token_probs(model, tokenizer, prompt)
        prob_sum = float(np.sum(data["probs"]))
        is_valid = np.isclose(prob_sum, 1.0, atol=1e-6)
        msg = {
            "prompt_index": i,
            "prompt": prompt,
            "token_type": data["token_ids_type"],
            "token_length": len(data["token_ids"]),
            "mlx_dtype": data["token_array_dtype"],
            "logits_shape": data["logits_shape"],
            "vocab_size": data["vocab_size"],
            "prob_sum": prob_sum,
            "prob_sum_ok": bool(is_valid),
        }
        if not is_valid:
            raise QwenStartupError(f"Probabilities do not sum to 1 for prompt {i}: {prob_sum}")
        lines.append(msg)
        print(f"[QWEN-SANITY] {json.dumps(msg)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    print(f"[QWEN] Sanity-check artifact written to {output_file}")
