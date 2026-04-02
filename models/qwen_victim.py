"""Utilities for loading and querying Qwen2-0.5B via MLX/MLX-LM with defensive checks."""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm import load

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"


@dataclass
class QwenContext:
    model: Any
    tokenizer: Any
    model_id: str
    cache_dir: str


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    ex = np.exp(shifted)
    denom = np.sum(ex)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Invalid softmax denominator: {denom}")
    probs = ex / denom
    return probs.astype(np.float64)


def _format_bool(x: bool) -> str:
    return "YES" if x else "NO"


def _tokenize_to_python_int_list(tokenizer: Any, prompt: str) -> list[int]:
    # Keep raw tokenized output as plain Python integers until final mx.array conversion.
    token_ids = tokenizer.encode(prompt)
    py_ids = [int(t) for t in token_ids]
    return py_ids


def _to_mx_int_array(token_ids: list[int], dtype: Any = mx.int32) -> Any:
    arr = mx.array(token_ids, dtype=dtype)
    if str(arr.dtype) not in {str(mx.int32), str(mx.int64)}:
        raise TypeError(f"Token dtype must be signed int32/int64, got {arr.dtype}")
    return arr


def _forward_logits(model: Any, input_ids_1d: Any) -> Any:
    # Expect [seq] -> [1, seq] for batch dimension.
    batched = mx.expand_dims(input_ids_1d, axis=0)
    out = model(batched)
    if isinstance(out, tuple):
        logits = out[0]
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out
    return logits


def _last_token_logits_numpy(logits: Any) -> np.ndarray:
    np_logits = np.array(logits)
    if np_logits.ndim != 3:
        raise ValueError(f"Expected logits rank 3 [batch, seq, vocab], got shape {np_logits.shape}")
    vec = np_logits[0, -1, :]
    if vec.ndim != 1:
        raise ValueError(f"Expected next-token logits vector rank 1, got shape {vec.shape}")
    return vec.astype(np.float64)


def ensure_download(model_id: str, cache_dir: str, token: str | None, allow_auth: bool) -> None:
    print(f"[QWEN] Ensuring model files in cache. model_id={model_id} cache_dir={cache_dir}")
    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=False,
            token=token if allow_auth else None,
        )
    except Exception as exc:  # noqa: BLE001
        auth_msg = "enabled" if allow_auth else "disabled"
        raise RuntimeError(
            "Failed to download Qwen model from Hugging Face. "
            f"Authentication is {auth_msg}. "
            "Provide HF_TOKEN secret or reduce rate-limited anonymous access. "
            f"Original error: {exc}"
        ) from exc


def load_qwen(model_id: str = MODEL_ID, cache_dir: str | None = None) -> QwenContext:
    cache_dir = cache_dir or os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    hf_token = os.environ.get("HF_TOKEN")
    allow_auth = bool(hf_token)
    print(f"[QWEN] HF authentication enabled: {_format_bool(allow_auth)}")
    ensure_download(model_id=model_id, cache_dir=cache_dir, token=hf_token, allow_auth=allow_auth)
    try:
        model, tokenizer = load(model_id, tokenizer_config={"use_fast": False})
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model/tokenizer via mlx_lm.load for {model_id}: {exc}") from exc
    return QwenContext(model=model, tokenizer=tokenizer, model_id=model_id, cache_dir=cache_dir)


def qwen_next_token_probs(ctx: QwenContext, prompt: str) -> dict[str, Any]:
    py_ids = _tokenize_to_python_int_list(ctx.tokenizer, prompt)
    in_arr = _to_mx_int_array(py_ids, dtype=mx.int32)
    logits = _forward_logits(ctx.model, in_arr)
    vec = _last_token_logits_numpy(logits)
    probs = _softmax(vec)
    return {
        "token_ids": py_ids,
        "token_ids_python_type": str(type(py_ids)),
        "mlx_dtype": str(in_arr.dtype),
        "logits": vec,
        "probs": probs,
        "vocab_size": int(probs.shape[0]),
    }


def run_sanity_check(ctx: QwenContext) -> dict[str, Any]:
    prompts = ["Hello world", "The sky is", "2 + 2 ="]
    details: list[dict[str, Any]] = []
    for p in prompts:
        r = qwen_next_token_probs(ctx, p)
        probs = r["probs"]
        s = float(np.sum(probs))
        info = {
            "prompt": p,
            "token_list_python_type": r["token_ids_python_type"],
            "token_list_length": len(r["token_ids"]),
            "mlx_dtype": r["mlx_dtype"],
            "vocab_size": r["vocab_size"],
            "prob_sum": s,
            "sum_close_to_1": bool(abs(s - 1.0) < 1e-6),
        }
        details.append(info)
    return {"ok": all(d["sum_close_to_1"] for d in details), "details": details}


def write_startup_selfcheck(ctx: QwenContext, out_path: str | Path) -> None:
    out_path = Path(out_path)
    hf_token = os.environ.get("HF_TOKEN")
    auth_enabled = bool(hf_token)
    sanity = run_sanity_check(ctx)
    tokenizer_name = type(ctx.tokenizer).__name__
    vocab_size = getattr(ctx.tokenizer, "vocab_size", None)
    lines = [
        "QWEN STARTUP SELF-CHECK",
        f"python_version: {sys.version}",
        f"macos_version: {platform.platform()}",
        f"mlx_version: {getattr(mx, '__version__', 'unknown')}",
        f"mlx_lm_version: {__import__('mlx_lm').__version__ if hasattr(__import__('mlx_lm'), '__version__') else 'unknown'}",
        f"model_id: {ctx.model_id}",
        f"cache_dir: {ctx.cache_dir}",
        f"hf_token_present: {_format_bool(bool(hf_token))}",
        f"hf_auth_enabled: {_format_bool(auth_enabled)}",
        f"tokenizer_class: {tokenizer_name}",
        f"vocabulary_size: {vocab_size}",
        "sanity_check_result:",
        json.dumps(sanity, indent=2),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[QWEN] Wrote startup self-check to {out_path}")
    if not sanity["ok"]:
        raise RuntimeError("Qwen sanity check failed: probability sum check did not pass.")
