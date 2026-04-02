from __future__ import annotations

import json
import os
import platform
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm import load

from experiments.metrics import softmax


@dataclass
class QwenRuntime:
    model: Any
    tokenizer: Any
    model_id: str
    cache_dir: str
    hf_token_present: bool
    hf_auth_enabled: bool


class QwenVictim:
    def __init__(self, model_id: str, cache_dir: str = ".hf_cache") -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.runtime: Optional[QwenRuntime] = None

    def initialize(self, artifact_path: Path) -> QwenRuntime:
        hf_token = os.getenv("HF_TOKEN")
        hf_auth_enabled = bool(hf_token)
        lines: List[str] = []
        lines.append("=== Qwen startup self-check ===")
        lines.append(f"python_version={platform.python_version()}")
        lines.append(f"macos_version={platform.platform()}")
        lines.append(f"mlx_version={getattr(mx, '__version__', 'unknown')}")

        try:
            import mlx_lm  # pylint: disable=import-outside-toplevel

            lines.append(f"mlx_lm_version={getattr(mlx_lm, '__version__', 'unknown')}")
        except Exception as exc:  # pragma: no cover
            lines.append(f"mlx_lm_version_error={exc}")

        lines.append(f"model_id={self.model_id}")
        lines.append(f"cache_dir={self.cache_dir}")
        lines.append(f"hf_token_present={bool(hf_token)}")
        lines.append(f"hf_auth_enabled={hf_auth_enabled}")

        try:
            snapshot_download(
                repo_id=self.model_id,
                cache_dir=self.cache_dir,
                token=hf_token,
                local_files_only=False,
            )
            lines.append("hf_snapshot_download=success")
        except Exception as exc:
            lines.append(f"hf_snapshot_download=failed: {exc}")
            lines.append("FAIL: Hugging Face download/auth failed. Set HF_TOKEN in GitHub secrets.")
            artifact_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            raise RuntimeError("Qwen initialization failed due to Hugging Face access error") from exc

        try:
            model, tokenizer = load(self.model_id)
            vocab_size = self._tokenizer_vocab_size(tokenizer)
            lines.append(f"tokenizer_class={tokenizer.__class__.__name__}")
            lines.append(f"vocab_size={vocab_size}")
            probs = self.next_token_probs_with_runtime(model, tokenizer, "hello world")
            lines.append(f"tiny_forward_pass=success; probs_sum={float(probs.sum()):.10f}")
        except Exception as exc:
            lines.append(f"tiny_forward_pass=failed: {exc}")
            lines.append(traceback.format_exc())
            artifact_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            raise

        artifact_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.runtime = QwenRuntime(
            model=model,
            tokenizer=tokenizer,
            model_id=self.model_id,
            cache_dir=self.cache_dir,
            hf_token_present=bool(hf_token),
            hf_auth_enabled=hf_auth_enabled,
        )
        return self.runtime

    def _tokenizer_vocab_size(self, tokenizer: Any) -> int:
        if hasattr(tokenizer, "vocab_size"):
            return int(tokenizer.vocab_size)
        if hasattr(tokenizer, "get_vocab"):
            return int(len(tokenizer.get_vocab()))
        raise ValueError("Unable to determine tokenizer vocab size")

    def tokenize_to_python_int_list(self, tokenizer: Any, prompt: str) -> List[int]:
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(prompt)
        else:
            raise ValueError("Tokenizer lacks encode method")
        if not isinstance(ids, list):
            ids = list(ids)
        if not all(isinstance(x, int) for x in ids):
            raise TypeError("Token ids must be plain Python ints before MLX conversion")
        return ids

    def token_ids_to_mx(self, token_ids: List[int], dtype: Any = mx.int32) -> mx.array:
        arr = mx.array(token_ids, dtype=dtype)
        if str(arr.dtype) not in {"int32", "int64"}:
            raise TypeError(f"Token tensor dtype must be signed integer; got {arr.dtype}")
        return arr[None, :]

    def next_token_probs_with_runtime(self, model: Any, tokenizer: Any, prompt: str) -> np.ndarray:
        token_ids = self.tokenize_to_python_int_list(tokenizer, prompt)
        x = self.token_ids_to_mx(token_ids, dtype=mx.int32)
        logits = model(x)
        last_logits = logits[:, -1, :]
        np_logits = np.array(last_logits)[0]
        return softmax(np_logits)

    def next_token_probs(self, prompt: str) -> np.ndarray:
        if self.runtime is None:
            raise RuntimeError("QwenVictim not initialized")
        return self.next_token_probs_with_runtime(self.runtime.model, self.runtime.tokenizer, prompt)

    def sanity_check_llm(self, prompts: Iterable[str]) -> List[Dict[str, Any]]:
        if self.runtime is None:
            raise RuntimeError("Initialize model before sanity check")
        rows: List[Dict[str, Any]] = []
        vocab = self._tokenizer_vocab_size(self.runtime.tokenizer)
        for idx, prompt in enumerate(prompts):
            token_ids = self.tokenize_to_python_int_list(self.runtime.tokenizer, prompt)
            x = self.token_ids_to_mx(token_ids, dtype=mx.int32)
            logits = self.runtime.model(x)
            last_logits = np.array(logits[:, -1, :])[0]
            probs = softmax(last_logits)
            row = {
                "prompt_id": idx,
                "prompt": prompt,
                "python_type": str(type(token_ids)),
                "token_list_length": len(token_ids),
                "mlx_dtype": str(x.dtype),
                "input_shape": tuple(int(v) for v in x.shape),
                "logits_shape": tuple(int(v) for v in logits.shape),
                "vocab_size": vocab,
                "prob_sum": float(probs.sum()),
                "prob_valid": bool(np.isfinite(probs).all() and abs(probs.sum() - 1.0) < 1e-6),
            }
            if row["logits_shape"][-1] != vocab:
                raise ValueError("Logits vocabulary dimension mismatch")
            rows.append(row)
        return rows


def interface_target_from_dense(victim_probs: np.ndarray, interface: str, topk_default: int = 5) -> np.ndarray:
    p = np.asarray(victim_probs, dtype=np.float64)
    if interface == "argmax":
        out = np.zeros_like(p)
        out[int(np.argmax(p))] = 1.0
        return out
    if interface == "probs":
        return p.copy()
    if interface == "topk":
        k = topk_default
    elif interface.startswith("top") and interface[3:].isdigit():
        k = int(interface[3:])
    else:
        raise ValueError(f"Unknown interface {interface}")
    idx = np.argsort(-p)[:k]
    out = np.zeros_like(p)
    out[idx] = p[idx]
    s = out.sum()
    if s <= 0:
        raise ValueError(f"Top-k renormalization failed for interface={interface}")
    out /= s
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
