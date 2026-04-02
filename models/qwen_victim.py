"""Qwen2-0.5B victim wrapper for MLX experiments."""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import mlx.core as mx
import numpy as np
from huggingface_hub import HfApi
from mlx_lm import load

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "huggingface")


@dataclass
class QwenConfig:
    model_id: str = MODEL_ID
    cache_dir: str = DEFAULT_CACHE_DIR
    hf_token: str | None = None


class QwenVictim:
    def __init__(self, cfg: QwenConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        token = self.cfg.hf_token or os.environ.get("HF_TOKEN")
        try:
            self.model, self.tokenizer = load(
                self.cfg.model_id,
                tokenizer_config={"trust_remote_code": True},
                token=token,
            )
        except Exception as exc:
            auth_enabled = bool(token)
            raise RuntimeError(
                "Failed to load Qwen model from Hugging Face. "
                f"model_id={self.cfg.model_id}, cache_dir={self.cfg.cache_dir}, "
                f"authenticated_hf_download={auth_enabled}. "
                "Verify HF_TOKEN is set in GitHub Actions secrets and has model access."
            ) from exc

    @staticmethod
    def _to_mx_token_tensor(token_ids: Sequence[int]) -> mx.array:
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)
        if not all(isinstance(x, int) for x in token_ids):
            raise TypeError("Token ids must be a Python list of integers before MLX conversion")
        return mx.array([token_ids], dtype=mx.int32)

    def tokenize(self, prompt: str) -> List[int]:
        token_ids = self.tokenizer.encode(prompt)
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)
        if not all(isinstance(x, int) for x in token_ids):
            raise TypeError("Tokenizer returned non-integer token ids")
        return token_ids

    def forward_logits(self, prompt: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        token_ids = self.tokenize(prompt)
        token_type = str(type(token_ids))
        x = self._to_mx_token_tensor(token_ids)
        outputs = self.model(x)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        # last-token logits
        last = logits[:, -1, :]
        probs = mx.softmax(last, axis=-1)
        probs_np = np.array(probs).reshape(-1)
        info = {
            "token_ids": token_ids,
            "token_ids_python_type": token_type,
            "mlx_dtype": str(x.dtype),
            "input_shape": tuple(x.shape),
            "logits_shape": tuple(last.shape),
            "vocab_size": int(last.shape[-1]),
            "prob_sum": float(probs_np.sum()),
            "nan": bool(np.isnan(probs_np).any()),
            "inf": bool(np.isinf(probs_np).any()),
        }
        return probs_np, info

    def startup_selfcheck(self, output_file: str) -> None:
        token = self.cfg.hf_token or os.environ.get("HF_TOKEN")
        auth_enabled = bool(token)
        auth_ok = False
        auth_msg = "HF_TOKEN missing"
        if auth_enabled:
            try:
                HfApi(token=token).whoami()
                auth_ok = True
                auth_msg = "Authenticated with Hugging Face"
            except Exception as exc:  # noqa: BLE001
                auth_msg = f"HF auth check failed: {exc}"

        prompt = "Hello"
        fp_ok = False
        fp_msg = "not-run"
        fp_info: Dict[str, Any] = {}
        try:
            probs, fp_info = self.forward_logits(prompt)
            fp_ok = bool(np.isfinite(probs).all() and abs(probs.sum() - 1.0) < 1e-4)
            fp_msg = "forward pass succeeded" if fp_ok else "forward pass produced invalid probabilities"
        except Exception as exc:  # noqa: BLE001
            fp_msg = f"forward pass failed: {exc}"

        vocab_size = None
        try:
            vocab_size = len(self.tokenizer)
        except Exception:
            vocab_size = fp_info.get("vocab_size")

        payload = {
            "python_version": sys.version,
            "macos_version": platform.platform(),
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "mlx_lm_version": _safe_version("mlx_lm"),
            "model_id": self.cfg.model_id,
            "cache_dir": self.cfg.cache_dir,
            "hf_token_present": auth_enabled,
            "authenticated_hf_download": auth_ok,
            "auth_message": auth_msg,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "vocabulary_size": vocab_size,
            "tiny_forward_pass_ok": fp_ok,
            "tiny_forward_pass_message": fp_msg,
            "tiny_forward_pass_info": fp_info,
        }

        Path(output_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not auth_ok:
            raise RuntimeError(
                "Qwen startup self-check authentication failed. "
                "Set HF_TOKEN in GitHub Actions secrets and ensure it has access."
            )
        if not fp_ok:
            raise RuntimeError("Qwen startup self-check forward pass failed. See artifact file for details.")


def _safe_version(module_name: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(module_name)
    except Exception:
        return "unknown"
