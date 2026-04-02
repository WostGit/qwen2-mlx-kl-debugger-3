# qwen2-mlx-kl-debugger

A self-contained research repository for black-box model extraction experiments on **GitHub Actions macOS runners** with Python + MLX + mlx-lm.

## Scope

This repo intentionally includes exactly two experiment families:

1. **Toy baseline** (always dense valid probability vectors; always valid agreement + KL).
2. **Qwen2-0.5B one-step next-token extraction** over fixed short prompts using victim interfaces:
   - `argmax`
   - `topk` (specifically `top2`, `top3`, `top5`)
   - `probs`

## Dependency policy

Installed from a single `requirements.txt`:

- numpy
- pandas
- matplotlib
- pyyaml
- mlx
- mlx-lm
- huggingface_hub

`transformers` is not used.

## Repository layout

- `models/` Qwen loading, token conversion, startup + sanity checks
- `attacks/` student estimator logic
- `experiments/` toy and Qwen experiment runners + metrics
- `scripts/` summary, plotting, and stage wrappers
- `results/` generated outputs
- `.github/workflows/` macOS-only workflow
- `run_all.py` top-level orchestrator

## Hugging Face authentication in GitHub Actions

1. In GitHub repo settings, create secret: `HF_TOKEN`.
2. Workflow passes it via env: `HF_TOKEN: ${{ secrets.HF_TOKEN }}`.
3. Download code passes token into `snapshot_download(..., token=token)`.
4. Logs show only authentication state (`enabled/disabled`), never token value.
5. If download/auth fails, run exits with a clear startup error and does **not** continue.

## Token-ID conversion rules (dtype/buffer safety)

To avoid MLX token buffer/dtype failures (including PEP 3118 `B` format item-size mismatch):

- Token IDs remain plain Python `list[int]` after tokenization.
- Final conversion only happens immediately before model call.
- Conversion is explicit: `mx.array([tokens], dtype=mx.int32)`.
- Forbidden paths: bytes, `uint8` arrays, memoryviews, raw binary buffers, implicit buffer conversion.

## Mandatory startup checks

### 1) Startup self-check artifact (`qwen_startup_selfcheck.txt`)

Runs before budget sweeps and logs:

- Python version
- macOS version
- MLX version
- mlx-lm version
- model id
- cache directory
- HF token present flag
- authenticated download enabled flag
- tokenizer class
- vocabulary size
- tiny forward-pass sanity result

### 2) `--sanity-check-llm`

Tokenizes 3 short prompts and verifies:

- Python type and length of token list
- MLX dtype after conversion
- array shape
- logits shape and vocabulary dimension
- next-token softmax sums to 1 within tolerance

Exits before full experiment.

## Experiment details

### Day-1 sequence

1. Budget sweep with budgets: `64, 128, 256, 512, 1024` for toy and Qwen.
2. Fixed-budget (`256`) Qwen top-k sweep across: `argmax, top2, top3, top5, probs`.
3. Multi-seed summary with mean/std.
4. Plots:
   - `agreement_vs_budget.png`
   - `kl_vs_budget.png`
   - `qwen_topk_kl.png`

### Metrics and validity

- **Agreement**: top-1 token match between victim full distribution argmax and student full distribution argmax.
- **KL divergence**: real KL between student full dense distribution and victim full dense softmax over the *same vocabulary support*.

Rules:

- Never compute KL from labels only.
- Never compare mismatched vocabularies.
- Never silently coerce NaN to zero.
- Keep raw NaNs in `results_raw.csv`.
- `results_summary.csv` reports valid/invalid counts.
- Workflow fails if invalid KL rate for Qwen exceeds 1%.

## Debug artifacts

`--debug-llm` emits verbose artifacts for at least first 10 prompts per interface/seed:

- `qwen_debug_examples.jsonl`
- `qwen_debug_examples.csv`
- `qwen_nan_report.csv`
- `qwen_vocab_alignment_check.csv`

These include prompt text, token IDs, type and dtype metadata, top-10 victim/student tokens and probabilities, vector sums, NaN/inf flags, vocab alignment, exact KL, and skip reason.

## Failure triage order

If Qwen fails to initialize, inspect in this exact order:

1. `results/qwen_startup_selfcheck.txt`
2. workflow logs from the "Qwen startup self-check" step
3. `results/qwen_sanity_check.jsonl`
4. `results/qwen_nan_report.csv`

## Local usage

```bash
pip install -r requirements.txt
python run_all.py --qwen-selfcheck --results-dir results
python run_all.py --sanity-check-llm --results-dir results
python run_all.py --debug-llm --results-dir results
```
