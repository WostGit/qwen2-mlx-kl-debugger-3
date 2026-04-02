# qwen2-mlx-kl-debugger

A fully self-contained research repo for black-box model extraction experiments on **GitHub Actions macOS runners** using:

- `python`
- `mlx`
- `mlx-lm`
- tiny victim model: `Qwen/Qwen2-0.5B-Instruct`

This repository includes exactly two experiment families:

1. **Toy baseline** (always valid dense probability vectors, so agreement and KL are always well-defined).
2. **Qwen next-token extraction** on a fixed short prompt set for three API interfaces: `argmax`, `topk`, `probs`.

---

## Repository structure

- `models/` – victim model wrappers (toy and Qwen)
- `attacks/` – student extraction logic
- `experiments/` – experiment definitions and metrics
- `scripts/` – stage-specific runners (self-check, sanity-check, full day1, summary, plotting)
- `results/` – output artifacts
- `.github/workflows/` – macOS CI workflow
- `run_all.py` – top-level entrypoint

---

## Dependencies

Only these dependencies are used:

- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `mlx`
- `mlx-lm`
- `huggingface_hub`

No `transformers` dependency is used.

---

## GitHub Actions + HF_TOKEN setup (exact steps)

1. Open your GitHub repository.
2. Go to **Settings → Secrets and variables → Actions**.
3. Click **New repository secret**.
4. Set **Name** = `HF_TOKEN`.
5. Set **Secret** = your Hugging Face access token.
6. Save.

Workflow behavior:

- `HF_TOKEN` is passed via environment variable to model download code.
- Logs print whether auth is enabled (`YES`/`NO`) but never print the token value.
- The workflow caches `HF_HOME` to reuse downloaded model files across runs.
- If download/auth fails, the run exits with a clear error and does not continue ambiguously.

---

## Mandatory Qwen startup self-check

Before budget sweeps, run:

```bash
python run_all.py --qwen-startup-selfcheck
```

This writes `results/qwen_startup_selfcheck.txt` including:

- Python version
- macOS version
- MLX version
- mlx-lm version
- model id
- cache directory
- whether `HF_TOKEN` is present
- whether authenticated HF download is enabled
- tokenizer class
- vocabulary size
- tiny forward-pass sanity check on short prompts

---

## Dedicated LLM sanity-check mode

Run:

```bash
python run_all.py --sanity-check-llm
```

This mode:

- tokenizes 3 short prompts
- prints Python type and length of token id lists
- converts to MLX arrays with explicit signed int dtype
- verifies array shape assumptions
- runs one forward pass
- checks logits and vocabulary dimension
- computes next-token softmax and verifies sum≈1
- exits early (does not run full experiments)

---

## How token IDs are converted (dtype/buffer bug prevention)

To avoid MLX PEP 3118 / buffer format failures (e.g., `B` item-size mismatch):

1. Tokenizer output is immediately converted to a **plain Python `list[int]`**.
2. No raw bytes / uint8 / memoryview / binary-buffer construction is used.
3. Final conversion is done explicitly with signed MLX integer dtype:
   - `mx.array(token_ids, dtype=mx.int32)` (or int64)

This avoids implicit binary-buffer dtype coercions.

---

## Metrics: agreement and KL (exact definition)

For each evaluation prompt:

1. Victim returns full next-token logits from MLX.
2. Victim full dense softmax over full vocab is computed.
3. Interface targets are derived from this same dense victim vector:
   - `argmax`: one-hot at top token
   - `topk`: keep top-k probs and renormalize
   - `probs`: unchanged full softmax
4. Student always outputs a **full dense distribution on the same vocabulary**.
5. Evaluation computes:
   - `agreement`: top-1 token match between victim full softmax and student full probs
   - `kl_divergence`: `KL(victim || student)` on identical token support

Rules enforced:

- no KL from labels only
- no mismatched vocabulary KL
- no silent NaN-to-zero coercion
- workflow fails if >1% Qwen KL rows are invalid

---

## Invalid Qwen row conditions

A Qwen metric row is invalid if any of the following occur:

- victim/student probability shape mismatch
- NaN or inf in either vector
- probability sums not ~1
- no valid KL rows for a run
- run-level invalid fraction exceeds 1%

Invalid rows and reasons are logged in `results/qwen_nan_report.csv`.

Vocabulary checks are logged in `results/qwen_vocab_alignment_check.csv`.

---

## Verbose debug mode

Run full experiments with detailed diagnostics:

```bash
python run_all.py --qwen-startup-selfcheck --debug-llm
```

For the first 10 eval prompts per interface/seed, artifacts include per-example JSONL and CSV fields:

- prompt id / prompt text
- tokenized prompt ids
- Python type before MLX conversion
- MLX dtype after conversion
- victim and student top-10 token ids/probabilities
- sums of full probability vectors
- NaN/inf flags
- vocab size + alignment flag
- exact per-example KL
- human-readable skip reason (if invalid)

---

## Day 1 experiment plan implemented

1. Budget sweep (`64,128,256,512,1024`) for both toy and Qwen.
2. Fixed-budget Qwen top-k sweep at budget `256` for:
   - `argmax`, `top2`, `top3`, `top5`, `probs`
3. Multi-seed summary with mean/std.
4. Plots:
   - `agreement_vs_budget.png`
   - `kl_vs_budget.png`
   - `qwen_topk_kl.png`

---

## Artifacts to inspect first when Qwen init fails

1. `results/qwen_startup_selfcheck.txt`
2. GitHub Action log lines from self-check + sanity-check steps
3. `results/qwen_nan_report.csv`
4. `results/qwen_vocab_alignment_check.csv`

These files are sufficient for diagnosis from workflow artifacts alone.
