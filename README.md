# qwen2-mlx-kl-debugger

A self-contained research repo for black-box model extraction experiments that run on **GitHub Actions macOS runners** using **Python + MLX + mlx-lm** with **Qwen2-0.5B-Instruct** as the tiny-LLM victim.

## Repository layout

```text
models/
attacks/
experiments/
scripts/
results/
.github/workflows/
run_all.py
```

## Dependencies

Only these dependencies are used:

- numpy
- pandas
- matplotlib
- pyyaml
- mlx
- mlx-lm
- huggingface_hub

`transformers` is intentionally not used.

## Hugging Face authentication (HF_TOKEN)

Qwen downloads are done with `huggingface_hub.snapshot_download(...)` and explicitly pass `token=os.getenv("HF_TOKEN")`.

### Configure `HF_TOKEN` in GitHub Actions

1. Open your GitHub repo.
2. Go to **Settings → Secrets and variables → Actions**.
3. Click **New repository secret**.
4. Name: `HF_TOKEN`
5. Value: your Hugging Face access token.
6. Save.

The workflow exports `HF_TOKEN` into the job environment and logs only whether auth is enabled (`hf_token_present` and `hf_auth_enabled`), never the token itself.

If download/auth fails, startup self-check writes a clear failure reason to `results/qwen_startup_selfcheck.txt` and aborts.

## Startup self-check (mandatory)

Before any Qwen sweep runs, `QwenVictim.initialize()` writes `results/qwen_startup_selfcheck.txt` with:

- Python version
- macOS/platform version
- MLX version
- mlx-lm version
- model id
- cache directory
- HF token presence
- whether authenticated download is enabled
- tokenizer class
- vocabulary size
- tiny one-prompt forward pass result

## Avoiding MLX dtype / PEP 3118 buffer bugs

To avoid token dtype/buffer conversion failures (including PEP 3118 style `itemsize` mismatches):

1. Prompt token ids are kept as plain **Python `list[int]`**.
2. The code validates every id is an `int`.
3. Final tensor conversion is explicit: `mx.array(token_ids, dtype=mx.int32)` (signed integer).
4. Conversion from bytes, uint8 buffers, memoryviews, and implicit binary buffers is never used.

## Dedicated `--sanity-check-llm` mode

Run:

```bash
python run_all.py --sanity-check-llm
```

It tokenizes 3 short prompts and logs/verifies:

- Python token list type and length
- MLX dtype after conversion
- tensor input shape
- forward-pass logits shape
- logits vocabulary dimension
- next-token softmax sum-to-one check

It exits before full experiment sweeps.

## Experiment families (exactly two)

## 1) Toy baseline

A toy victim always emits valid dense probabilities. This ensures valid agreement/KL metrics and acts as a baseline sanity family.

## 2) Qwen one-step next-token extraction

For a fixed short prompt set, Qwen next-token experiments compare three victim interfaces:

- `argmax`
- `topk`
- `probs`

The victim full dense softmax is always computed first, then interface-specific targets are derived from the same support:

- `argmax`: one-hot at argmax token
- `topk`: keep top-k victim probabilities then renormalize
- `probs`: full softmax unchanged

The student always predicts a **full dense distribution over the same vocabulary**.

### Metrics and validity

- **agreement**: top-1 token equality between victim dense softmax and student dense distribution.
- **kl_divergence**: real KL `KL(victim || student)` over the full aligned token support.

Rules enforced:

- no label-only KL
- no vocabulary mismatch KL
- no NaN coercion to zero
- if >1% rows in a Qwen setting are invalid, mark failure for that setting

## Debugging and transparency artifacts

Run with verbose debugging:

```bash
python run_all.py --debug-llm
```

Artifacts include:

- `results/qwen_debug_examples.jsonl`
- `results/qwen_debug_examples.csv`
- `results/qwen_nan_report.csv`
- `results/qwen_vocab_alignment_check.csv`
- `results/results_raw.csv` (raw NaN preserved)
- `results/results_summary.csv` (valid/invalid KL row counts)
- `results/agreement_vs_budget.png`
- `results/kl_vs_budget.png`
- `results/qwen_topk_kl.png`

For each debug example row (first 10 prompts/interface/seed), logging includes prompt id/text, token ids, token types/dtypes, top-10 tokens/probabilities for victim and student, probability sums, NaN/inf flags, vocab size/alignment, per-example KL, and invalid reason.

## Day 1 sweep plan implemented

1. Budget sweep over `64,128,256,512,1024` queries for toy and Qwen with `source, interface, budget, seed, agreement, kl_divergence, validity flags`.
2. Fixed-budget Qwen top-k sweep over `argmax, top2, top3, top5, probs`.
3. Multi-seed summary with mean/std.
4. Plot generation:
   - `agreement_vs_budget.png`
   - `kl_vs_budget.png`
   - `qwen_topk_kl.png`

## GitHub Actions behavior

Workflow: `.github/workflows/macos_research.yml`

- macOS runner only
- install from single `requirements.txt`
- cache pip and model downloads
- deterministic environment prints
- mandatory startup self-check
- mandatory sanity-check-llm
- full experiment run with debug logs
- upload CSV/JSONL/TXT/PNG artifacts

## Local usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# startup + sanity
python run_all.py --sanity-check-llm

# full run
python run_all.py --debug-llm
```

## If Qwen initialization fails, inspect first

1. `results/qwen_startup_selfcheck.txt`
2. GitHub Actions step logs for "Qwen startup self-check"
3. `results/qwen_sanity_check.json` (if generated)
4. `results/qwen_nan_report.csv`
