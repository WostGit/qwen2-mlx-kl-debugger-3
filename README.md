# qwen2-mlx-kl-debugger

A self-contained research repository for black-box model extraction experiments on **GitHub Actions macOS runners** using **Python + MLX + mlx-lm**.

## Repository layout

- `models/`: victim-model wrappers (`toy` and `Qwen2-0.5B`).
- `attacks/`: student model + interface-target conversions.
- `experiments/`: budget sweeps and top-k sweeps.
- `scripts/`: startup self-check, sanity check, summary, and plotting.
- `results/`: generated outputs and artifacts.
- `.github/workflows/`: macOS CI pipeline.
- `run_all.py`: top-level Day 1 orchestrator.

## Dependencies

Only these packages are used:

- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `mlx`
- `mlx-lm`
- `huggingface_hub`

`transformers` is intentionally not used.

## HF_TOKEN setup in GitHub Actions

1. Open your GitHub repository.
2. Go to **Settings → Secrets and variables → Actions**.
3. Create a new repository secret named **`HF_TOKEN`**.
4. Paste your Hugging Face user access token value.
5. Run the workflow `.github/workflows/macos_qwen_extraction.yml`.

The workflow passes `HF_TOKEN` to Python and logs:

- whether a token is present (`hf_token_present`), and
- whether authenticated download is enabled (`authenticated_hf_download`).

The token value itself is never printed.

## Mandatory Qwen startup self-check

Before any budget sweep starts, the workflow runs:

```bash
python scripts/selfcheck_qwen.py --output results/qwen_startup_selfcheck.txt
```

It writes `qwen_startup_selfcheck.txt` with:

- Python version
- macOS version
- MLX version
- mlx-lm version
- model id
- cache directory
- HF token present flag
- authenticated HF download flag
- tokenizer class
- vocabulary size
- tiny forward-pass sanity result on one short prompt

If download/auth fails, execution stops with a clear error.

## Robust token-id conversion (dtype / PEP 3118 mitigation)

To avoid MLX token dtype/buffer failures (e.g., PEP 3118 `B` item-size mismatch), this repo enforces:

1. Tokenized prompt ids remain **plain Python `list[int]`** until final tensor creation.
2. Conversion happens only via:

```python
mx.array([token_ids], dtype=mx.int32)
```

3. We never build prompt tensors from raw bytes, numpy byte arrays, uint8 buffers, memoryviews, or implicitly typed binary buffers.

## `--sanity-check-llm` mode

Run:

```bash
python run_all.py --sanity-check-llm
```

This mode tokenizes 3 short prompts and logs:

- python type + length of token list
- MLX dtype after conversion
- resulting array shape
- single forward pass
- logits shape and vocabulary dimension
- next-token softmax sum-to-1 check

Then it exits before full experiments.

## Experiment families (exactly two)

1. **Toy baseline** (`models/toy_victim.py`)
   - Always returns valid dense probability vectors.
   - Always supports valid top-1 agreement and KL metrics.

2. **Qwen2-0.5B one-step next-token experiments**
   - Fixed short prompt set.
   - Three victim API interfaces: `argmax`, `topk`, `probs`.
   - Full victim logits → dense full-vocab softmax first.
   - Interface targets are derived from that same dense vector:
     - `argmax`: one-hot at top token
     - `topk`: keep top-k probs and renormalize
     - `probs`: full softmax unchanged

## Metrics (strict)

- Student always emits a **full dense distribution** over the same vocabulary.
- Top-1 agreement compares `argmax(student)` vs `argmax(victim_full_softmax)`.
- `kl_divergence` is computed only using two dense aligned vectors over identical token support.

Rules:

- never compute KL from labels only,
- never compare mismatched vocabularies,
- never silently coerce NaN to zero,
- fail workflow if invalid Qwen KL rate exceeds 1%.

## Verbose debug artifacts

With `--debug-llm`, the Qwen path writes per-example JSONL for at least first 10 eval prompts per interface/seed with:

- prompt id/text
- tokenized ids
- Python type before MLX conversion
- MLX dtype after conversion
- victim/student top-10 ids/probabilities
- full-vector sums
- NaN/Inf flags
- vocabulary size
- vocab alignment result
- exact per-example KL
- human-readable KL skip reason

Also generated:

- `results/qwen_nan_report.csv` (every invalid KL row and reason)
- `results/qwen_vocab_alignment_check.csv` (vocab-size/index alignment checks)
- `results/results_raw.csv` (raw NaNs preserved)
- `results/results_summary.csv` (valid/invalid KL counts)

## Day 1 experiment plan implemented

- Budget sweep: `64, 128, 256, 512, 1024` for toy + qwen.
- Logged fields: `source, interface, budget, seed, agreement, kl_divergence, kl_valid`.
- Fixed-budget Qwen top-k sweep over: `argmax, top2, top3, top5, probs`.
- Multi-seed summary with mean and std.
- Plots:
  - `agreement_vs_budget.png`
  - `kl_vs_budget.png`
  - `qwen_topk_kl.png`

## Run locally

```bash
pip install -r requirements.txt
python scripts/selfcheck_qwen.py --output results/qwen_startup_selfcheck.txt
python run_all.py --sanity-check-llm
python run_all.py --multi-seed 0,1,2 --debug-llm
```

## If Qwen initialization fails, inspect these artifacts first

1. `results/qwen_startup_selfcheck.txt`
2. GitHub Actions job logs for `Qwen startup self-check`
3. `results/qwen_nan_report.csv` (if generated)
4. `results/qwen_vocab_alignment_check.csv`
