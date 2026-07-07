# Datasets

Raw and processed data are **not committed** (see `.gitignore`). This document is the
authoritative pointer to the exact datasets, splits, and sizes used, so results can be
reproduced. Use `python scripts/prepare_data.py --config <cfg>` to download/stream and
cache the small subsets defined by the config's `data.mixture` and
`data.max_samples_per_modality`.

## Multi-task mixture (default `configs/train_small.yaml`)

| Modality | Dataset(s) (HuggingFace id) | Default mix weight | Notes |
|---|---|---|---|
| Language | `wikipedia` (subset) and/or `allenai/c4` (slice) | 0.40 | streamed; capped subset |
| Code     | `codeparrot/codeparrot-clean` (Python) or `bigcode/the-stack-v2` (Python) | 0.25 | Python only |
| Math     | `openai/gsm8k`, `hendrycks/competition_math` (MATH) | 0.20 | question + step-by-step solution |
| Logic    | `lucasmccabe/logiqa`, ReClor | 0.15 | multiple-choice → QA text |

> Exact dataset revisions/configs are pinned inside `scripts/prepare_data.py` and logged
> to the run directory at preparation time.

## Held-out / stretch evaluation
- University exams (SAT / GRE / GMAT / ENS) — held-out evaluation only.
- VQA (vision) — only if the optional vision-language extension (Phase 9) is built.

## Formatting
QA / reasoning datasets are converted to a uniform `prompt + answer` text template and
trained with a loss mask on the prompt (train on the completion only). Each example is
prefixed with a modality tag: `<|lang|>`, `<|code|>`, `<|logic|>`, `<|math|>`.

## Preparation workflow
```bash
# downloads a capped subsample of each modality → data/raw/<modality>.jsonl
python scripts/prepare_data.py --config configs/train_small.yaml
# smaller subsample:
python scripts/prepare_data.py --config configs/train_small.yaml --set data.max_samples_per_modality=300
```
Each line of `data/raw/<modality>.jsonl` is one `{"modality", "prompt", "completion"}`
example (uniform schema; `prompt=""` for plain text, question for QA). Verified locally on a
300/modality subsample (1200 examples) — full pipeline (mixture → encode → collate → model)
tested in `tests/test_pipeline_real.py`.

## Directory contents
- `raw/`        — prepared JSONL subsamples (gitignored)
- `processed/`  — tokenized / packed shards (gitignored)
- `hf_cache/`   — HuggingFace datasets streaming cache (gitignored)
