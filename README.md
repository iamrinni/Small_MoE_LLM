# Small Language Model with Sparse Mixture of Experts (Top-2 Gating)

A **small, sparse Mixture-of-Experts (MoE)** decoder-only language model built on the
**OLMoE** architecture, trained as a **multi-task** model across four text modalities:
**natural language, code, logic, and math**. Each token is routed to its **top-2 of 8
experts**; experts are GeLU MLPs; positional encoding is RoPE; training/inference run in
bfloat16.

> Course project — ACDL 2026. Full task spec: `claude/33-ACDL-2026-Project-small-language-model with MoE (1).pdf`.
> Step-by-step implementation plan: [`claude/instruction.md`](claude/instruction.md).

## Architecture at a glance

| Aspect | Choice |
|---|---|
| Base | HF `transformers` `OlmoeForCausalLM` (OLMoE), instantiated small |
| Experts / gating | 8 experts, **top-2** routing |
| Expert FFN | **GeLU MLP** (SwiGLU as ablation) |
| Positional encoding | **RoPE** (learnable as ablation) |
| Precision | **bfloat16** |
| Load balancing | aux load-balance loss + router z-loss |
| Routing analysis | ported from `allenai/OLMoE` (expert load, entropy, specialization) |

See [`claude/instruction.md`](claude/instruction.md) for the full design rationale and
locked decisions.

## Repository layout

```
configs/     YAML model + training configs (model_small, train_small, smoke)
scripts/     train / evaluate / prepare_data / analyze_routing + setup.sh
src/         model, data, training, eval, utils packages
data/        dataset preparation scripts + pointers (raw data is NOT committed)
notebooks/   EDA + routing-analysis + results notebooks
report/      LaTeX sources, figures, final PDF
tests/       unit / scaffold tests
```

## Setup (clean environment)

Requires **Python ≥ 3.10**.

```bash
# 1. (recommended) create a virtual env
python -m venv .venv && source .venv/bin/activate

# 2. install PyTorch for your platform first (see requirements.txt notes), then:
pip install -r requirements.txt

# 3. one-command bootstrap: install + tests + tiny smoke run
bash scripts/setup.sh
```

Or with conda: `conda env create -f environment.yml`.

## Usage

```bash
make data                                  # prepare/download dataset subsets
make train  CONFIG=configs/train_small.yaml
make eval   CONFIG=configs/train_small.yaml
make smoke                                 # tiny end-to-end check
make test                                  # unit tests
```

All entry points take `--config <yaml>` and support `--set key=value` overrides, e.g.:

```bash
python scripts/train.py --config configs/train_small.yaml --set training.lr=1e-4 model.num_experts=16
```

## Running on Colab GPU (real training)

The full 172M model trains in **bfloat16** on a GPU (it doesn't fit on a 16GB CPU box).
Use the ready notebook [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb):

1. Push this repo to GitHub.
2. Open the notebook in Colab, set runtime to **GPU** (T4/A100).
3. Edit `REPO_URL`, then *Run All* — it clones, installs, prepares data, trains, evaluates,
   runs ablations, and zips the results for download.

Notes: Colab already has a CUDA `torch`, so the notebook installs deps **without** torch;
bf16 is selected automatically on GPU (fp16 fallback on pre-Ampere T4 via `resolve_dtype`).

## Reproducibility

- Single YAML config fully describes each run; global seeding via `src/utils/seed.py`.
- Pinned dependencies (`requirements.txt`), deterministic flags, logged configs.
- Logs to TensorBoard / Weights & Biases and a local JSON lines file.

## Status

Implementation proceeds in phases (see `claude/instruction.md`). **Phase 1 (scaffold &
environment)** is complete; model, data, training, and evaluation land in later phases.

## License & attribution

Built on [OLMoE](https://github.com/allenai/OLMoE) (Allen Institute for AI) and
HuggingFace `transformers`. Routing-analysis utilities are adapted from the OLMoE repo.
