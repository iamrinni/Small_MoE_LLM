# Small Language Model with Sparse Mixture of Experts (Top-2 Gating)

A **small, sparse Mixture-of-Experts (MoE)** decoder-only language model built on the
**OLMoE** architecture and trained as a **multi-task** model across four text modalities:
**natural language, code, logic, and math**. Each token is routed to its **top-2 of 8
experts**; experts are SwiGLU MLPs; positional encoding is RoPE; training and inference run in
**bfloat16**. The model has **172.6M** total parameters but activates only **~68.8M** per
token — the compute–capacity decoupling that motivates sparse MoE.

> Course project — ACDL 2026. Author: **Iryna Yevdokymova**.
> Full technical report: [`report/report.pdf`](report/report.pdf).
> Step-by-step implementation plan: [`claude/instruction.md`](claude/instruction.md).

## Architecture at a glance

| Aspect | Choice |
|---|---|
| Base | HF `transformers` `OlmoeForCausalLM` (OLMoE), instantiated small, trained from scratch |
| Experts / gating | 8 experts, **top-2** routing |
| Expert FFN | **SwiGLU** (GeLU MLP as ablation) |
| Positional encoding | **RoPE** (learnable as ablation) |
| Precision | **bfloat16** on GPU (fp32 on CPU for dev) |
| Load balancing | auxiliary load-balance loss (the "routing loss" metric) |
| Routing analysis | expert load, entropy, and modality→expert specialization heatmap |

## Quickstart

### Option 1 (recommended): Colab GPU

The full model trains in bf16 on a GPU (it does not fit on a 16 GB CPU box). Open the ready
notebook directly from GitHub — no local setup:

1. In Colab: **File → Open notebook → GitHub**, enter **`iamrinni/Small_MoE_LLM`**, pick
   **`notebooks/colab_train.ipynb`**.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all.** It clones, installs, prepares data, trains, evaluates, renders the
   routing heatmap, optionally ablates, and zips the model + logs + figures for download.

### Option 2: local (clean environment, Python ≥ 3.10)

```bash
git clone https://github.com/iamrinni/Small_MoE_LLM.git && cd Small_MoE_LLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup.sh          # install + tests + tiny smoke run
```

On CPU the code runs in fp32 (development / smoke only); full bf16 training needs a GPU.

## Usage

```bash
make data                                    # prepare/download dataset subsets
make train  CONFIG=configs/train_small.yaml  # train
make eval   CONFIG=configs/train_small.yaml  # evaluate a checkpoint
make smoke                                   # tiny end-to-end check
make test                                    # unit tests (135)
make report                                  # build report PDF/HTML
```

Every entry point takes `--config <yaml>` and supports `--set key=value` overrides:

```bash
python scripts/train.py --config configs/train_small.yaml \
    --set training.max_steps=3000 model.num_experts=16
```

## Repository layout

```
configs/     YAML model + training configs (model_small, train_small, smoke, local_small)
scripts/     train / evaluate / prepare_data / run_ablations / plot_metrics / setup.sh
src/         model, data, training, eval, utils packages
data/        dataset preparation (raw data is NOT committed)
notebooks/   colab_train + 01_data_eda + 02_routing_analysis + 03_results
report/      report.md (source), report.pdf, style.css, figures/
results/     training log (JSONL) + TensorBoard from the GPU run
tests/       135 unit tests (model, data, trainer, eval, ablations)
```

## Results (summary)

A short GPU run (Colab T4, bf16) demonstrates the full pipeline; see the report for details
and the important note on the limited training budget.

- **Training**: cross-entropy falls from ~10.6 toward the low single digits; per-modality
  validation perplexity orders as logic ≈ 2 < math ≈ 32 < code ≈ 64 < language ≈ 560.
- **Routing**: expert load balance stays ~0.92–0.997 (no expert collapse).
- **Specialization**: experts specialize by modality in a *diffuse, overlapping* way
  (`report/figures/routing_heatmap.png`), not one-expert-per-modality.
- **Ablations**: load balancing improves utilization; a clear balance↔specialization
  trade-off; RoPE beats learnable positions (`report/figures/ablation_comparison.png`).

> **Note on compute.** Due to limited GPU resources the model was trained for few iterations;
> reported metrics are lower bounds and improve with longer training.

## Trained model checkpoint

The 172M checkpoint (~329 MB) is not stored in git (GitHub's 100 MB/file limit). Reproduce it
via the Colab notebook (Option 1), or attach it to a GitHub Release. See
[`results/README.md`](results/README.md).

## Reproducibility

- A single YAML config fully describes each run; global seeding via `src/utils/seed.py`.
- Pinned dependencies (`requirements.txt`); logs to TensorBoard and JSON-lines.
- 135 tests cover model, data, trainer, eval, and ablations — including an overfit-a-tiny-batch
  gate and a checkpoint save/reload round-trip.

## License & attribution

Built on [OLMoE](https://github.com/allenai/OLMoE) (Allen Institute for AI) and HuggingFace
`transformers`. Datasets: C4, CodeParrot, GSM8K, LogiQA (see `data/README.md`).
