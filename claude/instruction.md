# Implementation Plan — Small Language Model with Sparse Mixture of Experts (Top-2 Gating)

> **Source task:** `33-ACDL-2026-Project-small-language-model with MoE (1).pdf`
> **Goal:** Design, implement, train, and evaluate a *small* multimodal (language · code · logic · math) transformer based on a **Sparse Mixture-of-Experts (MoE)** architecture with **top-2 gating**, following the **OLMoE** design. Ship runnable, reproducible code + a 15-page technical report.

---

## 0. Scope, strategy & key decisions

The PDF mixes two ambitions: (a) a **sparse MoE language model** (the core, well-defined and achievable) and (b) a **vision-language / VQA** multimodal model (heavy, optional). The pragmatic, fully-achievable interpretation that still satisfies every *mandatory* deliverable:

- **Core (must build):** a decoder-only transformer LM with sparse top-2 MoE FFN layers, **built on / forked from the existing OLMoE codebase** (the original repo or HuggingFace `transformers`' `OlmoeForCausalLM`), with our extensions (config, multi-task data, routing instrumentation, ablations) living in this repo. Trained as a **multi-task text model** over four modalities: **natural language, code, logic, math**.
- **Multimodal note (LOCKED):** "multimodal" here = **multi-task across text domains** (language/code/logic/math). The base model is **text-only**; **vision-language cross-attention (spec items #1 & #8) is deferred to the optional Phase 9.**
  - **Justification for the report:** cross-attention fusion requires **image-derived keys/values** for text queries to attend to. The spec lists **only text datasets** (code, logic, math, education — no image-text/VQA corpus), so a fusion module could not be *meaningfully trained or evaluated* under the provided data — it would be either an untrained (random-init) dead module or trained on dummy inputs. Adding an unlisted image-text dataset (e.g. VQA v2 / COCO-Captions) is the only way to make it real, which is a deliberate scope increase we leave optional. This is an internal inconsistency in the spec (vision architecture + VQA metric requested, but no vision data provided); we document it rather than ship a hollow layer.
  - **Architecture-requirement compliance:** of the 8 Core-Architecture items, **#2–#7 are fully met** in the base model (OLMoE framework, top-2 gating, 8–16 experts, SwiGLU experts, strict bf16, RoPE/learnable PE w/ experiment). **#1 (vision-language) and #8 (cross-attention)** are met **only if Phase 9 is completed** with an image-text dataset.
- **Scale:** target a **small** model (~50M–200M total params, e.g. hidden 512, 8–12 layers, 8 experts, top-2). Must train end-to-end on a single GPU (Colab/A100 or local) within hours, not weeks.

### Decisions LOCKED before coding (record each in the report)
| Decision | Choice | Alternatives to ablate |
|---|---|---|
| **Build approach** | **Hybrid:** model + router + load-balance loss from **HF `transformers` `OlmoeForCausalLM`**; port the **analysis/visualization functions from `allenai/OLMoE`** (`run_moe_analysis.py`, `run_routing_analysis.py`, plotting notebooks). Extensions live in this repo. | — |
| **# experts** | **8** | 16 (ablation) |
| Gating | top-2 | top-1 (ablation) |
| **Expert FFN** | **SwiGLU MLP** (HF OLMoE native; matches spec) | GeLU MLP (ablation) |
| **Positional encoding** | **RoPE (rotary)** (HF native; matches spec) | learnable (ablation) |
| Precision | **bfloat16** | fp16 |
| Tokenizer | reuse OLMoE/GPT-NeoX BPE tokenizer | — |
| Load balancing | aux load-balance loss (HF built-in = "Routing loss" metric) | none (ablation) |

> **Why these match the spec (no deviation needed):** the spec asks for *"MLP with SwiGLU activation"* and *"Rotary or learnable"* positional encoding. HF `OlmoeForCausalLM` provides **SwiGLU experts and RoPE natively**, so we keep both as defaults — zero custom code and fully spec-compliant. GeLU MLP and learnable PE are kept only as **ablation variants** (built when those experiments are run, in Phase 6).
>
> **One choice to record:** the spec says *"# experts: 8–16 (adjustable)"*. We fix **8** as the default and ablate 16. (A natural intuition is "4 experts for 4 modalities" — but experts are token-level, not modality-level, specialists; top-2-of-8 also preserves real sparsity at 25% active. We turn this into an explicit *finding* via the modality→expert routing heatmap.)

---

## Phase 1 — Repository scaffold & environment

**Goal:** clean-environment reproducibility (a hard requirement).

1. Create the required repo layout:
   ```
   Small_MoE_LLM/
   ├── README.md
   ├── requirements.txt          # pinned versions
   ├── environment.yml           # optional conda
   ├── configs/                  # YAML model + training configs
   ├── scripts/                  # train / eval / data-prep / logging
   ├── notebooks/                # EDA, routing visualizations
   ├── data/                     # download/prep scripts + pointers (NOT raw data)
   ├── src/                      # model + training library
   │   ├── model/                # transformer, MoE layer, router, attention, RoPE
   │   ├── data/                 # datasets, tokenization, multi-task sampler
   │   ├── training/             # trainer, losses, lr schedule
   │   └── eval/                 # per-task metrics
   ├── report/                   # LaTeX + figures + final PDF
   └── tests/                    # unit tests (shape/router/loss sanity)
   ```
2. `requirements.txt`: `torch>=2.2`, `transformers`, `datasets`, `accelerate`, `deepspeed` (optional), `wandb`, `tensorboard`, `numpy`, `pyyaml`, `tqdm`, `evaluate`, `sentencepiece`, `tiktoken`/`tokenizers`, `matplotlib`, `pytest`.
3. Pin Python ≥ 3.10. Add `.gitignore` (checkpoints, `data/raw/`, `wandb/`, `*.pt`).
4. Add a `Makefile` / `scripts/setup.sh` so a clean machine runs: install → tiny smoke train → eval.
5. Set global **seed** + deterministic flags for reproducibility.

**Deliverable check:** repo layout matches PDF's "GitHub Repository Requirements" exactly.

---

## Phase 2 — Model architecture (the heart)

**Approach: hybrid — HF model + AllenAI analysis functions.** Build on HF
`OlmoeForCausalLM`; add only the thin layer of extensions OLMoE doesn't expose.

### What HF `OlmoeForCausalLM` (transformers 4.49, verified) gives for free
- Decoder transformer, **RoPE**, RMSNorm, attention (incl. GQA), bf16.
- **Top-2 sparse MoE block** (`OlmoeSparseMoeBlock`): softmax routing + `norm_topk_prob`.
- **SwiGLU expert** (`OlmoeMLP`, gate/up/down) — this *is* our locked default; no custom code.
- **Load-balancing aux loss** (`load_balancing_loss_func`, gated by `router_aux_loss_coef`).
- `router_logits` output (`output_router_logits=True`) — raw material for routing metrics.

### What we build (the actual Phase-2 work)
- **Combined-loss helper**: CE + HF's built-in load-balance aux loss (the required "Routing loss" metric). *(Router z-loss is NOT used — it stays only as a dormant config var, default 0.0, to enable only if bf16 routing destabilizes.)*
- **Routing-stats computation**: per-expert load, gate entropy, top-2 ids from `router_logits`.
- **Config glue** (YAML → `OlmoeConfig` + our `*_ext` flags) and a clean **wrapper**.
- *(Deferred to Phase 6, only for ablations):* GeLU-MLP expert override and learnable-PE switch.

### Functions ported from `allenai/OLMoE` (into `src/eval/`, `notebooks/`)
Routing-analysis logic from `run_moe_analysis.py` / `run_routing_analysis.py` (per-expert
load, expert/gate entropy, router saturation, domain/vocabulary specialization) and the
`plot_routing_analysis*.ipynb` / `olmoe_visuals.ipynb` figure recipes. We do **not** import
111OLMoE's training internals (OLMo fork + megablocks, cluster-scale). Record what was ported
and from where (attribution + report).

### Precision policy — bf16 target, device-aware (spec: "bfloat16 throughout training and inference")
**Decision (measured):** CPU has no native bf16 matmul — `OlmoeForCausalLM` bf16 training on
CPU is **~67x slower** than fp32 (benchmark: 170M @ 11 tok/s bf16 vs 736 tok/s fp32; full bf16
CPU run ≈ months). So precision is **device-aware** via `resolve_dtype(cfg, device)`:
- **GPU (real runs + inference)** → **bf16** — satisfies the spec where training actually happens.
- **Local CPU (dev / tests / smoke)** → **fp32** — bf16 emulation is impractical; fp32 is for debugging logic, not reported results.

**Where training runs:** real/reported runs on **free GPU (Colab/Kaggle T4-A100)** in bf16;
local 16GB CPU box is for development, smoke tests, and debugging only.

Two unavoidable fp32 details that don't reflect model precision: the scalar **loss** value
(CE reduction output) and HF's internal **router-softmax** reduction.

Consequence for tests: the **overfit-tiny gate (2.7) is precision-aware** — assert a large
loss drop / sub-threshold loss, not exactly ~0.

### Decomposed steps (ordered, each independently testable)

| # | Step | File(s) | Exit test |
|---|---|---|---|
| **2.1** | **Config layer** — `SmallMoEConfig(OlmoeConfig)` subclass adding only the 4 fields HF lacks (`router_z_loss_coef`, `pos_encoding_ext`, `expert_activation_ext`, `dtype_ext`); YAML loader feeds it and validates locked invariants. **Do not duplicate OlmoeConfig** — reuse its fields + serialization. **Register the subclass** (`AutoConfig.register`) so checkpoints round-trip (reproducibility deliverable). | `src/model/config.py` | builds from `model_small.yaml`; `isinstance(cfg, OlmoeConfig)`; `save_pretrained`→`from_pretrained` round-trips incl. the 4 extras; asserts invariants |
| **2.2** | **Baseline builder** — `build_model(cfg)` → small `OlmoeForCausalLM`, `output_router_logits=True`, **strict bf16 via `resolve_dtype`** (always bf16). (SwiGLU experts + RoPE come native here.) | `src/model/build.py` | forward shape `[B,T,V]`; param dtype is bf16; aux_loss present w/ labels |
| **2.3** | **Routing instrumentation** — from `router_logits` compute per-expert load, gate entropy, top-2 ids | `src/model/routing.py` | load sums to ~1; exactly 2 active per token; entropy ∈ [0, log E] |
| **2.4** | **Combined loss** — `CE + aux·coef` using HF's built-in load-balance aux loss (the "Routing loss" metric). z-loss left out (dormant var only). | `src/model/losses.py` | combined loss finite & differentiable; aux term present |
| **2.5** | **Positional-encoding switch** — RoPE wired (native default); learnable kept as a flag, full impl deferred to Phase 6 | `src/model/pos_encoding.py` | rope config builds + forwards; flag recognized |
| **2.6** | **Top-level wrapper** `SmallMoE` — build + routing collection + combined loss; clean `forward` → logits, loss, routing stats | `src/model/model.py` | end-to-end forward returns all three |
| **2.7** | **Test suite + overfit-tiny gate** (Phase-2 exit criterion) | `tests/test_model.py` | tiny model memorizes a tiny batch → near-0 loss |

**Expert width (`intermediate_size`) rule:** derive from `hidden_size`, don't hard-code.
SwiGLU MLP (3 matrices, default) → `d_ff ≈ (8/3) × hidden ≈ 1408` at hidden 512. The GeLU
ablation (2 matrices) uses `d_ff ≈ 1.5 × d_ff_swiglu ≈ 2112` (round to a multiple of 64/128)
so the two variants have **matched parameter counts** — otherwise the comparison is unfair.
At the default size the model is ~170M total / ~70M active (top-2).

**Dependencies:** all build on 2.1→2.2; 2.3–2.5 are independent; 2.6 integrates; 2.7 gates.
**Sanity gate:** overfit a tiny batch to ~0 loss before any real training (step 2.7).

---

## Phase 3 — Tokenization & data pipeline

Implement in `src/data/` + `data/` scripts.

1. **Tokenizer:** reuse the OLMoE / GPT-NeoX HF tokenizer (fast, no training needed). Add special tokens / task tags: `<|lang|> <|code|> <|logic|> <|math|>` prepended per example so the model (and router) can condition on modality.
2. **Dataset adapters** (one loader per modality, streaming where possible):
   - **Language (NLP):** pick **1–2** of mC4 / OSCAR / CC100 / Wikipedia. *Default: a small Wikipedia + a slice of C4* (easy via HF `datasets`, streaming).
   - **Code:** CodeParrot (Python) or a small slice of The Stack v2 (Python).
   - **Logic:** LogiQA, ReClor → format as instruction/QA text.
   - **Math:** GSM8K, MATH (and Minerva-style prompts) → question + step-by-step solution.
   - (Stretch) **Education exams:** SAT/GRE/GMAT/ENS → held-out eval only.
3. **Formatting:** convert QA/reasoning datasets to a uniform text template (prompt + answer), so everything is next-token prediction with a loss mask on the prompt (train on completion).
4. **Multi-task sampler** (`mixture.py`): the PDF's "mixture-of-tasks routing logic" → a weighted interleaving sampler that draws batches from each modality by configurable proportions; log the realized mixture. Document the chosen weights + rationale.
5. **Packing & batching:** tokenize, pack to `max_seq_len`, build batches; bf16-friendly collator.
6. `scripts/prepare_data.py`: downloads/streams + caches small subsets; `data/README.md` documents exact dataset names, versions, splits, and sizes (reproducibility).

**Keep it small:** cap each corpus (e.g. tens–hundreds of MB tokens) so the whole run is feasible.

---

## Phase 4 — Training loop

Implement in `src/training/` + `scripts/train.py`.

1. **Trainer** built on **HuggingFace Accelerate** (DeepSpeed ZeRO optional via config) for bf16 + (optional) multi-GPU.
2. **Loss** = cross-entropy (LM) + `aux_loss_coef * load_balance_loss` + `z_loss_coef * router_z_loss`.
3. Optimizer **AdamW**, cosine LR schedule + warmup, grad clipping, grad accumulation.
4. **Checkpointing:** save model + optimizer + config + RNG state; resumable. Save final + best.
5. **Logging (wandb *and/or* TensorBoard):** train/val loss, perplexity, LR, grad norm, **per-expert load**, **expert/gate entropy**, **routing (aux) loss**, throughput. Also dump local JSON logs (PDF allows "local JSON").
6. **Config-driven runs:** `configs/train_small.yaml`; one command launches a run.
7. **Validation:** held-out perplexity per modality, logged each eval interval.

**Milestones:** (a) smoke run (10 steps, tiny config) green in CI; (b) short real run; (c) full small run with logged curves.

---

## Phase 5 — Evaluation suite

Implement in `src/eval/` + `scripts/evaluate.py`. One metric module per task type (per PDF table):

| Task | Metric | Notes |
|---|---|---|
| Programming | **Pass@k**, CodeBLEU | Pass@k needs sandboxed execution of generated Python on unit tests (e.g. HumanEval/MBPP-style or CodeParrot eval); CodeBLEU via `codebleu` pkg |
| Logic | Accuracy | multiple-choice exact match (LogiQA/ReClor) |
| Math | Exact match + step-by-step accuracy | GSM8K final-answer EM; reasoning-step heuristic |
| University exams | Score %, normalized | held-out MC accuracy |
| MoE (always) | **Expert load, expert entropy, routing loss** | from router stats over eval set |
| (Stretch) VQA | Accuracy, VQA-Score | only if Phase 9 done |

1. Generation utilities (greedy + sampling) for open-ended tasks.
2. `evaluate.py` runs all applicable tasks on a checkpoint, writes a JSON results table + markdown summary.
3. **MoE analysis script** (`scripts/analyze_routing.py`): expert-load histograms, gate-entropy curves, per-modality expert specialization heatmap — feeds report figures.

---

## Phase 6 — Experiments & ablations

Run a small, well-documented experiment matrix (each → logged run + curves):

1. **Positional encoding:** RoPE vs learnable (PDF explicitly asks to "experiment and report findings").
2. **Top-k gating:** top-2 vs top-1.
3. **# experts:** 8 (default) vs 16.
4. **Expert FFN:** SwiGLU (default) vs GeLU MLP.
5. **Load-balancing loss:** with vs without (show expert-collapse).
6. **(Optional) task-mixture weights:** balanced vs language-heavy.

For each: final perplexity, per-task metrics, expert-load balance, training cost. Tabulate.

---

## Phase 7 — Visualizations & notebooks

In `notebooks/`:
1. `01_data_eda.ipynb` — dataset stats, token-length distributions, mixture proportions.
2. `02_routing_analysis.ipynb` — expert load, entropy over training, modality→expert heatmap, top-2 co-activation.
3. `03_results.ipynb` — training/val loss curves, ablation comparison plots, sample generations per modality.

Export figures to `report/figures/`.

---

## Phase 8 — Technical report (15 pages) & documentation

1. **README.md:** project summary, architecture diagram, **clean-env setup**, exact commands to reproduce (`prepare_data → train → evaluate`), config explanations, expected results, checkpoint links.
2. **Report** (`report/`, LaTeX):
   - Intro & objectives; Related work (OLMoE, Switch/Mixtral, sparse MoE).
   - **Architecture** (MoE layer, top-2 router, SwiGLU, RoPE, load balancing) with diagrams.
   - **Datasets** (sources, sizes, preprocessing, multi-task mixture + design decisions).
   - **Training** (setup, hyperparams, precision, hardware, cost).
   - **Metrics & results** per task + tables/graphs.
   - **Ablation studies** (Phase 6) with analysis.
   - **MoE behavior analysis** (expert load/entropy/specialization).
   - Limitations (incl. multimodal/VQA scoping), future work, conclusion.
   - Reproducibility appendix (seeds, versions, commands).
3. Compile to `report/report.pdf`.

---

## Phase 9 — (Optional / stretch) Vision-language fusion

Only if core is complete and stable. **Prerequisite: an image-text dataset must be added first** — the spec lists none, so this phase is blocked until one is chosen (e.g. VQA v2 / COCO-Captions / a small VQA subset). Without it, cross-attention cannot be trained (see the vision justification in §0).
1. Source + prepare a small image-text/VQA dataset (the missing prerequisite).
2. Add a frozen vision encoder (e.g. CLIP ViT) producing patch embeddings.
3. **Cross-attention** adapter layers fusing vision tokens into the decoder (PDF's "cross-attention for vision-language fusion") — satisfies Core-Architecture #1 & #8.
4. Train on the VQA set; evaluate Accuracy / VQA-Score.
5. Report as an extension; keep behind a config flag so the text-only path stays the default reproducible one.

---

## Phase 10 — Final packaging & reproducibility audit

1. Fresh clone → `scripts/setup.sh` → smoke train → eval, on a clean environment. Fix anything that breaks.
2. Upload **final checkpoints + sample outputs** + experiment logs (wandb export / TensorBoard / JSON).
3. Verify all PDF deliverables present:
   - [ ] Runnable source on GitHub + docs
   - [ ] 15-page technical report (PDF + sources)
   - [ ] Models & logs sufficient to replicate
   - [ ] README with detailed usage
   - [ ] Final checkpoints & sample outputs
4. Tag a release.

---

## Suggested build order (dependency-aware)
1. Phase 1 scaffold → 2. **Model + unit tests** (overfit-tiny gate) → 3. Data pipeline → 4. Training loop (smoke run) → 5. Eval suite → 6. Short real run → 7. Ablations → 8. Viz → 9. Report → 10. (optional) Vision → 11. Repro audit & release.

## Top risks & mitigations
- **Compute** → keep model small, cap data, stream datasets, bf16, grad accumulation.
- **Expert collapse / unbalanced routing** → load-balance + z-loss; monitor expert load from step 1.
- **Pass@k sandboxing complexity** → use an established harness (HumanEval/MBPP) or restrict to CodeBLEU if execution infra is hard.
- **Scope creep (vision)** → vision is explicitly optional, behind a flag, documented as such.
- **Reproducibility** → seeds, pinned versions, clean-env audit in Phase 10.
