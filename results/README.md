# Experiment results

Artifacts from GPU (Colab T4) runs. Large checkpoints are **not** committed here (GitHub's
100 MB/file limit); see "Model checkpoint" below.

## Contents
- `real_run_metrics_300steps.jsonl` — training log of the full 172M model on **real data**
  (C4 / CodeParrot / GSM8K / LogiQA mixture), bf16, T4. **300-step run** (short; the model is
  undertrained — a longer run improves absolute numbers).
- `tensorboard/` — TensorBoard event file for the same run.
- `../report/figures/`:
  - `training_curves.png`, `expert_load.png` — training + routing curves.
  - `ablation_results.{json,md}`, `ablation_comparison.png` — ablation matrix
    (6 variants). **Note:** ablations use a compact ~15M model on a synthetic mixture — they
    demonstrate *trends* (top-1 vs top-2, load-balance on/off, RoPE vs learnable), not
    absolute numbers of the 172M model.

## Model checkpoint
The trained 172M model (`model.safetensors`, ~329 MB) is not stored in git. To obtain it:
- re-run `notebooks/colab_train.ipynb` (produces `checkpoints/small-moe-baseline/final/hf/`), or
- attach it to a GitHub Release / Git LFS (see README).

## Reproduce
```bash
python scripts/train.py    --config configs/train_small.yaml   # train
python scripts/evaluate.py --config configs/train_small.yaml \
    --checkpoint checkpoints/small-moe-baseline/final           # eval
```
