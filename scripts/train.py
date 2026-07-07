"""Step 4.5 — training entry point.

One-command, config-driven run:
    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/train_small.yaml --set training.max_steps=2000

Builds tokenizer → model (sized to the tokenizer vocab) → data (local JSONL or synthetic)
→ logger → Trainer, then trains. All knobs come from the YAML (+ --set overrides).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.tokenizer import MODALITIES, build_tokenizer, ensure_vocab_fits  # noqa: E402
from src.model.config import load_model_config  # noqa: E402
from src.model.model import SmallMoE  # noqa: E402
from src.training.data import (  # noqa: E402
    MixtureIterableDataset,
    build_dataloader,
    local_source_factory,
    synthetic_source_factory,
)
from src.training.logger import MetricLogger  # noqa: E402
from src.training.trainer import Trainer, TrainConfig  # noqa: E402
from src.utils.config import add_config_args, load_config, merge_overrides  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _source_factory(data_cfg: dict):
    source = data_cfg.get("source", "auto")
    data_dir = REPO_ROOT / data_cfg.get("data_dir", "data/raw")
    have_local = all((data_dir / f"{m}.jsonl").exists() for m in MODALITIES)

    if source == "synthetic" or (source == "auto" and not have_local):
        print(f"[data] using SYNTHETIC sources (source={source}, local_present={have_local})")
        return synthetic_source_factory(MODALITIES, data_cfg.get("synthetic_size", 2000))
    print(f"[data] using LOCAL sources from {data_dir}")
    return local_source_factory(data_dir, MODALITIES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SmallMoE.")
    add_config_args(parser)
    args = parser.parse_args()

    cfg = merge_overrides(load_config(args.config), args.set)
    run, data_cfg = cfg.get("run", {}), cfg.get("data", {})
    set_seed(run.get("seed", 42), deterministic=run.get("deterministic", True))

    output_dir = REPO_ROOT / run.get("output_dir", "checkpoints/run")
    max_len = data_cfg.get("max_seq_len", 1024)

    # tokenizer + model (vocab sized to fit the tokenizer incl. modality tags)
    tokenizer = build_tokenizer(data_cfg.get("tokenizer", "allenai/OLMoE-1B-7B-0924"))
    # load_model_config understands both a `model:` block and defaults+model_overrides,
    # so pass the run file directly (smoke.yaml's tiny overrides are applied here).
    model_cfg = load_model_config(args.config)
    model_cfg.vocab_size = max(model_cfg.vocab_size, len(tokenizer))
    ensure_vocab_fits(tokenizer, model_cfg.vocab_size)
    model = SmallMoE(model_cfg)
    total, _ = model.num_parameters()
    print(f"[model] {total/1e6:.1f}M params | experts={model_cfg.num_experts} "
          f"top_k={model_cfg.num_experts_per_tok} dtype={model.dtype}")

    # data
    factory = _source_factory(data_cfg)
    dataset = MixtureIterableDataset(factory, data_cfg["mixture"], tokenizer, max_len,
                                     seed=run.get("seed", 42),
                                     max_examples=data_cfg.get("train_examples"))
    dataloader = build_dataloader(dataset, tokenizer,
                                  batch_size=cfg["training"].get("per_device_batch_size", 8),
                                  max_len=max_len)

    # validation sources (per-modality, small held-out sample)
    val_factory = _source_factory(data_cfg)()
    val_sources = {m: val_factory[m] for m in MODALITIES}

    # logging
    log_cfg = cfg.get("logging", {})
    logger = MetricLogger(
        output_dir, backend=log_cfg.get("backend", "tensorboard"),
        json_path=REPO_ROOT / log_cfg["json_log_path"] if log_cfg.get("json_log_path") else None,
        wandb_project=log_cfg.get("wandb_project"), run_name=run.get("name"), config=cfg,
    )

    train_cfg = TrainConfig.from_dict(cfg)
    trainer = Trainer(train_cfg, model, tokenizer, dataloader, output_dir,
                      logger=logger, val_sources=val_sources, max_len=max_len)
    print(f"[train] max_steps={train_cfg.max_steps} bs={cfg['training'].get('per_device_batch_size')} "
          f"grad_accum={train_cfg.grad_accum_steps} precision={train_cfg.precision}")
    history = trainer.train()
    logger.close()

    loss_entries = [h for h in history if "loss/total" in h]
    if loss_entries:
        last = loss_entries[-1]
        print(f"[done] final step={last['step']} loss/total={last['loss/total']:.3f} "
              f"loss/ce={last.get('loss/ce', float('nan')):.3f}")


if __name__ == "__main__":
    main()
