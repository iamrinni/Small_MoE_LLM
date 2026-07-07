"""Step 4.4 — the training loop.

Accelerate-based trainer for `SmallMoE`. Handles gradient accumulation, gradient clipping,
AdamW + cosine-warmup schedule, metric logging (loss components, lr, grad-norm, routing
stats, throughput), resumable checkpoints, and per-modality validation perplexity.

Precision: Accelerate mixed-precision = bf16 on CUDA (per spec) / off on CPU (dev), keeping
fp32 master weights — the standard, stable way to train "in bf16".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from src.data.collate import pad_collate
from src.data.format import encode_example, has_trainable_tokens
from src.data.sources import Example
from src.model.losses import compute_loss
from src.model.model import SmallMoE
from src.training.logger import MetricLogger
from src.training.optim import build_optimizer, build_scheduler


@dataclass
class TrainConfig:
    max_steps: int = 5000
    warmup_steps: int = 200
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    precision: str = "bfloat16"
    gradient_checkpointing: bool = False
    log_every: int = 20
    eval_every: int = 500
    save_every: int = 1000
    log_routing: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "TrainConfig":
        t, e, l = cfg.get("training", {}), cfg.get("eval", {}), cfg.get("logging", {})
        return cls(
            max_steps=t.get("max_steps", 5000),
            warmup_steps=t.get("warmup_steps", 200),
            lr=float(t.get("lr", 3e-4)),
            min_lr=float(t.get("min_lr", 3e-5)),
            weight_decay=t.get("weight_decay", 0.1),
            grad_accum_steps=t.get("grad_accum_steps", 1),
            grad_clip=t.get("grad_clip", 1.0),
            adam_beta1=t.get("adam_beta1", 0.9),
            adam_beta2=t.get("adam_beta2", 0.95),
            precision=t.get("precision", "bfloat16"),
            gradient_checkpointing=t.get("gradient_checkpointing", False),
            log_every=l.get("log_every_steps", 20),
            eval_every=e.get("eval_every_steps", 500),
            save_every=e.get("save_every_steps", 1000),
            log_routing=l.get("log_routing_stats", True),
        )


def _mixed_precision(precision: str) -> str:
    if precision == "bfloat16" and torch.cuda.is_available():
        return "bf16"
    if precision == "float16" and torch.cuda.is_available():
        return "fp16"
    return "no"  # CPU dev → fp32


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: SmallMoE,
        tokenizer,
        train_dataloader: DataLoader,
        output_dir: str | Path,
        logger: MetricLogger | None = None,
        val_sources: dict[str, Iterator[Example]] | None = None,
        max_len: int = 1024,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.val_sources = val_sources
        self.max_len = max_len
        self.step = 0

        self.accelerator = Accelerator(
            mixed_precision=_mixed_precision(cfg.precision),
            gradient_accumulation_steps=cfg.grad_accum_steps,
        )
        self.smallmoe = model  # keep the wrapper (for cfg + save)
        if cfg.gradient_checkpointing:
            # trade compute for memory — recompute activations in backward (fits big models on T4)
            model.model.gradient_checkpointing_enable()
            model.model.config.use_cache = False
        optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay, (cfg.adam_beta1, cfg.adam_beta2))
        scheduler = build_scheduler(optimizer, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)

        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        self.logger = logger

    # --- infinite batch stream over the (finite) dataloader --------------------------
    def _batches(self) -> Iterator[dict]:
        while True:
            for batch in self.dataloader:
                yield batch

    def train(self) -> list[dict]:
        history: list[dict] = []
        self.model.train()
        t0 = time.time()
        tokens_since_log = 0

        for batch in self._batches():
            if self.step >= self.cfg.max_steps:
                break
            with self.accelerator.accumulate(self.model):
                out = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                    collect_routing=self.cfg.log_routing,
                )
                loss = out.loss
                self.accelerator.backward(loss)
                grad_norm = None
                if self.accelerator.sync_gradients and self.cfg.grad_clip > 0:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            tokens_since_log += int(batch["attention_mask"].sum().item())

            if self.accelerator.sync_gradients:
                self.step += 1

                if self.step % self.cfg.log_every == 0:
                    dt = max(time.time() - t0, 1e-6)
                    metrics = out.metrics()
                    metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                    metrics["throughput/tok_per_s"] = tokens_since_log / dt
                    if grad_norm is not None:
                        metrics["grad_norm"] = float(grad_norm)
                    if "loss/ce" in metrics:
                        metrics["train/perplexity"] = math.exp(min(metrics["loss/ce"], 20))
                    if self.logger:
                        self.logger.log(metrics, step=self.step)
                    history.append({"step": self.step, **metrics})
                    t0, tokens_since_log = time.time(), 0

                if self.val_sources and self.step % self.cfg.eval_every == 0:
                    val = self.validate()
                    if self.logger:
                        self.logger.log(val, step=self.step)
                    history.append({"step": self.step, **val})
                    self.model.train()

                if self.step % self.cfg.save_every == 0:
                    self.save_checkpoint()

        self.save_checkpoint(final=True)
        return history

    @torch.no_grad()
    def validate(self, n_per_modality: int = 16) -> dict[str, float]:
        """Per-modality validation perplexity over a small held-out sample."""
        self.model.eval()
        metrics: dict[str, float] = {}
        for modality, source in (self.val_sources or {}).items():
            encoded = []
            for ex in source:
                enc = encode_example(self.tokenizer, ex, self.max_len)
                if has_trainable_tokens(enc):
                    encoded.append(enc)
                if len(encoded) >= n_per_modality:
                    break
            if not encoded:
                continue
            batch = pad_collate(encoded, pad_token_id=self.tokenizer.pad_token_id, max_len=self.max_len)
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
            out = self.model(input_ids=batch["input_ids"], labels=batch["labels"],
                             attention_mask=batch["attention_mask"], collect_routing=False)
            ce = float(out.loss_breakdown.ce)
            metrics[f"val/{modality}/ce"] = ce
            metrics[f"val/{modality}/perplexity"] = math.exp(min(ce, 20))
        if metrics:
            ces = [v for k, v in metrics.items() if k.endswith("/ce")]
            metrics["val/mean_ce"] = sum(ces) / len(ces)
        return metrics

    def save_checkpoint(self, final: bool = False) -> Path:
        name = "final" if final else f"step_{self.step}"
        path = self.output_dir / name
        # accelerate state (model+optimizer+scheduler+RNG) for RESUMING training
        self.accelerator.save_state(str(path))
        if self.accelerator.is_main_process:
            # HF model in an `hf/` subdir for EVAL/INFERENCE loading (correct key names)
            hf_model = self.accelerator.unwrap_model(self.model).model
            hf_model.save_pretrained(str(path / "hf"))
            (path / "trainer_state.json").write_text(f'{{"step": {self.step}}}')
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        self.accelerator.load_state(str(path))
        import json

        state = json.loads((Path(path) / "trainer_state.json").read_text())
        self.step = state["step"]
