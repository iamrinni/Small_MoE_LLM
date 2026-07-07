"""Step 4.2 — optimizer + learning-rate schedule.

AdamW with a linear **warmup** followed by **cosine decay** to a floor (`min_lr`). Weight
decay is not applied to biases or norm weights (standard practice). The schedule is a plain
`LambdaLR` returning a multiplier on the base LR.
"""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """AdamW with decay applied only to matrix weights (not biases / norms)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=betas)


def cosine_warmup_lambda(warmup_steps: int, max_steps: int, min_ratio: float) -> "callable":
    """Return a step→multiplier fn: linear warmup, then cosine decay to `min_ratio`."""
    warmup_steps = max(warmup_steps, 0)
    max_steps = max(max_steps, warmup_steps + 1)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return fn


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    lr: float,
    min_lr: float,
) -> LambdaLR:
    min_ratio = (min_lr / lr) if lr > 0 else 0.0
    return LambdaLR(optimizer, lr_lambda=cosine_warmup_lambda(warmup_steps, max_steps, min_ratio))
