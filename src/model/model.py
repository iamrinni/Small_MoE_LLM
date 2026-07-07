"""Step 2.6 — top-level `SmallMoE` wrapper.

One clean object that ties together the Phase-2 pieces: build the OLMoE model (2.2),
run a forward, and return a structured output with **logits, combined loss breakdown (2.4),
and routing stats (2.3)**. This is what the trainer (Phase 4) and eval (Phase 5) consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.model.build import build_model, count_parameters
from src.model.config import SmallMoEConfig, load_model_config
from src.model.losses import LossBreakdown, compute_loss
from src.model.routing import RoutingStats, aggregate_routing_stats


@dataclass
class SmallMoEOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None              # = loss_breakdown.total (for .backward())
    loss_breakdown: LossBreakdown | None = None
    routing: RoutingStats | None = None           # model-level aggregate
    routing_per_layer: list[RoutingStats] | None = None

    def metrics(self) -> dict[str, float]:
        """Flat scalar metrics for logging (loss components + routing)."""
        out: dict[str, float] = {}
        if self.loss_breakdown is not None:
            out.update(self.loss_breakdown.to_dict())
        if self.routing is not None:
            out.update(self.routing.to_dict(prefix="routing/"))
        return out


class SmallMoE(nn.Module):
    """Thin wrapper over `OlmoeForCausalLM` with routing + loss instrumentation."""

    def __init__(self, cfg: SmallMoEConfig, device: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg, device)

    @classmethod
    def from_yaml(cls, path, device: str | None = None) -> "SmallMoE":
        return cls(load_model_config(path), device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        collect_routing: bool = True,
    ) -> SmallMoEOutput:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        loss = breakdown = None
        if labels is not None:
            breakdown = compute_loss(out, labels=labels, cfg=self.cfg)
            loss = breakdown.total

        routing = per_layer = None
        if collect_routing and out.router_logits is not None:
            per_layer, routing = aggregate_routing_stats(out.router_logits, self.cfg.num_experts_per_tok)

        return SmallMoEOutput(
            logits=out.logits,
            loss=loss,
            loss_breakdown=breakdown,
            routing=routing,
            routing_per_layer=per_layer,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Delegate generation to the underlying HF model."""
        return self.model.generate(*args, **kwargs)

    def num_parameters(self) -> tuple[int, int]:
        return count_parameters(self.model)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def save_pretrained(self, path) -> None:
        self.model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path, device: str | None = None) -> "SmallMoE":
        from pathlib import Path

        from transformers import AutoConfig, OlmoeForCausalLM

        import src.model.config  # noqa: F401  (ensures SmallMoEConfig is registered)

        # Trainer checkpoints store the HF model under `hf/`; plain HF saves use the dir itself.
        p = Path(path)
        load_path = p / "hf" if (p / "hf").exists() else p

        cfg = AutoConfig.from_pretrained(load_path)
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.cfg = cfg
        model = OlmoeForCausalLM.from_pretrained(load_path)
        if device:
            model = model.to(device)
        obj.model = model
        return obj
