"""Step 2.4 — combined loss with a logged breakdown.

`OlmoeForCausalLM` already folds the load-balance aux loss into `outputs.loss`
(`loss = CE + router_aux_loss_coef * aux_loss`) and exposes the unweighted `aux_loss`
separately. For training we need the **components** so we can log them — in particular the
load-balance term, which is the task's required "Routing loss" MoE metric.

This helper recomputes the language-modeling CE in fp32 (exact, dtype-independent) and pairs
it with HF's `aux_loss`. Router **z-loss is intentionally NOT included** here — it lives only
as a dormant config field (`router_z_loss_coef`, default 0.0); we'd wire it in only if bf16
routing destabilizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.model.config import SmallMoEConfig


@dataclass
class LossBreakdown:
    total: torch.Tensor   # CE + coef*aux — the value to call .backward() on
    ce: torch.Tensor      # language-modeling cross-entropy
    aux: torch.Tensor     # unweighted load-balance aux loss ("Routing loss" metric)
    aux_weighted: torch.Tensor  # router_aux_loss_coef * aux

    def to_dict(self, prefix: str = "loss/") -> dict[str, float]:
        return {
            f"{prefix}total": float(self.total),
            f"{prefix}ce": float(self.ce),
            f"{prefix}aux": float(self.aux),           # the routing/load-balance loss
            f"{prefix}aux_weighted": float(self.aux_weighted),
        }


def cross_entropy_lm(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Causal-LM cross-entropy with the standard next-token shift, computed in fp32."""
    shift_logits = logits[:, :-1, :].float()           # predict token t+1 from t
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def compute_loss(outputs, labels: torch.Tensor, cfg: SmallMoEConfig) -> LossBreakdown:
    """Build the combined loss + breakdown from a model forward output.

    `outputs` must come from a forward with `output_router_logits=True`. We use HF's
    `outputs.aux_loss` (the load-balance term) and recompute CE for an exact breakdown.
    """
    ce = cross_entropy_lm(outputs.logits, labels)

    aux = getattr(outputs, "aux_loss", None)
    if aux is None:
        aux = torch.zeros((), device=ce.device, dtype=ce.dtype)
    aux = aux.float()

    coef = float(cfg.router_aux_loss_coef)
    aux_weighted = coef * aux
    total = ce + aux_weighted
    return LossBreakdown(total=total, ce=ce, aux=aux, aux_weighted=aux_weighted)
