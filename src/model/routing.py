"""Step 2.3 — routing instrumentation.

Turns the raw `router_logits` that `OlmoeForCausalLM` emits (one `[N_tokens, E]` tensor per
layer, when `output_router_logits=True`) into the MoE metrics the task asks for:
**expert load, expert entropy, routing distribution** — plus the selected top-k experts.

All reductions run in fp32 for numerical stability (independent of model dtype). These are
diagnostics, so everything is computed under `no_grad`.

Two distinct "entropy" notions, both reported (both bounded in [0, log E]):
  * ``gate_entropy``  — mean over tokens of the per-token softmax entropy. Low ⇒ the router
    is confident/peaky per token; high ⇒ each token spread across experts.
  * ``load_entropy``  — entropy of the aggregate expert-load distribution. High ⇒ tokens are
    spread evenly across experts (good balance); low ⇒ collapse onto few experts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class RoutingStats:
    """Aggregated routing diagnostics (per layer, or averaged across layers)."""

    expert_load: torch.Tensor   # [E] fraction of top-k assignments per expert; sums to 1
    gate_entropy: float         # mean per-token softmax entropy (nats), in [0, log E]
    load_entropy: float         # entropy of expert_load (nats), in [0, log E]
    max_entropy: float          # log E — the balanced/uniform reference
    n_tokens: int
    n_experts: int
    top_k: int

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Flatten to scalar metrics for logging (per-expert load expanded)."""
        d = {
            f"{prefix}gate_entropy": self.gate_entropy,
            f"{prefix}load_entropy": self.load_entropy,
            f"{prefix}load_balance": self.load_entropy / self.max_entropy if self.max_entropy else 0.0,
            f"{prefix}load_max": float(self.expert_load.max()),
            f"{prefix}load_min": float(self.expert_load.min()),
        }
        for e, v in enumerate(self.expert_load.tolist()):
            d[f"{prefix}load/expert_{e}"] = v
        return d


def _entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Shannon entropy (nats) of a distribution, safe at p=0."""
    p = probs.clamp_min(0)
    logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
    return -(p * logp).sum(dim=dim)


@torch.no_grad()
def routing_stats_from_logits(logits: torch.Tensor, top_k: int) -> RoutingStats:
    """Compute routing stats for ONE layer's router logits `[N_tokens, E]`."""
    if logits.ndim != 2:
        raise ValueError(f"expected [N_tokens, E] logits, got shape {tuple(logits.shape)}")
    logits = logits.float()
    n_tokens, n_experts = logits.shape
    if not (1 <= top_k <= n_experts):
        raise ValueError(f"top_k must be in [1, {n_experts}], got {top_k}")

    probs = torch.softmax(logits, dim=-1)                       # [N, E]
    gate_entropy = _entropy(probs, dim=-1).mean().item()

    # hard load: each token dispatches to its top-k experts (equal weight per assignment)
    _, top_idx = probs.topk(top_k, dim=-1)                      # [N, k]
    counts = torch.bincount(top_idx.reshape(-1), minlength=n_experts).float()
    expert_load = counts / counts.sum()                         # [E], sums to 1

    load_entropy = _entropy(expert_load, dim=0).item()

    return RoutingStats(
        expert_load=expert_load,
        gate_entropy=gate_entropy,
        load_entropy=load_entropy,
        max_entropy=math.log(n_experts),
        n_tokens=n_tokens,
        n_experts=n_experts,
        top_k=top_k,
    )


@torch.no_grad()
def aggregate_routing_stats(
    router_logits: tuple[torch.Tensor, ...] | list[torch.Tensor],
    top_k: int,
) -> tuple[list[RoutingStats], RoutingStats]:
    """Per-layer stats + a model-level aggregate over ALL tokens of ALL layers.

    Returns ``(per_layer, overall)``. The overall aggregate stacks every layer's tokens, so
    its expert_load/entropies reflect the whole model's routing behavior.
    """
    if not router_logits:
        raise ValueError("router_logits is empty — was output_router_logits=True?")
    per_layer = [routing_stats_from_logits(l, top_k) for l in router_logits]
    overall = routing_stats_from_logits(torch.cat([l.float() for l in router_logits], dim=0), top_k)
    return per_layer, overall


@torch.no_grad()
def selected_experts(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Return the top-k expert ids per token, `[N_tokens, top_k]` (for specialization analysis)."""
    return logits.float().topk(top_k, dim=-1).indices
