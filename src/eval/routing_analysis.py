"""Step 5.3 — MoE routing analysis (adapted from allenai/OLMoE's routing analysis).

Answers the project's central question: *do experts specialize by modality?* Runs the model
over examples grouped by modality, aggregates per-expert load from `router_logits`, and
builds a **modality × expert** matrix — the specialization heatmap the report needs.

Also reports per-modality gate entropy and a scalar `specialization_score` = mean L1 distance
of each modality's expert-load row from the uniform distribution (0 = no specialization,
higher = experts increasingly modality-specific).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from src.data.collate import pad_collate
from src.data.format import encode_example, has_trainable_tokens
from src.data.sources import Example
from src.model.model import SmallMoE


@dataclass
class RoutingAnalysis:
    modalities: list[str]
    expert_load_matrix: np.ndarray          # [n_modalities, n_experts], rows sum to 1
    gate_entropy: dict[str, float]          # per modality
    n_experts: int
    top_k: int

    def specialization_score(self) -> float:
        """Mean L1 distance of each row from uniform (0 = none, up to ~2*(1-1/E))."""
        uniform = np.full(self.n_experts, 1.0 / self.n_experts)
        return float(np.mean([np.abs(row - uniform).sum() for row in self.expert_load_matrix]))

    def to_dict(self) -> dict:
        return {
            "modalities": self.modalities,
            "expert_load_matrix": self.expert_load_matrix.tolist(),
            "gate_entropy": self.gate_entropy,
            "specialization_score": self.specialization_score(),
            "n_experts": self.n_experts,
            "top_k": self.top_k,
        }


@torch.no_grad()
def analyze_routing(
    model: SmallMoE,
    tokenizer: PreTrainedTokenizerBase,
    examples_by_modality: dict[str, list[Example]],
    max_len: int,
    max_examples: int = 64,
) -> RoutingAnalysis:
    """Build the modality×expert load matrix by forwarding examples per modality."""
    model.model.eval()
    modalities = list(examples_by_modality)
    rows, gate_entropy = [], {}

    for modality in modalities:
        encoded = []
        for ex in examples_by_modality[modality][:max_examples]:
            enc = encode_example(tokenizer, ex, max_len)
            if has_trainable_tokens(enc):
                encoded.append(enc)
        if not encoded:
            rows.append(np.full(model.cfg.num_experts, 1.0 / model.cfg.num_experts))
            gate_entropy[modality] = 0.0
            continue

        batch = pad_collate(encoded, pad_token_id=tokenizer.pad_token_id, max_len=max_len)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    collect_routing=True)
        rows.append(out.routing.expert_load.cpu().numpy())
        gate_entropy[modality] = out.routing.gate_entropy

    return RoutingAnalysis(
        modalities=modalities,
        expert_load_matrix=np.vstack(rows),
        gate_entropy=gate_entropy,
        n_experts=model.cfg.num_experts,
        top_k=model.cfg.num_experts_per_tok,
    )


def save_heatmap(analysis: RoutingAnalysis, path) -> None:
    """Save the modality×expert load heatmap to `path` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.2 * analysis.n_experts, 0.8 * len(analysis.modalities) + 1))
    im = ax.imshow(analysis.expert_load_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(analysis.n_experts))
    ax.set_xticklabels([f"E{i}" for i in range(analysis.n_experts)])
    ax.set_yticks(range(len(analysis.modalities)))
    ax.set_yticklabels(analysis.modalities)
    ax.set_xlabel("expert"); ax.set_ylabel("modality")
    ax.set_title(f"Expert load by modality (spec. score={analysis.specialization_score():.3f})")
    fig.colorbar(im, ax=ax, label="fraction of tokens")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
