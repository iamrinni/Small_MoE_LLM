"""Phase 6 — architecture ablation variants that need real code (not just config flags).

  * `apply_gelu_experts`        — swap HF's SwiGLU experts (`down(silu(gate(x))*up(x))`, 3
    matrices) for 2-matrix **GeLU MLP** experts (`down(gelu(up(x)))`), param-matched.
  * `apply_learnable_pos_encoding` — neutralize OLMoE's RoPE (identity rotation) and add a
    trainable absolute **position embedding** to the token embeddings.

Both operate on a built `OlmoeForCausalLM`; `build_model` calls them based on the config
flags `expert_activation_ext` / `pos_encoding_ext`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeLUExpert(nn.Module):
    """A single 2-matrix GeLU MLP expert: down(gelu(up(x)))."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


def apply_gelu_experts(model, intermediate_size: int | None = None):
    """Replace every SwiGLU expert with a param-matched GeLU MLP expert."""
    olmoe = model.model  # OlmoeModel
    example = next(model.parameters())
    for layer in olmoe.layers:
        experts = layer.mlp.experts
        hidden = experts[0].gate_proj.in_features
        swiglu_inter = experts[0].gate_proj.out_features
        # GeLU has 2 matrices vs SwiGLU's 3 → 1.5x width keeps params matched
        inter = intermediate_size or int(round(1.5 * swiglu_inter))
        new = nn.ModuleList([GeLUExpert(hidden, inter) for _ in experts])
        new.to(device=example.device, dtype=example.dtype)
        layer.mlp.experts = new
    model.config.expert_activation_ext = "gelu"
    return model


class LearnablePositionalEncoding(nn.Module):
    """Trainable absolute position embedding, added to token embeddings via a forward hook."""

    def __init__(self, max_positions: int, hidden_size: int) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_positions, hidden_size)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def add_to(self, embeddings: torch.Tensor) -> torch.Tensor:
        seq_len = embeddings.shape[1]
        pos = torch.arange(seq_len, device=embeddings.device)
        return embeddings + self.pos_emb(pos).unsqueeze(0)


def apply_learnable_pos_encoding(model):
    """Neutralize RoPE (identity rotation) and add a trainable position embedding."""
    olmoe = model.model
    hidden = olmoe.config.hidden_size
    max_pos = olmoe.config.max_position_embeddings
    example = next(model.parameters())

    lpe = LearnablePositionalEncoding(max_pos, hidden).to(example.device, example.dtype)
    olmoe.learnable_pe = lpe  # register as a submodule → trained + saved in state_dict

    def _add_pos_hook(_module, _inp, out):
        return lpe.add_to(out)

    olmoe.embed_tokens.register_forward_hook(_add_pos_hook)

    # Neutralize RoPE: make the rotary embedding return cos=1, sin=0 (identity rotation),
    # so apply_rotary_pos_emb becomes a no-op and only the learnable PE carries position.
    rotary = olmoe.rotary_emb
    orig_forward = rotary.forward

    def _identity_rope(x, position_ids):
        cos, sin = orig_forward(x, position_ids)
        return torch.ones_like(cos), torch.zeros_like(sin)

    rotary.forward = _identity_rope
    model.config.pos_encoding_ext = "learnable"
    return model
