"""Phase 6 tests — GeLU expert + learnable PE variants build, forward, and learn."""

import torch

from src.model.ablations import GeLUExpert, apply_gelu_experts, apply_learnable_pos_encoding
from src.model.build import build_model
from src.model.config import SmallMoEConfig


def _cfg(**kw):
    base = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=4, num_experts=8,
                num_experts_per_tok=2, vocab_size=64, max_position_embeddings=32,
                dtype_ext="float32")
    base.update(kw)
    return SmallMoEConfig(**base)


def test_gelu_expert_forward():
    exp = GeLUExpert(16, 32)
    out = exp(torch.randn(4, 16))
    assert out.shape == (4, 16)


def test_gelu_experts_swapped():
    model = build_model(_cfg(expert_activation_ext="gelu"), device="cpu")
    expert = model.model.layers[0].mlp.experts[0]
    assert isinstance(expert, GeLUExpert)
    assert not hasattr(expert, "gate_proj")            # 2-matrix, no gate
    ids = torch.randint(0, 64, (2, 16))
    assert model(input_ids=ids).logits.shape == (2, 16, 64)


def test_gelu_params_roughly_matched():
    swiglu = build_model(_cfg(expert_activation_ext="swiglu"), device="cpu")
    gelu = build_model(_cfg(expert_activation_ext="gelu"), device="cpu")
    s = sum(p.numel() for p in swiglu.parameters())
    g = sum(p.numel() for p in gelu.parameters())
    assert abs(s - g) / s < 0.05                        # within 5% (param-matched)


def test_learnable_pe_builds_and_has_submodule():
    model = build_model(_cfg(pos_encoding_ext="learnable"), device="cpu")
    assert hasattr(model.model, "learnable_pe")
    ids = torch.randint(0, 64, (2, 16))
    assert model(input_ids=ids).logits.shape == (2, 16, 64)


def test_learnable_pe_changes_with_position():
    """With learnable PE + neutralized RoPE, position still affects the output."""
    model = build_model(_cfg(pos_encoding_ext="learnable"), device="cpu")
    model.eval()
    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = model(input_ids=ids.flip(1)).logits          # reversed positions
    assert not torch.allclose(a, b.flip(1), atol=1e-4)    # position matters


def test_gelu_variant_overfits_tiny_batch():
    from src.model.model import SmallMoE
    from src.utils.seed import set_seed

    set_seed(0)
    model = SmallMoE(_cfg(expert_activation_ext="gelu"), device="cpu")
    ids = torch.randint(0, 64, (2, 8))
    opt = torch.optim.AdamW(model.model.parameters(), lr=1e-2)
    for _ in range(150):
        out = model(input_ids=ids, labels=ids, collect_routing=False)
        opt.zero_grad(); out.loss.backward(); opt.step()
    assert float(out.loss_breakdown.ce) < 0.2            # GeLU variant learns too
