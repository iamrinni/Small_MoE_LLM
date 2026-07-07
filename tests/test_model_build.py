"""Step 2.2 tests — baseline builder: forward shape [B,T,V], bf16 params, aux loss with labels."""

from pathlib import Path

import torch
from transformers import OlmoeForCausalLM

from src.model.build import build_model, count_parameters, resolve_dtype
from src.model.config import load_model_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def _tiny_model():
    """Build from the smoke config (tiny + fast) for forward/backward tests."""
    cfg = load_model_config(CONFIGS / "smoke.yaml")
    return build_model(cfg), cfg


def test_builds_olmoe():
    model, _ = _tiny_model()
    assert isinstance(model, OlmoeForCausalLM)


def test_dtype_policy_is_device_aware():
    """bf16 targeted on GPU (spec); fp32 on CPU (bf16 emulation ~67x slower, measured)."""
    _, cfg = _tiny_model()
    assert cfg.dtype_ext == "bfloat16"                       # config still targets bf16
    assert resolve_dtype(cfg, "cuda") == torch.bfloat16      # honored on GPU
    assert resolve_dtype(cfg, "cpu") == torch.float32        # fp32 fallback on CPU


def test_cpu_model_is_fp32():
    """Built on CPU (test env) → params are fp32 per the device-aware policy."""
    model, _ = _tiny_model()  # no GPU in test env → CPU
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_forward_shape_BTV():
    model, cfg = _tiny_model()
    B, T = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    out = model(input_ids=ids)
    assert out.logits.shape == (B, T, cfg.vocab_size)


def test_output_router_logits_enabled():
    model, cfg = _tiny_model()
    assert cfg.output_router_logits is True
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=ids)
    # one router_logits tensor per layer
    assert out.router_logits is not None
    assert len(out.router_logits) == cfg.num_hidden_layers


def test_aux_loss_present_with_labels():
    """With labels, loss exists and the load-balance aux term is computed & folded in.

    NOTE: we check the explicit `aux_loss` field rather than diffing the total loss —
    in bf16 the aux contribution (coef * aux ~ 0.02) is below the rounding resolution of
    a ~10.8 CE, so the rounded total looks unchanged even though aux IS added.
    """
    model, cfg = _tiny_model()
    assert cfg.router_aux_loss_coef > 0
    ids = torch.randint(0, cfg.vocab_size, (2, 16))

    out = model(input_ids=ids, labels=ids)
    assert out.loss is not None and torch.isfinite(out.loss)
    # HF returns the (unweighted) aux loss separately; it must be present and positive.
    assert out.aux_loss is not None
    assert torch.isfinite(out.aux_loss) and out.aux_loss > 0


def test_backward_runs_in_bf16():
    model, cfg = _tiny_model()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_param_count_reported():
    model, _ = _tiny_model()
    total, trainable = count_parameters(model)
    assert total > 0 and trainable > 0
