"""Step 2.1 tests — config layer: build from YAML, locked invariants, checkpoint round-trip."""

import tempfile
from pathlib import Path

import pytest
from transformers import AutoConfig, OlmoeConfig

from src.model.config import (
    SmallMoEConfig,
    build_model_config,
    load_model_config,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def test_is_olmoe_config():
    cfg = load_model_config(CONFIGS / "model_small.yaml")
    assert isinstance(cfg, OlmoeConfig)
    assert isinstance(cfg, SmallMoEConfig)


def test_locked_default_values():
    """Guard the locked decisions against drift in the default model config."""
    cfg = load_model_config(CONFIGS / "model_small.yaml")
    assert cfg.num_experts == 8
    assert cfg.num_experts_per_tok == 2            # top-2 gating
    assert cfg.expert_activation_ext == "swiglu"   # SwiGLU (HF-native, spec-compliant)
    assert cfg.pos_encoding_ext == "rope"          # RoPE default
    assert cfg.dtype_ext == "bfloat16"             # strict bf16


def test_extension_fields_present():
    cfg = load_model_config(CONFIGS / "model_small.yaml")
    for field in ("pos_encoding_ext", "expert_activation_ext", "dtype_ext", "router_z_loss_coef"):
        assert hasattr(cfg, field)
    assert cfg.router_z_loss_coef == 0.0           # dormant: not enabled anywhere


def test_run_config_with_overrides():
    """smoke.yaml references a base via defaults + applies model_overrides."""
    cfg = load_model_config(CONFIGS / "smoke.yaml")
    assert cfg.hidden_size == 64                    # from model_overrides
    assert cfg.num_experts == 8                     # inherited from base
    assert cfg.expert_activation_ext == "swiglu"    # inherited extension default


def test_roundtrip_save_load(tmp_path: Path):
    """save_pretrained -> from_pretrained must preserve OLMoE fields AND our 4 extras."""
    cfg = load_model_config(CONFIGS / "model_small.yaml")
    cfg.save_pretrained(tmp_path)
    reloaded = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(reloaded, SmallMoEConfig)
    assert reloaded.num_experts == cfg.num_experts
    assert reloaded.num_experts_per_tok == cfg.num_experts_per_tok
    assert reloaded.pos_encoding_ext == cfg.pos_encoding_ext
    assert reloaded.expert_activation_ext == cfg.expert_activation_ext
    assert reloaded.dtype_ext == cfg.dtype_ext
    assert reloaded.router_z_loss_coef == cfg.router_z_loss_coef


@pytest.mark.parametrize(
    "bad",
    [
        {"num_experts": 4, "num_experts_per_tok": 8},   # top-k > experts
        {"pos_encoding_ext": "sinusoidal"},             # invalid enum
        {"expert_activation_ext": "relu"},              # invalid enum
        {"dtype_ext": "int8"},                          # invalid enum
        {"hidden_size": 65, "num_attention_heads": 8},  # not divisible
    ],
)
def test_validation_rejects_bad_config(bad):
    base = dict(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=8,
        num_experts=8, num_experts_per_tok=2, vocab_size=256,
    )
    base.update(bad)
    with pytest.raises(ValueError):
        build_model_config(base)


def test_validate_returns_config():
    cfg = SmallMoEConfig(hidden_size=64, num_attention_heads=8, num_key_value_heads=8,
                         num_experts=8, num_experts_per_tok=2)
    assert validate_config(cfg) is cfg
