"""Step 2.5 tests — positional-encoding switch: rope works, learnable flag recognized but deferred."""

from pathlib import Path

import pytest
import torch

from src.model.build import build_model
from src.model.config import SmallMoEConfig, load_model_config
from src.model.pos_encoding import configure_positional_encoding

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def test_rope_is_default_and_active():
    cfg = load_model_config(CONFIGS / "model_small.yaml")
    assert cfg.pos_encoding_ext == "rope"
    assert configure_positional_encoding(cfg) == "rope"
    assert cfg.rope_theta is not None


def test_rope_model_builds_and_forwards():
    cfg = load_model_config(CONFIGS / "smoke.yaml")
    model = build_model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)


def test_learnable_flag_recognized():
    cfg = SmallMoEConfig(hidden_size=64, num_attention_heads=8, num_key_value_heads=8,
                         num_experts=8, num_experts_per_tok=2, pos_encoding_ext="learnable")
    assert configure_positional_encoding(cfg) == "learnable"


def test_build_learnable_model_forwards():
    """Learnable PE is implemented (Phase 6): the model builds and forwards."""
    import torch

    cfg = load_model_config(CONFIGS / "smoke.yaml")
    cfg.pos_encoding_ext = "learnable"
    model = build_model(cfg)
    # learnable position embedding submodule was registered
    assert hasattr(model.model, "learnable_pe")
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
