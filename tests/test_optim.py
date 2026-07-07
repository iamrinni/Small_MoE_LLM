"""Step 4.2 tests — optimizer groups + cosine-warmup schedule shape."""

import math

import torch

from src.model.config import SmallMoEConfig
from src.model.build import build_model
from src.training.optim import build_optimizer, build_scheduler, cosine_warmup_lambda


def _tiny_model():
    cfg = SmallMoEConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, num_experts=8,
                         num_experts_per_tok=2, vocab_size=64, max_position_embeddings=32,
                         dtype_ext="float32")
    return build_model(cfg, device="cpu")


def test_optimizer_splits_decay_groups():
    opt = build_optimizer(_tiny_model(), lr=1e-3, weight_decay=0.1)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_warmup_increases_then_cosine_decays():
    fn = cosine_warmup_lambda(warmup_steps=10, max_steps=110, min_ratio=0.1)
    # warmup: strictly increasing up to ~1.0
    assert fn(0) < fn(5) < fn(9)
    assert math.isclose(fn(9), 1.0, rel_tol=0.2)
    # peak near end of warmup
    assert fn(10) > 0.9
    # decays afterwards toward min_ratio
    assert fn(10) > fn(60) > fn(109)
    assert math.isclose(fn(110), 0.1, abs_tol=1e-6)      # floor at min_ratio


def test_scheduler_drives_lr_on_optimizer():
    opt = build_optimizer(_tiny_model(), lr=1e-3)
    sched = build_scheduler(opt, warmup_steps=5, max_steps=50, lr=1e-3, min_lr=1e-4)
    lrs = []
    for _ in range(50):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert max(lrs) <= 1e-3 + 1e-9
    assert lrs[-1] < lrs[4]                                # decayed below early-warmup lr
    assert lrs[-1] >= 1e-4 - 1e-9                          # not below floor
