"""Step 2.4 tests — combined loss: finite, differentiable, aux present, breakdown consistent."""

import math
from pathlib import Path

import torch

from src.model.build import build_model
from src.model.config import load_model_config
from src.model.losses import compute_loss, cross_entropy_lm

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def _model_and_batch():
    cfg = load_model_config(CONFIGS / "smoke.yaml")
    model = build_model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    return model, cfg, ids


def test_total_finite_and_differentiable():
    model, cfg, ids = _model_and_batch()
    out = model(input_ids=ids)
    lb = compute_loss(out, labels=ids, cfg=cfg)
    assert torch.isfinite(lb.total) and lb.total.requires_grad
    lb.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_aux_term_present_and_positive():
    model, cfg, ids = _model_and_batch()
    out = model(input_ids=ids)
    lb = compute_loss(out, labels=ids, cfg=cfg)
    assert torch.isfinite(lb.aux) and lb.aux > 0          # the "Routing loss" metric
    assert math.isclose(float(lb.aux_weighted), cfg.router_aux_loss_coef * float(lb.aux), rel_tol=1e-5)


def test_breakdown_consistent():
    model, cfg, ids = _model_and_batch()
    out = model(input_ids=ids)
    lb = compute_loss(out, labels=ids, cfg=cfg)
    assert math.isclose(float(lb.total), float(lb.ce) + float(lb.aux_weighted), rel_tol=1e-5)


def test_ce_matches_hf_loss_when_aux_off():
    """With aux coef = 0, our CE should match HF's reported loss (both pure CE)."""
    model, cfg, ids = _model_and_batch()
    model.config.router_aux_loss_coef = 0.0
    out = model(input_ids=ids, labels=ids)
    ours = cross_entropy_lm(out.logits, ids)
    assert math.isclose(float(ours), float(out.loss), rel_tol=1e-2)


def test_to_dict_keys():
    model, cfg, ids = _model_and_batch()
    out = model(input_ids=ids)
    d = compute_loss(out, labels=ids, cfg=cfg).to_dict()
    assert set(d) == {"loss/total", "loss/ce", "loss/aux", "loss/aux_weighted"}
