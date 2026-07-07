"""Step 2.3 tests — routing instrumentation: load sums to ~1, exactly top-k active, entropy bounds."""

import math
from pathlib import Path

import pytest
import torch

from src.model.build import build_model
from src.model.config import load_model_config
from src.model.routing import (
    aggregate_routing_stats,
    routing_stats_from_logits,
    selected_experts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def test_load_sums_to_one():
    torch.manual_seed(0)
    logits = torch.randn(100, 8)
    stats = routing_stats_from_logits(logits, top_k=2)
    assert stats.expert_load.shape == (8,)
    assert math.isclose(stats.expert_load.sum().item(), 1.0, abs_tol=1e-5)


def test_exactly_top_k_active_per_token():
    torch.manual_seed(0)
    logits = torch.randn(50, 8)
    sel = selected_experts(logits, top_k=2)
    assert sel.shape == (50, 2)
    # each token's selected experts are distinct → exactly 2 active per token
    assert all(len(set(row.tolist())) == 2 for row in sel)


def test_entropy_bounds():
    torch.manual_seed(0)
    logits = torch.randn(200, 8)
    stats = routing_stats_from_logits(logits, top_k=2)
    log_e = math.log(8)
    assert 0.0 <= stats.gate_entropy <= log_e + 1e-6
    assert 0.0 <= stats.load_entropy <= log_e + 1e-6
    assert math.isclose(stats.max_entropy, log_e)


def test_uniform_logits_give_max_entropy():
    """Equal logits → uniform softmax → gate entropy == log E; balanced load → load entropy == log E."""
    logits = torch.zeros(64, 8)            # all experts equally likely
    stats = routing_stats_from_logits(logits, top_k=2)
    assert math.isclose(stats.gate_entropy, math.log(8), abs_tol=1e-4)
    # with ties, top-2 selection is deterministic but load should still be near-balanced-ish;
    # the gate entropy is the robust uniform check.


def test_collapsed_routing_low_load_entropy():
    """If one expert dominates the logits, load collapses → load_entropy near 0."""
    logits = torch.full((100, 8), -10.0)
    logits[:, 3] = 10.0                    # expert 3 always top-1
    logits[:, 5] = 5.0                     # expert 5 always top-2
    stats = routing_stats_from_logits(logits, top_k=2)
    # only experts 3 and 5 used, equally → load entropy ~= log(2), far below log(8)
    assert stats.load_entropy < math.log(8) - 0.5
    assert stats.expert_load[3] > 0 and stats.expert_load[5] > 0
    assert math.isclose(stats.expert_load.sum().item(), 1.0, abs_tol=1e-5)


def test_validation_errors():
    with pytest.raises(ValueError):
        routing_stats_from_logits(torch.randn(8), top_k=2)        # wrong ndim
    with pytest.raises(ValueError):
        routing_stats_from_logits(torch.randn(10, 8), top_k=9)    # k > E


def test_aggregate_empty_errors():
    with pytest.raises(ValueError):
        aggregate_routing_stats([], top_k=2)


def test_on_real_model_router_logits():
    """End-to-end: pull router_logits from a built model and aggregate stats."""
    cfg = load_model_config(CONFIGS / "smoke.yaml")
    model = build_model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=ids)

    per_layer, overall = aggregate_routing_stats(out.router_logits, cfg.num_experts_per_tok)
    assert len(per_layer) == cfg.num_hidden_layers
    assert overall.n_experts == cfg.num_experts
    assert overall.top_k == cfg.num_experts_per_tok
    assert math.isclose(overall.expert_load.sum().item(), 1.0, abs_tol=1e-5)
    assert 0.0 <= overall.gate_entropy <= math.log(cfg.num_experts) + 1e-6

    # logging dict is flat & scalar
    d = overall.to_dict(prefix="routing/")
    assert "routing/gate_entropy" in d
    assert sum(1 for k in d if k.startswith("routing/load/expert_")) == cfg.num_experts
