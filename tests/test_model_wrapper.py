"""Step 2.6 tests — SmallMoE wrapper: forward returns logits + loss + routing; round-trips."""

import math
from pathlib import Path

import torch

from src.model.model import SmallMoE, SmallMoEOutput

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def _wrapper():
    return SmallMoE.from_yaml(CONFIGS / "smoke.yaml")


def test_forward_returns_all_three():
    m = _wrapper()
    ids = torch.randint(0, m.cfg.vocab_size, (2, 16))
    out = m(input_ids=ids, labels=ids)
    assert isinstance(out, SmallMoEOutput)
    # 1) logits [B,T,V]
    assert out.logits.shape == (2, 16, m.cfg.vocab_size)
    # 2) loss (differentiable total)
    assert out.loss is not None and torch.isfinite(out.loss) and out.loss.requires_grad
    # 3) routing stats
    assert out.routing is not None
    assert math.isclose(out.routing.expert_load.sum().item(), 1.0, abs_tol=1e-5)
    assert out.routing_per_layer is not None and len(out.routing_per_layer) == m.cfg.num_hidden_layers


def test_forward_without_labels_has_no_loss_but_has_routing():
    m = _wrapper()
    ids = torch.randint(0, m.cfg.vocab_size, (2, 16))
    out = m(input_ids=ids)
    assert out.loss is None and out.loss_breakdown is None
    assert out.routing is not None


def test_metrics_dict_combines_loss_and_routing():
    m = _wrapper()
    ids = torch.randint(0, m.cfg.vocab_size, (2, 16))
    metrics = m(input_ids=ids, labels=ids).metrics()
    assert "loss/total" in metrics and "loss/aux" in metrics
    assert "routing/gate_entropy" in metrics and "routing/load_entropy" in metrics


def test_backward_through_wrapper():
    m = _wrapper()
    ids = torch.randint(0, m.cfg.vocab_size, (2, 16))
    m(input_ids=ids, labels=ids).loss.backward()
    grads = [p.grad for p in m.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_collect_routing_can_be_disabled():
    m = _wrapper()
    ids = torch.randint(0, m.cfg.vocab_size, (2, 16))
    out = m(input_ids=ids, labels=ids, collect_routing=False)
    assert out.routing is None and out.loss is not None


def test_properties_and_param_count():
    m = _wrapper()
    total, trainable = m.num_parameters()
    assert total > 0 and trainable > 0
    assert m.device.type == "cpu"
    assert m.dtype == torch.float32  # CPU dev → fp32 per device-aware policy


def test_save_and_load_roundtrip(tmp_path: Path):
    m = _wrapper()
    ids = torch.randint(0, m.cfg.vocab_size, (1, 8))
    m.model.eval()
    logits_before = m(input_ids=ids).logits

    m.save_pretrained(tmp_path)
    reloaded = SmallMoE.from_pretrained(tmp_path)
    reloaded.model.eval()
    logits_after = reloaded(input_ids=ids).logits

    assert torch.allclose(logits_before, logits_after, atol=1e-4)
