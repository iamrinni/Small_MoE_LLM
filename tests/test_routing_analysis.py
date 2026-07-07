"""Step 5.3 tests — routing analysis: modality×expert matrix, specialization score, heatmap."""

import math
from pathlib import Path

import pytest

from src.data.sources import synthetic_modality, take
from src.data.tokenizer import MODALITIES, build_tokenizer
from src.eval.routing_analysis import analyze_routing, save_heatmap
from src.model.config import SmallMoEConfig
from src.model.model import SmallMoE

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tok():
    try:
        return build_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def model(tok):
    cfg = SmallMoEConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, num_experts=8,
                         num_experts_per_tok=2, vocab_size=len(tok),
                         max_position_embeddings=64, dtype_ext="float32")
    return SmallMoE(cfg, device="cpu")


def _examples():
    return {m: take(synthetic_modality(m, 16), 16) for m in MODALITIES}


def test_matrix_shape_and_rows_sum_to_one(model, tok):
    a = analyze_routing(model, tok, _examples(), max_len=48, max_examples=16)
    assert a.expert_load_matrix.shape == (len(MODALITIES), model.cfg.num_experts)
    for row in a.expert_load_matrix:
        assert math.isclose(row.sum(), 1.0, abs_tol=1e-5)


def test_gate_entropy_per_modality(model, tok):
    a = analyze_routing(model, tok, _examples(), max_len=48)
    assert set(a.gate_entropy) == set(MODALITIES)
    log_e = math.log(model.cfg.num_experts)
    assert all(0 <= v <= log_e + 1e-6 for v in a.gate_entropy.values())


def test_specialization_score_bounds(model, tok):
    a = analyze_routing(model, tok, _examples(), max_len=48)
    score = a.specialization_score()
    assert 0.0 <= score <= 2.0                        # L1-from-uniform range


def test_to_dict_serializable(model, tok):
    import json
    a = analyze_routing(model, tok, _examples(), max_len=48)
    json.dumps(a.to_dict())                            # must not raise


def test_heatmap_saved(model, tok, tmp_path: Path):
    a = analyze_routing(model, tok, _examples(), max_len=48)
    p = tmp_path / "heatmap.png"
    save_heatmap(a, p)
    assert p.exists() and p.stat().st_size > 0
