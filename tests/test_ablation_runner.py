"""Phase 6 tests — ablation runner produces comparable metrics for variants."""

import pytest

from src.data.tokenizer import build_tokenizer
from src.eval.ablation import results_to_markdown, run_variant


@pytest.fixture(scope="module")
def tok():
    try:
        return build_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


def test_run_single_variant(tok):
    r = run_variant("baseline", tok, steps=12)
    assert r.name == "baseline"
    assert r.ce > 0 and r.perplexity > 1
    assert 0.0 <= r.load_balance <= 1.0
    assert 0.0 <= r.specialization <= 2.0
    assert r.total_params > 0


def test_variant_overrides_apply(tok):
    r16 = run_variant("16E", tok, model_overrides={"num_experts": 16}, steps=8)
    r1 = run_variant("top1", tok, model_overrides={"num_experts_per_tok": 1}, steps=8)
    assert r16.total_params > r1.total_params        # 16 experts → more params


def test_results_to_markdown(tok):
    r = run_variant("baseline", tok, steps=8)
    md = results_to_markdown([r])
    assert "Ablation results" in md and "baseline" in md and "specialization" in md
