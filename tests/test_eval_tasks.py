"""Step 5.4 tests — task runners produce well-formed metrics on the tiny model."""

from pathlib import Path

import pytest

from src.data.sources import synthetic_modality, take
from src.data.tokenizer import build_tokenizer
from src.eval.tasks import eval_code, eval_logic, eval_math, eval_code_pass_at_k
from src.model.config import SmallMoEConfig
from src.model.model import SmallMoE


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


def test_eval_math(model, tok):
    r = eval_math(model, tok, take(synthetic_modality("math", 4), 4), max_new_tokens=8)
    assert r["task"] == "math" and r["n"] == 4
    assert 0.0 <= r["exact_match"] <= 1.0


def test_eval_logic(model, tok):
    r = eval_logic(model, tok, take(synthetic_modality("logic", 4), 4))
    assert r["task"] == "logic" and 0.0 <= r["accuracy"] <= 1.0


def test_eval_code(model, tok):
    r = eval_code(model, tok, take(synthetic_modality("code", 4), 4),
                  prompt_tokens=4, max_new_tokens=8)
    assert r["task"] == "code"
    if r["n"]:
        assert 0.0 <= r["codebleu"] <= 1.0


def test_pass_at_k_runner(model, tok):
    problems = [{"prompt": "def add(a,b):", "check": lambda code: "return" in code}]
    r = eval_code_pass_at_k(model, tok, problems, k=1, n_samples=3, max_new_tokens=8)
    assert r["task"] == "code_pass_at_k"
    assert 0.0 <= r["pass@1"] <= 1.0
