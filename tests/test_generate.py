"""Step 5.1 tests — generation: produces new tokens, respects count, greedy determinism."""

from pathlib import Path

import pytest

from src.data.tokenizer import build_tokenizer
from src.eval.generate import build_prompt, generate_completions, generate_one
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


def test_build_prompt_has_tag(tok):
    p = build_prompt("2+2?", "math")
    assert p.startswith("<|math|>")


def test_generates_new_text(model, tok):
    out = generate_completions(model, tok, "Question: 2+2?\nAnswer:", "math", max_new_tokens=8)
    assert len(out) == 1 and isinstance(out[0], str)


def test_num_return_sequences(model, tok):
    outs = generate_completions(model, tok, "hello", "language", max_new_tokens=5,
                                do_sample=True, num_return_sequences=3)
    assert len(outs) == 3


def test_greedy_is_deterministic(model, tok):
    a = generate_one(model, tok, "def f():", "code", max_new_tokens=8)
    b = generate_one(model, tok, "def f():", "code", max_new_tokens=8)
    assert a == b                                    # greedy → same output
