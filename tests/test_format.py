"""Step 3.3 tests — formatting: prompt masked, completion trained, tag preserved, truncation."""

import pytest

from src.data.format import IGNORE_INDEX, encode_example, has_trainable_tokens
from src.data.sources import Example
from src.data.tokenizer import build_tokenizer, modality_token_id


@pytest.fixture(scope="module")
def tok():
    try:
        return build_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


def test_input_and_labels_same_length(tok):
    enc = encode_example(tok, Example("math", "Question: 2+2?\nAnswer:", " 4"), max_len=64)
    assert len(enc["input_ids"]) == len(enc["labels"])


def test_tag_is_first_and_masked(tok):
    ex = Example("code", "", "print(1)")
    enc = encode_example(tok, ex, max_len=64)
    assert enc["input_ids"][0] == modality_token_id(tok, "code")
    assert enc["labels"][0] == IGNORE_INDEX            # tag never contributes to loss


def test_prompt_masked_completion_trained(tok):
    ex = Example("math", "Question: 2+2?\nAnswer:", " 4")
    enc = encode_example(tok, ex, max_len=64)
    n_prompt = len(tok.encode(ex.prompt, add_special_tokens=False))
    # positions 0..n_prompt (tag + prompt) are masked
    assert all(l == IGNORE_INDEX for l in enc["labels"][: 1 + n_prompt])
    # the completion region has real (non-masked) labels
    assert any(l != IGNORE_INDEX for l in enc["labels"][1 + n_prompt :])


def test_plain_text_trains_on_everything_after_tag(tok):
    enc = encode_example(tok, Example("language", "", "hello world"), max_len=64)
    assert enc["labels"][0] == IGNORE_INDEX            # only the tag masked
    assert all(l != IGNORE_INDEX for l in enc["labels"][1:])


def test_ends_with_eos_trained(tok):
    enc = encode_example(tok, Example("code", "", "x=1"), max_len=64)
    assert enc["input_ids"][-1] == tok.eos_token_id
    assert enc["labels"][-1] == tok.eos_token_id       # EOS is trained


def test_truncation_preserves_tag_and_respects_max_len(tok):
    long_completion = " ".join(["word"] * 500)
    enc = encode_example(tok, Example("language", "", long_completion), max_len=32)
    assert len(enc["input_ids"]) == 32
    assert enc["input_ids"][0] == modality_token_id(tok, "language")  # tag kept


def test_has_trainable_tokens(tok):
    enc = encode_example(tok, Example("code", "", "x=1"), max_len=64)
    assert has_trainable_tokens(enc)
