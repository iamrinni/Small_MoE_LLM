"""Step 3.1 tests — tokenizer wrapper: modality tags, single-token, round-trip, vocab fits.

Requires network on first run to fetch the OLMoE tokenizer (then cached). Skips cleanly if
the tokenizer can't be downloaded so the offline suite stays green.
"""

import pytest

from src.data.tokenizer import (
    MODALITIES,
    MODALITY_TOKENS,
    build_tokenizer,
    ensure_vocab_fits,
    modality_tag,
    modality_token_id,
)


@pytest.fixture(scope="module")
def tok():
    try:
        return build_tokenizer()
    except Exception as e:  # offline / network failure
        pytest.skip(f"tokenizer unavailable: {e}")


def test_modalities_are_four():
    assert set(MODALITIES) == {"language", "code", "logic", "math"}


def test_pad_token_set(tok):
    assert tok.pad_token is not None


def test_modality_tags_are_single_tokens(tok):
    for m in MODALITIES:
        assert modality_token_id(tok, m) is not None  # raises if not single-token


def test_modality_tag_lookup():
    assert modality_tag("code") == "<|code|>"
    with pytest.raises(KeyError):
        modality_tag("vision")


def test_encode_decode_roundtrip(tok):
    text = "def add(a, b):\n    return a + b"
    ids = tok.encode(text, add_special_tokens=False)
    assert tok.decode(ids) == text


def test_tag_prepend_roundtrips(tok):
    ids = tok.encode(modality_tag("math") + " 2+2=4", add_special_tokens=False)
    assert modality_token_id(tok, "math") == ids[0]


def test_vocab_fits_model_embedding(tok):
    ensure_vocab_fits(tok, 50304)              # our model_small vocab_size
    with pytest.raises(ValueError):
        ensure_vocab_fits(tok, 10)             # too small → must raise
