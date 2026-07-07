"""Step 3.2 tests — modality adapters: uniform schema, synthetic mirror, validation."""

import pytest

from src.data.sources import (
    DEFAULT_SPECS,
    Example,
    synthetic_modality,
    take,
)
from src.data.tokenizer import MODALITIES


def test_example_validation():
    with pytest.raises(ValueError):
        Example("vision", "", "x")          # bad modality
    with pytest.raises(ValueError):
        Example("code", "p", "")            # empty completion


def test_synthetic_all_modalities_wellformed():
    for m in MODALITIES:
        exs = take(synthetic_modality(m, 5), 5)
        assert len(exs) == 5
        for ex in exs:
            assert ex.modality == m
            assert ex.completion                       # non-empty
            assert isinstance(ex.prompt, str)


def test_plain_text_has_empty_prompt():
    for m in ("language", "code"):
        ex = next(synthetic_modality(m, 1))
        assert ex.prompt == ""                          # loss on whole text


def test_qa_has_prompt_and_completion():
    for m in ("math", "logic"):
        ex = next(synthetic_modality(m, 1))
        assert ex.prompt and ex.completion              # prompt masked, completion trained
        assert "Answer:" in ex.prompt


def test_default_specs_cover_all_modalities():
    assert set(DEFAULT_SPECS) == set(MODALITIES)
    for spec in DEFAULT_SPECS.values():
        assert spec.hf_id and spec.split and callable(spec.mapper)
