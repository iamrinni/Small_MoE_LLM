"""Phase-3 integration — full pipeline on the REAL downloaded subsample (data/raw/*.jsonl).

Skips cleanly if the data hasn't been prepared or the tokenizer is unavailable, so the
offline suite stays green. Exercises: local JSONL → mixture → encode → collate → model.
"""

from pathlib import Path

import pytest
import torch

from src.data.collate import pad_collate
from src.data.mixture import MixtureSampler
from src.data.pipeline import encoded_stream, make_local_sources
from src.data.sources import iter_local_examples, take
from src.data.tokenizer import MODALITIES, build_tokenizer
from src.model.model import SmallMoE

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw"

MIX = {"language": 0.4, "code": 0.25, "math": 0.2, "logic": 0.15}


def _require_data():
    missing = [m for m in MODALITIES if not (DATA_DIR / f"{m}.jsonl").exists()]
    if missing:
        pytest.skip(f"prepared data missing for {missing} (run scripts/prepare_data.py)")


@pytest.fixture(scope="module")
def tok():
    try:
        return build_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


def test_local_examples_load_and_are_wellformed():
    _require_data()
    for m in MODALITIES:
        exs = take(iter_local_examples(DATA_DIR / f"{m}.jsonl"), 10)
        assert len(exs) == 10
        assert all(e.modality == m and e.completion for e in exs)


def test_real_examples_encode_with_prompt_mask(tok):
    _require_data()
    # a math example: prompt (question) masked, completion (answer) trained
    from src.data.format import IGNORE_INDEX, encode_example

    ex = next(iter_local_examples(DATA_DIR / "math.jsonl"))
    enc = encode_example(tok, ex, max_len=256)
    assert enc["labels"][0] == IGNORE_INDEX                      # tag masked
    assert any(l != IGNORE_INDEX for l in enc["labels"])         # answer trained
    assert len(enc["input_ids"]) == len(enc["labels"])


def test_full_pipeline_into_model(tok):
    """Real data → mixture → encode → collate → SmallMoE forward; loss + routing valid."""
    _require_data()
    sources = make_local_sources(DATA_DIR, MODALITIES)
    sampler = MixtureSampler(sources, MIX, seed=0)
    encoded = take(encoded_stream(sampler, tok, max_len=128), 8)
    assert len(encoded) == 8

    batch = pad_collate(encoded, pad_token_id=tok.pad_token_id, max_len=128)

    # tiny model sized to the real tokenizer vocab
    model = SmallMoE.from_yaml(REPO_ROOT / "configs" / "smoke.yaml")
    model.cfg.vocab_size = len(tok)
    model = SmallMoE(model.cfg, device="cpu")  # rebuild with correct vocab

    out = model(input_ids=batch["input_ids"], labels=batch["labels"],
                attention_mask=batch["attention_mask"])
    assert torch.isfinite(out.loss)
    assert out.routing is not None
    assert abs(out.routing.expert_load.sum().item() - 1.0) < 1e-5


def test_realized_mixture_on_real_data(tok):
    _require_data()
    sources = make_local_sources(DATA_DIR, MODALITIES)
    sampler = MixtureSampler(sources, MIX, seed=1)
    take(encoded_stream(sampler, tok, max_len=128), 400)
    realized = sampler.realized_mixture()
    # loose bound — small finite sources, but language should dominate, logic least
    assert realized["language"] > realized["logic"]
