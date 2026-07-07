"""Step 4.1 tests — DataLoader: batch shapes, re-iterability, caps."""

from pathlib import Path

import pytest
import torch

from src.data.tokenizer import MODALITIES, build_tokenizer
from src.training.data import (
    MixtureIterableDataset,
    build_dataloader,
    synthetic_source_factory,
)

WEIGHTS = {"language": 0.4, "code": 0.25, "math": 0.2, "logic": 0.15}


@pytest.fixture(scope="module")
def tok():
    try:
        return build_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


def _dataset(tok, max_examples=20):
    return MixtureIterableDataset(
        synthetic_source_factory(MODALITIES, 1000),
        WEIGHTS, tok, max_len=64, seed=0, max_examples=max_examples,
    )


def test_dataset_yields_encoded(tok):
    ds = _dataset(tok, max_examples=10)
    items = list(ds)
    assert len(items) == 10
    assert all("input_ids" in x and "labels" in x for x in items)


def test_dataset_is_reiterable(tok):
    ds = _dataset(tok, max_examples=10)
    first = [x["input_ids"] for x in ds]
    second = [x["input_ids"] for x in ds]      # fresh pass, same seed
    assert first == second and len(first) == 10


def test_dataloader_batch_shapes(tok):
    ds = _dataset(tok, max_examples=16)
    dl = build_dataloader(ds, tok, batch_size=4, max_len=64)
    batch = next(iter(dl))
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].dtype == torch.long


def test_max_examples_cap(tok):
    ds = _dataset(tok, max_examples=7)
    assert len(list(ds)) == 7
