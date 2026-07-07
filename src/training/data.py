"""Step 4.1 — DataLoader over the Phase-3 pipeline.

Wraps the mixture→encode stream in a re-iterable `IterableDataset` so the trainer can run
multiple epochs / restart cleanly. Sources are rebuilt on every `__iter__` via a factory
(local JSONL re-reads the file each pass), so iteration is repeatable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase

from src.data.collate import make_collate_fn
from src.data.mixture import MixtureSampler
from src.data.pipeline import encoded_stream, make_local_sources, make_synthetic_sources
from src.data.sources import Example

SourceFactory = Callable[[], dict[str, Iterator[Example]]]


class MixtureIterableDataset(IterableDataset):
    """Streams encoded `{input_ids, labels}` from a freshly-built mixture each epoch."""

    def __init__(
        self,
        source_factory: SourceFactory,
        weights: dict[str, float],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        seed: int = 0,
        max_examples: int | None = None,
    ) -> None:
        self.source_factory = source_factory
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed
        self.max_examples = max_examples

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        sampler = MixtureSampler(self.source_factory(), self.weights, seed=self.seed)
        n = 0
        for enc in encoded_stream(sampler, self.tokenizer, self.max_len):
            yield enc
            n += 1
            if self.max_examples is not None and n >= self.max_examples:
                break


def local_source_factory(data_dir: str | Path, modalities) -> SourceFactory:
    return lambda: make_local_sources(data_dir, modalities)


def synthetic_source_factory(modalities, n: int) -> SourceFactory:
    return lambda: make_synthetic_sources(modalities, n)


def build_dataloader(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_len: int,
) -> DataLoader:
    """DataLoader with the padding collator bound to the tokenizer's pad id."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=make_collate_fn(pad_token_id=tokenizer.pad_token_id, max_len=max_len),
    )
