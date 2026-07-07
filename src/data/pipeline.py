"""Phase-3 glue — compose tokenizer + sources + mixture + format into a training stream.

Bridges the modular pieces into what the trainer (Phase 4) consumes:

    sources (local JSONL or synthetic)
        → MixtureSampler (weighted interleave)
        → encode_example (tag + prompt-mask)
        → encoded `{input_ids, labels}` stream

`pad_collate` then turns batches of these into model-ready tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from transformers import PreTrainedTokenizerBase

from src.data.format import encode_example, has_trainable_tokens
from src.data.mixture import MixtureSampler
from src.data.sources import Example, iter_local_examples, synthetic_modality


def make_local_sources(data_dir: str | Path, modalities) -> dict[str, Iterator[Example]]:
    """Per-modality iterators reading cached JSONL from ``data_dir/<modality>.jsonl``."""
    data_dir = Path(data_dir)
    sources = {}
    for m in modalities:
        path = data_dir / f"{m}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing prepared data for {m}: {path} (run prepare_data.py)")
        sources[m] = iter_local_examples(path)
    return sources


def make_synthetic_sources(modalities, n: int) -> dict[str, Iterator[Example]]:
    """Per-modality synthetic iterators (network-free, for tests / smoke)."""
    return {m: synthetic_modality(m, n) for m in modalities}


def encoded_stream(
    sampler: MixtureSampler,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
) -> Iterator[dict[str, list[int]]]:
    """Encode each sampled example; drop any with no trainable tokens (all-masked)."""
    for ex in sampler:
        enc = encode_example(tokenizer, ex, max_len=max_len)
        if has_trainable_tokens(enc):
            yield enc
