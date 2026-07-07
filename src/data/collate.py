"""Step 3.5 — batching / collation.

Two ways to assemble encoded examples (`{input_ids, labels}` from `format.encode_example`)
into model-ready batches:

  * `pad_collate`  — pad each example to the batch's longest sequence; build an
    `attention_mask`; pad `labels` with -100 so padding contributes no loss. Simple and the
    default for the multi-task QA mixture (variable-length, prompt-masked).
  * `pack_sequences` — concatenate many examples into fixed `max_len` blocks (no padding
    waste); used for plain-text pretraining-style efficiency.

Tensors are built in int64 (ids/labels); dtype of the *model* is handled separately by
`resolve_dtype` — collation stays dtype-agnostic.
"""

from __future__ import annotations

import torch

from src.data.format import IGNORE_INDEX


def pad_collate(
    batch: list[dict[str, list[int]]],
    pad_token_id: int,
    max_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pad a list of `{input_ids, labels}` to a common length → batched tensors."""
    if not batch:
        raise ValueError("empty batch")
    lengths = [len(ex["input_ids"]) for ex in batch]
    width = max(lengths) if max_len is None else min(max(lengths), max_len)

    input_ids, labels, attn = [], [], []
    for ex in batch:
        ids = ex["input_ids"][:width]
        lab = ex["labels"][:width]
        pad = width - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(lab + [IGNORE_INDEX] * pad)            # padding never trains
        attn.append([1] * len(ids) + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def pack_sequences(
    encoded: list[dict[str, list[int]]],
    max_len: int,
    drop_last: bool = True,
) -> list[dict[str, list[int]]]:
    """Concatenate examples and re-chunk into fixed `max_len` blocks (no padding)."""
    flat_ids: list[int] = []
    flat_labels: list[int] = []
    for ex in encoded:
        flat_ids.extend(ex["input_ids"])
        flat_labels.extend(ex["labels"])

    blocks = []
    for i in range(0, len(flat_ids), max_len):
        ids = flat_ids[i : i + max_len]
        lab = flat_labels[i : i + max_len]
        if len(ids) < max_len:
            if drop_last:
                break
        blocks.append({"input_ids": ids, "labels": lab})
    return blocks


def make_collate_fn(pad_token_id: int, max_len: int | None = None):
    """Return a `collate_fn` bound to a pad id (for `torch.utils.data.DataLoader`)."""
    def _fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return pad_collate(batch, pad_token_id=pad_token_id, max_len=max_len)
    return _fn
