"""Step 3.5 tests — collation: padding, attention mask, label masking, packing."""

import pytest
import torch

from src.data.collate import make_collate_fn, pack_sequences, pad_collate
from src.data.format import IGNORE_INDEX

PAD = 0


def test_pad_collate_shapes_and_mask():
    batch = [
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
        {"input_ids": [8, 9], "labels": [-100, 9]},
    ]
    out = pad_collate(batch, pad_token_id=PAD)
    assert out["input_ids"].shape == (2, 3)
    assert out["labels"].shape == (2, 3)
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_pad_positions_masked_in_labels():
    batch = [{"input_ids": [5, 6, 7], "labels": [5, 6, 7]}, {"input_ids": [8], "labels": [8]}]
    out = pad_collate(batch, pad_token_id=PAD)
    # second row: position 0 real, positions 1-2 are padding → label -100, input pad id
    assert out["labels"][1].tolist() == [8, IGNORE_INDEX, IGNORE_INDEX]
    assert out["input_ids"][1].tolist() == [8, PAD, PAD]


def test_pad_collate_dtypes_are_long():
    out = pad_collate([{"input_ids": [1, 2], "labels": [1, 2]}], pad_token_id=PAD)
    assert out["input_ids"].dtype == torch.long and out["labels"].dtype == torch.long


def test_pad_collate_respects_max_len():
    batch = [{"input_ids": list(range(10)), "labels": list(range(10))}]
    out = pad_collate(batch, pad_token_id=PAD, max_len=4)
    assert out["input_ids"].shape == (1, 4)


def test_empty_batch_errors():
    with pytest.raises(ValueError):
        pad_collate([], pad_token_id=PAD)


def test_pack_sequences_fixed_blocks():
    encoded = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]} for _ in range(4)]  # 12 tokens
    blocks = pack_sequences(encoded, max_len=5, drop_last=True)
    assert all(len(b["input_ids"]) == 5 for b in blocks)
    assert len(blocks) == 2                      # 12 // 5 = 2 full blocks, remainder dropped


def test_make_collate_fn_usable_as_dataloader_collate():
    fn = make_collate_fn(pad_token_id=PAD)
    out = fn([{"input_ids": [1, 2], "labels": [1, 2]}, {"input_ids": [3], "labels": [3]}])
    assert out["input_ids"].shape == (2, 2)
