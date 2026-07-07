"""Step 3.3 — formatting + tokenization with a prompt loss-mask.

Turns an `Example` into `{input_ids, labels}`:

    input_ids = [tag] + prompt_ids + completion_ids + [eos]
    labels    = [-100] + [-100 …]  + completion_ids + [eos]

So the model is trained to predict the **completion** (and EOS) given the modality tag +
prompt, with the tag and prompt **masked** (-100) out of the loss. For plain text
(prompt=""), the whole text is the completion → loss on everything after the tag.

Over-length sequences are left-truncated *while preserving the leading tag and the trailing
completion* (the answer is what matters for QA).
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from src.data.sources import Example
from src.data.tokenizer import modality_token_id

IGNORE_INDEX = -100


def encode_example(
    tok: PreTrainedTokenizerBase,
    ex: Example,
    max_len: int,
) -> dict[str, list[int]]:
    """Encode one `Example` into masked `input_ids` / `labels`."""
    tag_id = modality_token_id(tok, ex.modality)
    eos_id = tok.eos_token_id

    prompt_ids = tok.encode(ex.prompt, add_special_tokens=False) if ex.prompt else []
    completion_ids = tok.encode(ex.completion, add_special_tokens=False)

    input_ids = [tag_id] + prompt_ids + completion_ids + [eos_id]
    labels = [IGNORE_INDEX] + [IGNORE_INDEX] * len(prompt_ids) + completion_ids + [eos_id]

    if len(input_ids) > max_len:
        # keep the tag (position 0) + the last (max_len-1) tokens (preserves the completion)
        keep = max_len - 1
        input_ids = [input_ids[0]] + input_ids[-keep:]
        labels = [labels[0]] + labels[-keep:]

    return {"input_ids": input_ids, "labels": labels}


def has_trainable_tokens(encoded: dict[str, list[int]]) -> bool:
    """True if at least one label is not masked (otherwise the example contributes no loss)."""
    return any(l != IGNORE_INDEX for l in encoded["labels"])
