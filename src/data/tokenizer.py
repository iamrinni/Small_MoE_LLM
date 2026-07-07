"""Step 3.1 — tokenizer wrapper.

Reuses the OLMoE / GPT-NeoX BPE tokenizer (no training needed) and adds four **modality
tags** so every example is conditioned on its domain. Prepending the tag lets the router
(and model) see which modality a token stream belongs to — useful for the modality→expert
specialization analysis (Phase 7).

Vocab note: the base tokenizer has 50280 tokens; +4 modality tags → 50284. The model
embedding is sized 50304 (padded to a multiple of 64), so the tags fit with no resize.
`ensure_vocab_fits` asserts this invariant for any (tokenizer, config) pair.
"""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TOKENIZER = "allenai/OLMoE-1B-7B-0924"

# canonical modality → tag string
MODALITY_TOKENS: dict[str, str] = {
    "language": "<|lang|>",
    "code": "<|code|>",
    "logic": "<|logic|>",
    "math": "<|math|>",
}
MODALITIES = tuple(MODALITY_TOKENS)


def build_tokenizer(
    name: str = DEFAULT_TOKENIZER,
    add_modality_tags: bool = True,
) -> PreTrainedTokenizerBase:
    """Load the BPE tokenizer, ensure a pad token, and register modality tags."""
    tok = AutoTokenizer.from_pretrained(name)

    # OLMoE has no dedicated pad token; reuse EOS for padding (standard for causal LMs).
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if add_modality_tags:
        existing = set(tok.additional_special_tokens or [])
        new = [t for t in MODALITY_TOKENS.values() if t not in existing]
        if new:
            tok.add_special_tokens({"additional_special_tokens": list(existing) + new})
    return tok


def modality_tag(modality: str) -> str:
    """Return the tag string for a modality (raises on unknown)."""
    if modality not in MODALITY_TOKENS:
        raise KeyError(f"unknown modality {modality!r}; expected one of {MODALITIES}")
    return MODALITY_TOKENS[modality]


def modality_token_id(tok: PreTrainedTokenizerBase, modality: str) -> int:
    """Return the single token id for a modality tag."""
    ids = tok.encode(modality_tag(modality), add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"modality tag {modality_tag(modality)!r} is not a single token: {ids}")
    return ids[0]


def ensure_vocab_fits(tok: PreTrainedTokenizerBase, vocab_size: int) -> None:
    """Assert the model's embedding (vocab_size) covers every tokenizer id (incl. tags)."""
    n = len(tok)
    if n > vocab_size:
        raise ValueError(
            f"tokenizer has {n} tokens but model vocab_size={vocab_size}; "
            f"increase vocab_size (>= {n}) or resize embeddings."
        )
