"""Step 5.1 — generation utilities for evaluation.

Wraps `SmallMoE.generate` (which delegates to HF `generate`) to produce completions from a
modality-tagged prompt. Supports greedy and sampling decoding, and returns only the newly
generated text (the completion), which the metrics in `metrics.py` then score.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from src.data.tokenizer import modality_tag
from src.model.model import SmallMoE


def build_prompt(prompt: str, modality: str) -> str:
    """Prepend the modality tag exactly as training did."""
    return f"{modality_tag(modality)}{prompt}"


@torch.no_grad()
def generate_completions(
    model: SmallMoE,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    modality: str,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
) -> list[str]:
    """Generate `num_return_sequences` completions for one modality-tagged prompt."""
    model.model.eval()
    text = build_prompt(prompt, modality)
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    # output_router_logits=True conflicts with the generation cache in HF OLMoE — disable it
    # for decoding, then restore (routing metrics are collected via forward, not generate).
    prev = model.model.config.output_router_logits
    model.model.config.output_router_logits = False
    try:
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    finally:
        model.model.config.output_router_logits = prev
    # keep only the newly generated tokens (strip the prompt prefix)
    new_tokens = out[:, input_ids.shape[1]:]
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in new_tokens]


def generate_one(model, tokenizer, prompt, modality, **kw) -> str:
    """Convenience: a single greedy completion string."""
    return generate_completions(model, tokenizer, prompt, modality, num_return_sequences=1, **kw)[0]
