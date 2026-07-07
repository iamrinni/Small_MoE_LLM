"""Step 5.4 — per-task evaluation runners.

Each runner takes the model + tokenizer + a list of `Example`s and returns a metrics dict,
using generation (`generate.py`) + metrics (`metrics.py`). Kept separate from the CLI so the
logic is unit-testable on the tiny model.
"""

from __future__ import annotations

from src.data.sources import Example
from src.eval.generate import generate_completions, generate_one
from src.eval.metrics import (
    aggregate_pass_at_k,
    codebleu_score,
    exact_match,
    multiple_choice_accuracy,
)


def eval_math(model, tokenizer, examples: list[Example], max_new_tokens: int = 96) -> dict:
    """GSM8K-style exact-match on the final numeric answer."""
    correct = 0
    for ex in examples:
        gen = generate_one(model, tokenizer, ex.prompt, "math", max_new_tokens=max_new_tokens)
        correct += int(exact_match(gen, ex.completion))
    n = len(examples)
    return {"task": "math", "n": n, "exact_match": correct / n if n else 0.0}


def eval_logic(model, tokenizer, examples: list[Example], n_options: int = 4) -> dict:
    """LogiQA multiple-choice accuracy."""
    preds, golds = [], []
    for ex in examples:
        preds.append(generate_one(model, tokenizer, ex.prompt, "logic", max_new_tokens=4))
        golds.append(ex.completion)
    acc = multiple_choice_accuracy(preds, golds, n_options=n_options)
    return {"task": "logic", "n": len(examples), "accuracy": acc}


def eval_code(model, tokenizer, examples: list[Example], prompt_tokens: int = 16,
              max_new_tokens: int = 64) -> dict:
    """CodeBLEU of generated continuations vs the reference code."""
    preds, refs = [], []
    for ex in examples:
        toks = tokenizer.encode(ex.completion, add_special_tokens=False)
        if len(toks) < prompt_tokens + 4:
            continue
        prompt = tokenizer.decode(toks[:prompt_tokens])
        reference = tokenizer.decode(toks[prompt_tokens:])
        gen = generate_one(model, tokenizer, prompt, "code", max_new_tokens=max_new_tokens)
        preds.append(gen)
        refs.append(reference)
    if not preds:
        return {"task": "code", "n": 0}
    cb = codebleu_score(preds, refs, lang="python")
    key = "codebleu" if cb.get("official") else "codebleu_approx"
    return {"task": "code", "n": len(preds), "codebleu": cb.get(key, 0.0),
            "codebleu_official": bool(cb.get("official"))}


def eval_code_pass_at_k(model, tokenizer, problems: list[dict], k: int = 1,
                        n_samples: int = 5, max_new_tokens: int = 128) -> dict:
    """Pass@k over problems that carry an executable check.

    `problems` = [{"prompt": str, "check": callable(generated_code)->bool}]. Generates
    `n_samples` completions per problem and counts how many pass. (Real problems come from a
    HumanEval/MBPP-style set; the metric itself is exercised in unit tests.)
    """
    results = []
    for p in problems:
        gens = generate_completions(model, tokenizer, p["prompt"], "code",
                                    max_new_tokens=max_new_tokens, do_sample=True,
                                    num_return_sequences=n_samples)
        c = sum(1 for g in gens if _safe_check(p["check"], g))
        results.append((n_samples, c))
    return {"task": "code_pass_at_k", "n": len(problems), f"pass@{k}": aggregate_pass_at_k(results, k)}


def _safe_check(check, code: str) -> bool:
    try:
        return bool(check(code))
    except Exception:
        return False
