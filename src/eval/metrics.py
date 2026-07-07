"""Step 5.2 — task evaluation metrics (per the spec's metrics table).

  * Math (GSM8K):   exact-match on the final numeric answer + step-count heuristic
  * Logic (LogiQA): multiple-choice accuracy
  * Programming:    Pass@k (unbiased estimator) + CodeBLEU
  * University exams: score % (multiple-choice accuracy, reused)

Metric *functions* are pure and unit-tested here; wiring to model generations lives in
`scripts/evaluate.py`.
"""

from __future__ import annotations

import math
import re


# ---- Math (GSM8K) ----------------------------------------------------------------------

def extract_final_number(text: str) -> str | None:
    """Extract the final answer. Prefers the GSM8K `#### N` marker, else the last number."""
    if text is None:
        return None
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def exact_match(prediction: str, gold: str) -> bool:
    """Exact match on extracted final numeric answers (robust to formatting)."""
    p, g = extract_final_number(prediction), extract_final_number(gold)
    if p is None or g is None:
        return False
    try:
        return math.isclose(float(p), float(g), rel_tol=1e-6, abs_tol=1e-9)
    except ValueError:
        return p == g


def step_count(text: str) -> int:
    """Reasoning-step heuristic: number of non-empty lines before the final answer."""
    if not text:
        return 0
    body = text.split("####")[0]
    return sum(1 for line in body.splitlines() if line.strip())


# ---- Multiple choice (Logic, exams) ----------------------------------------------------

def extract_choice(text: str, n_options: int = 4) -> str | None:
    """Pull the first option letter (A–E within range) from a generation."""
    if not text:
        return None
    letters = "ABCDE"[:n_options]
    m = re.search(rf"\b([{letters}])\b", text.strip())
    return m.group(1) if m else None


def multiple_choice_accuracy(predictions: list[str], golds: list[str], n_options: int = 4) -> float:
    """Fraction correct; predictions/golds are answer letters or free text containing them."""
    if not predictions:
        return 0.0
    correct = 0
    for pred, gold in zip(predictions, golds):
        p = extract_choice(pred, n_options) or pred.strip()
        g = extract_choice(gold, n_options) or gold.strip()
        correct += int(p == g)
    return correct / len(predictions)


# ---- Programming: Pass@k + CodeBLEU -----------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimator (Codex paper): 1 - C(n-c, k)/C(n, k).

    n = samples generated per problem, c = number that passed, k = the k in pass@k.
    """
    if n < k:
        raise ValueError(f"need n >= k, got n={n}, k={k}")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate_pass_at_k(results: list[tuple[int, int]], k: int) -> float:
    """Mean Pass@k over problems; each result is (n_samples, n_correct)."""
    if not results:
        return 0.0
    return sum(pass_at_k(n, c, k) for n, c in results) / len(results)


_PY_KEYWORDS = {
    "def", "return", "if", "else", "elif", "for", "while", "import", "from", "class",
    "try", "except", "finally", "with", "as", "lambda", "yield", "in", "not", "and",
    "or", "is", "None", "True", "False", "self", "raise", "assert", "pass", "break",
    "continue", "global", "nonlocal", "del",
}


def _tokenize_code(s: str) -> list[str]:
    """Split code into identifier/keyword/operator tokens (parser-free)."""
    return re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", s or "")


def _ngram_precision(pred_tokens, ref_tokens, n, weight_keywords=False):
    from collections import Counter

    def ngrams(toks):
        return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]

    pred_ng, ref_ng = ngrams(pred_tokens), ngrams(ref_tokens)
    if not pred_ng:
        return 0.0
    ref_counts = Counter(ref_ng)
    w = (lambda g: 1.0 + sum(1.0 for t in g if t in _PY_KEYWORDS)) if weight_keywords else (lambda g: 1.0)
    matched = total = 0.0
    seen = Counter()
    for g in pred_ng:
        total += w(g)
        if seen[g] < ref_counts.get(g, 0):
            matched += w(g)
            seen[g] += 1
    return matched / total if total else 0.0


def _fallback_codebleu(predictions, references) -> dict:
    """Parser-free CodeBLEU approximation: n-gram + keyword-weighted n-gram match.

    Covers 2 of CodeBLEU's 4 components; the AST syntax_match and dataflow_match need
    tree-sitter (used when the official `codebleu` package works). Labeled `codebleu_approx`
    so it's never confused with the official score.
    """
    ngram_scores, weighted_scores = [], []
    for pred, ref in zip(predictions, references):
        pt, rt = _tokenize_code(pred), _tokenize_code(ref)
        ns = [_ngram_precision(pt, rt, n) for n in (1, 2, 3, 4)]
        ws = [_ngram_precision(pt, rt, n, weight_keywords=True) for n in (1, 2, 3, 4)]
        ngram_scores.append(sum(ns) / len(ns))
        weighted_scores.append(sum(ws) / len(ws))
    ngram = sum(ngram_scores) / len(ngram_scores) if ngram_scores else 0.0
    weighted = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
    return {"codebleu_approx": 0.5 * ngram + 0.5 * weighted,
            "ngram_match": ngram, "weighted_ngram_match": weighted, "official": False}


def codebleu_score(predictions: list[str], references: list[str], lang: str = "python") -> dict:
    """CodeBLEU for the programming task.

    Prefers the official `codebleu` package (full 4-component score incl. AST/dataflow via
    tree-sitter). If it's unavailable or its parser is incompatible (e.g. codebleu 0.7.0 +
    tree-sitter on Python 3.12), falls back to a parser-free approximation so a score is
    always produced. The `official` flag in the result records which path was used.
    """
    try:
        from codebleu import calc_codebleu

        refs = [[r] for r in references]
        out = calc_codebleu(refs, predictions, lang=lang)
        out["official"] = True
        return out
    except Exception:
        return _fallback_codebleu(predictions, references)
