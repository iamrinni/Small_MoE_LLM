"""Step 5.2 tests — evaluation metrics: exact-match, MC accuracy, pass@k, codebleu."""

import math

import pytest

from src.eval.metrics import (
    aggregate_pass_at_k,
    codebleu_score,
    exact_match,
    extract_choice,
    extract_final_number,
    multiple_choice_accuracy,
    pass_at_k,
    step_count,
)


# --- math ---
def test_extract_gsm8k_marker():
    assert extract_final_number("blah\n#### 72") == "72"
    assert extract_final_number("answer is 1,234 apples") == "1234"
    assert extract_final_number("no numbers here") is None


def test_exact_match_numeric():
    assert exact_match("... #### 72", "The answer #### 72")
    assert exact_match("result: 42", "#### 42")
    assert not exact_match("#### 71", "#### 72")


def test_step_count():
    assert step_count("step one\nstep two\n#### 5") == 2


# --- multiple choice ---
def test_extract_choice():
    assert extract_choice("The answer is B.") == "B"
    assert extract_choice("D") == "D"
    assert extract_choice("nothing") is None


def test_mc_accuracy():
    preds = ["A", "The answer is B", "C", "D"]
    golds = ["A", "B", "C", "A"]
    assert math.isclose(multiple_choice_accuracy(preds, golds), 0.75)


# --- pass@k ---
def test_pass_at_k_edges():
    assert pass_at_k(n=5, c=0, k=1) == 0.0          # none correct
    assert pass_at_k(n=5, c=5, k=1) == 1.0          # all correct
    assert pass_at_k(n=5, c=5, k=3) == 1.0


def test_pass_at_k_partial():
    # 1 of 5 correct, pass@1 = 1/5
    assert math.isclose(pass_at_k(n=5, c=1, k=1), 0.2)
    # 2 of 5 correct, pass@1 = 2/5
    assert math.isclose(pass_at_k(n=5, c=2, k=1), 0.4)


def test_pass_at_k_requires_n_ge_k():
    with pytest.raises(ValueError):
        pass_at_k(n=2, c=1, k=5)


def test_aggregate_pass_at_k():
    # problem 1: 1/4 correct, problem 2: 4/4 correct → mean pass@1 = (0.25 + 1)/2
    assert math.isclose(aggregate_pass_at_k([(4, 1), (4, 4)], k=1), 0.625)


# --- codebleu ---
def test_codebleu_identical_is_high():
    code = ["def add(a, b):\n    return a + b"]
    score = codebleu_score(code, code, lang="python")
    assert score  # always returns a dict (official or approx)
    key = "codebleu" if score.get("official") else "codebleu_approx"
    assert score[key] > 0.99                          # identical code → ~1.0


def test_codebleu_different_is_lower():
    pred = ["def add(a, b):\n    return a + b"]
    ref = ["class Foo:\n    x = 1"]
    score = codebleu_score(pred, ref, lang="python")
    key = "codebleu" if score.get("official") else "codebleu_approx"
    assert score[key] < 0.5                            # unrelated code → low
