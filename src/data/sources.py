"""Step 3.2 — per-modality dataset adapters.

Each adapter yields a uniform `Example(modality, prompt, completion)`:
  * plain text (language, code): ``prompt=""``  → the whole text is the completion (loss on all)
  * QA / reasoning (logic, math): ``prompt=question`` (masked) + ``completion=answer`` (trained)

Real corpora are loaded via HF `datasets` (streaming, with a per-modality cap). A
network-free `synthetic_modality` mirror produces well-formed fake examples so the test
suite and the smoke run never need to download anything.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

from src.data.tokenizer import MODALITIES


@dataclass
class Example:
    modality: str
    prompt: str       # masked in the loss (may be empty)
    completion: str   # trained on

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(f"unknown modality {self.modality!r}; expected {MODALITIES}")
        if not self.completion:
            raise ValueError("completion must be non-empty (it carries the training signal)")


@dataclass
class DatasetSpec:
    """How to stream + map one HF dataset into `Example`s for a modality."""
    hf_id: str
    split: str
    mapper: Callable[[dict], Example]
    subset: str | None = None


# ---- mappers: raw HF record -> Example -------------------------------------------------

def _map_language(rec: dict) -> Example:
    return Example("language", "", (rec.get("text") or "").strip())


def _map_code(rec: dict) -> Example:
    return Example("code", "", (rec.get("content") or rec.get("text") or "").strip())


def _map_math(rec: dict) -> Example:
    # GSM8K: {"question", "answer"} where answer holds the step-by-step reasoning + "#### N"
    q = (rec.get("question") or "").strip()
    a = (rec.get("answer") or "").strip()
    return Example("math", prompt=f"Question: {q}\nAnswer:", completion=" " + a)


def _map_logic(rec: dict) -> Example:
    # LogiQA: {"context", "query", "options": [...], "correct_option": int}
    ctx = (rec.get("context") or "").strip()
    query = (rec.get("query") or rec.get("question") or "").strip()
    options = rec.get("options") or []
    letters = ["A", "B", "C", "D", "E"]
    opt_lines = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
    correct = rec.get("correct_option", rec.get("label", 0))
    answer = letters[int(correct)] if options else "A"
    prompt = f"{ctx}\nQuestion: {query}\n{opt_lines}\nAnswer:"
    return Example("logic", prompt=prompt, completion=" " + answer)


# ---- default real-corpus specs (overridable) -------------------------------------------
# Closest practical choices to the spec's lists; ids can be swapped via prepare_data config.
DEFAULT_SPECS: dict[str, DatasetSpec] = {
    "language": DatasetSpec("allenai/c4", split="train", subset="en", mapper=_map_language),
    "code": DatasetSpec("codeparrot/codeparrot-clean-valid", split="train", mapper=_map_code),
    "math": DatasetSpec("openai/gsm8k", split="train", subset="main", mapper=_map_math),
    "logic": DatasetSpec("lucasmccabe/logiqa", split="train", mapper=_map_logic),
}


def load_modality(
    modality: str,
    cap: int,
    specs: dict[str, DatasetSpec] | None = None,
    streaming: bool = True,
) -> Iterator[Example]:
    """Stream up to `cap` examples for one modality from its real HF dataset."""
    from datasets import load_dataset

    spec = (specs or DEFAULT_SPECS)[modality]
    ds = load_dataset(spec.hf_id, spec.subset, split=spec.split, streaming=streaming)
    n = 0
    for rec in ds:
        try:
            ex = spec.mapper(rec)
        except (ValueError, KeyError, IndexError):
            continue  # skip malformed/empty records
        yield ex
        n += 1
        if n >= cap:
            break


def synthetic_modality(modality: str, n: int) -> Iterator[Example]:
    """Network-free fake examples for tests / smoke runs (well-formed, modality-shaped)."""
    for i in range(n):
        if modality == "language":
            yield Example("language", "", f"The quick brown fox number {i} jumps over the lazy dog.")
        elif modality == "code":
            yield Example("code", "", f"def f{i}(x):\n    return x * {i} + 1\n")
        elif modality == "math":
            yield Example("math", prompt=f"Question: What is {i} + {i}?\nAnswer:", completion=f" {2*i}")
        elif modality == "logic":
            yield Example(
                "logic",
                prompt=f"All A are B. Item {i} is A.\nQuestion: Is item {i} B?\nA. Yes\nB. No\nAnswer:",
                completion=" A",
            )
        else:
            raise ValueError(f"unknown modality {modality!r}")


def take(iterable: Iterator[Example], n: int) -> list[Example]:
    """Convenience: materialize the first n examples."""
    return list(itertools.islice(iterable, n))


# ---- local JSONL cache (download once, train/test offline from disk) -------------------

def save_examples(examples: Iterator[Example], path: str | Path) -> int:
    """Write examples to a JSONL file; returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_local_examples(path: str | Path) -> Iterator[Example]:
    """Read examples back from a JSONL file produced by `save_examples`."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Example(**json.loads(line))
