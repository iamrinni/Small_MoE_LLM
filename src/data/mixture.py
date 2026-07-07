"""Step 3.4 — multi-task mixture sampler.

The spec asks for "mixture-of-tasks routing logic". This interleaves per-modality streams
into one training stream, drawing each example's modality from configurable weights
(seeded → reproducible). It tracks the **realized** mixture so we can log and report that
the actual proportions match the intended ones.

When a modality's source is exhausted (finite caps), it's dropped and the remaining weights
are renormalized; iteration ends when all sources are exhausted.
"""

from __future__ import annotations

import random
from typing import Iterator

from src.data.sources import Example
from src.data.tokenizer import MODALITIES


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Validate and L1-normalize mixture weights."""
    if not weights:
        raise ValueError("mixture weights are empty")
    for m in weights:
        if m not in MODALITIES:
            raise ValueError(f"unknown modality in mixture: {m!r}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("mixture weights must be non-negative")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("mixture weights sum to 0")
    return {m: w / total for m, w in weights.items()}


class MixtureSampler:
    """Weighted interleaving of per-modality `Example` iterators."""

    def __init__(
        self,
        sources: dict[str, Iterator[Example]],
        weights: dict[str, float],
        seed: int = 0,
    ) -> None:
        self.weights = normalize_weights(weights)
        if set(self.weights) - set(sources):
            raise ValueError(f"missing sources for modalities: {set(self.weights) - set(sources)}")
        self._sources = {m: iter(sources[m]) for m in self.weights}
        self._rng = random.Random(seed)
        self._counts: dict[str, int] = {m: 0 for m in self.weights}

    def __iter__(self) -> Iterator[Example]:
        active = dict(self.weights)
        while active:
            mods = list(active)
            w = [active[m] for m in mods]
            modality = self._rng.choices(mods, weights=w, k=1)[0]
            try:
                ex = next(self._sources[modality])
            except StopIteration:
                del active[modality]            # exhausted → drop & renormalize implicitly
                continue
            self._counts[modality] += 1
            yield ex

    def realized_mixture(self) -> dict[str, float]:
        """Actual proportion of yielded examples per modality (so far)."""
        total = sum(self._counts.values())
        if total == 0:
            return {m: 0.0 for m in self.weights}
        return {m: c / total for m, c in self._counts.items()}

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)
