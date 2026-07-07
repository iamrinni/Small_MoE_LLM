"""Step 3.4 tests — mixture sampler: realized proportions ≈ configured weights, exhaustion."""

import pytest

from src.data.mixture import MixtureSampler, normalize_weights
from src.data.sources import synthetic_modality, take


def _big_sources():
    # large synthetic sources so they don't exhaust during the proportion test
    return {m: synthetic_modality(m, 100_000) for m in ("language", "code", "math", "logic")}


def test_normalize_weights():
    w = normalize_weights({"language": 2, "code": 2})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w["language"] - 0.5) < 1e-9


def test_normalize_rejects_bad():
    with pytest.raises(ValueError):
        normalize_weights({})
    with pytest.raises(ValueError):
        normalize_weights({"vision": 1.0})
    with pytest.raises(ValueError):
        normalize_weights({"code": -1.0})


def test_realized_mixture_matches_weights():
    weights = {"language": 0.4, "code": 0.25, "math": 0.2, "logic": 0.15}
    sampler = MixtureSampler(_big_sources(), weights, seed=0)
    take(iter(sampler), 5000)
    realized = sampler.realized_mixture()
    for m, target in normalize_weights(weights).items():
        assert abs(realized[m] - target) < 0.03, f"{m}: {realized[m]:.3f} vs {target:.3f}"


def test_seed_is_reproducible():
    w = {"language": 0.5, "code": 0.5}
    a = [e.modality for e in take(iter(MixtureSampler({m: synthetic_modality(m, 1000) for m in w}, w, seed=7)), 200)]
    b = [e.modality for e in take(iter(MixtureSampler({m: synthetic_modality(m, 1000) for m in w}, w, seed=7)), 200)]
    assert a == b


def test_exhaustion_drops_modality():
    # code has only 3 examples; language is plentiful → sampler keeps going on language
    sources = {"language": synthetic_modality("language", 1000), "code": synthetic_modality("code", 3)}
    sampler = MixtureSampler(sources, {"language": 0.5, "code": 0.5}, seed=0)
    out = take(iter(sampler), 500)
    assert sampler.counts["code"] == 3            # capped by its source
    assert sampler.counts["language"] == 497      # remainder filled from language
    assert len(out) == 500


def test_missing_source_errors():
    with pytest.raises(ValueError):
        MixtureSampler({"language": synthetic_modality("language", 10)},
                       {"language": 0.5, "code": 0.5})
