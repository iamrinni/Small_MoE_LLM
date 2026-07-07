"""Phase-1 scaffold sanity checks: package imports, repo layout, configs load.

These run with zero heavyweight deps (only PyYAML + NumPy) so the clean-environment
bootstrap is green before any model code exists.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    import src
    from src.utils import config, seed

    assert src.__version__
    assert callable(seed.set_seed)
    assert callable(config.load_config)


def test_required_directories_exist():
    for d in ["configs", "scripts", "notebooks", "data", "src", "report", "tests"]:
        assert (REPO_ROOT / d).is_dir(), f"missing required directory: {d}"


@pytest.mark.parametrize("name", ["model_small.yaml", "train_small.yaml", "smoke.yaml"])
def test_configs_load(name):
    from src.utils.config import load_config

    cfg = load_config(REPO_ROOT / "configs" / name)
    assert isinstance(cfg, dict) and cfg


def test_locked_decisions_in_model_config():
    """Guard the decisions locked in claude/instruction.md against accidental drift."""
    from src.utils.config import load_config

    m = load_config(REPO_ROOT / "configs" / "model_small.yaml")["model"]
    assert m["num_experts"] == 8
    assert m["num_experts_per_tok"] == 2            # top-2 gating
    assert m["expert_activation_ext"] == "swiglu"   # SwiGLU expert (HF-native, spec-compliant)
    assert m["pos_encoding_ext"] == "rope"          # RoPE default
    assert m["dtype_ext"] == "bfloat16"


def test_set_seed_runs():
    from src.utils.seed import set_seed

    set_seed(123)  # must not raise even without torch installed
