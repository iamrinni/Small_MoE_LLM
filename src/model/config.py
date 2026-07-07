"""Step 2.1 — model config layer.

`SmallMoEConfig` extends HF `OlmoeConfig` with the few knobs OLMoE doesn't have, so we
reuse all of HF's architecture fields + serialization instead of reinventing a config.
The YAML files in ``configs/`` stay the human-readable source of truth (task requirement:
"configs/: JSON or YAML model configuration files").

Validation here is *structural* (must hold for any config, including ablations); the locked
default VALUES (8 experts, top-2, swiglu, rope, bf16) are guarded separately against the
default YAML in the test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig, OlmoeConfig

from src.utils.config import load_config

_POS_ENCODINGS = ("rope", "learnable")
_EXPERT_ACTS = ("swiglu", "gelu")
_DTYPES = ("bfloat16", "float16", "float32")


class SmallMoEConfig(OlmoeConfig):
    """OLMoE config + our 4 extension fields.

    Extras (everything else is inherited from ``OlmoeConfig``):
      * ``pos_encoding_ext``      — {rope, learnable}; OLMoE is RoPE-only, learnable is ours
      * ``expert_activation_ext`` — {swiglu, gelu}; triggers the GeLU ablation override
      * ``dtype_ext``             — {bfloat16, float16, float32}; strict-bf16 policy
      * ``router_z_loss_coef``    — dormant; NOT wired anywhere, kept for future use only
    """

    model_type = "small_moe"

    def __init__(
        self,
        *,
        pos_encoding_ext: str = "rope",
        expert_activation_ext: str = "swiglu",
        dtype_ext: str = "bfloat16",
        router_z_loss_coef: float = 0.0,  # dormant: not used (see plan §2.4)
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pos_encoding_ext = pos_encoding_ext
        self.expert_activation_ext = expert_activation_ext
        self.dtype_ext = dtype_ext
        self.router_z_loss_coef = router_z_loss_coef


# Register so checkpoints round-trip via AutoConfig.from_pretrained (reproducibility deliverable).
try:
    AutoConfig.register("small_moe", SmallMoEConfig)
except ValueError:
    pass  # already registered in this process (re-import)


def validate_config(cfg: SmallMoEConfig) -> SmallMoEConfig:
    """Structural sanity checks that must hold for ANY config (defaults and ablations)."""
    if cfg.num_experts < 1:
        raise ValueError(f"num_experts must be >= 1, got {cfg.num_experts}")
    if not (1 <= cfg.num_experts_per_tok <= cfg.num_experts):
        raise ValueError(
            f"num_experts_per_tok must be in [1, num_experts]; "
            f"got {cfg.num_experts_per_tok} of {cfg.num_experts}"
        )
    if cfg.pos_encoding_ext not in _POS_ENCODINGS:
        raise ValueError(f"pos_encoding_ext must be one of {_POS_ENCODINGS}, got {cfg.pos_encoding_ext!r}")
    if cfg.expert_activation_ext not in _EXPERT_ACTS:
        raise ValueError(f"expert_activation_ext must be one of {_EXPERT_ACTS}, got {cfg.expert_activation_ext!r}")
    if cfg.dtype_ext not in _DTYPES:
        raise ValueError(f"dtype_ext must be one of {_DTYPES}, got {cfg.dtype_ext!r}")
    if cfg.hidden_size % cfg.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size ({cfg.hidden_size}) must be divisible by "
            f"num_attention_heads ({cfg.num_attention_heads})"
        )
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0:
        raise ValueError(
            f"num_attention_heads ({cfg.num_attention_heads}) must be divisible by "
            f"num_key_value_heads ({cfg.num_key_value_heads})"
        )
    return cfg


def build_model_config(model_block: dict[str, Any]) -> SmallMoEConfig:
    """Build a validated `SmallMoEConfig` from a ``model:`` mapping."""
    return validate_config(SmallMoEConfig(**model_block))


def _resolve_base_path(run_path: Path, rel: str) -> Path:
    """Resolve a ``defaults.model_config`` reference (given relative to repo root)."""
    candidates = [Path(rel), run_path.parent.parent / rel, run_path.parent / Path(rel).name]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not resolve base model config {rel!r} referenced by {run_path}")


def load_model_config(path: str | Path) -> SmallMoEConfig:
    """Load a YAML and build the model config.

    Supports two YAML shapes:
      * a model file with a top-level ``model:`` block (e.g. ``model_small.yaml``);
      * a run file referencing a base via ``defaults.model_config`` and optionally
        overriding fields under ``model_overrides:`` (e.g. ``smoke.yaml``).
    """
    path = Path(path)
    raw = load_config(path)

    if "model" in raw:
        block = dict(raw["model"])
    elif "defaults" in raw and "model_config" in raw.get("defaults", {}):
        base = load_config(_resolve_base_path(path, raw["defaults"]["model_config"]))
        block = dict(base.get("model", {}))
        block.update(raw.get("model_overrides", {}))  # run-level overrides win
    else:
        raise ValueError(
            f"{path} has neither a 'model:' block nor 'defaults.model_config' reference"
        )
    return build_model_config(block)
