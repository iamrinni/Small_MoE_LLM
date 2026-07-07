"""Step 2.5 — positional-encoding switch.

The spec asks to "experiment and report" RoPE vs learnable. RoPE is the locked default and
is **native** to `OlmoeForCausalLM` (no work). Learnable absolute positions are an ablation
deferred to Phase 6; the flag is recognized here so configs are honest, but selecting it now
raises a clear error instead of silently falling back to RoPE.
"""

from __future__ import annotations

from src.model.config import SmallMoEConfig

ROPE = "rope"
LEARNABLE = "learnable"


def configure_positional_encoding(cfg: SmallMoEConfig) -> str:
    """Validate/apply the positional-encoding policy. Returns the active mode.

    * ``rope``      → native OLMoE RoPE; ensures ``rope_theta`` is set. No-op otherwise.
    * ``learnable`` → Phase-6 ablation, not implemented yet → ``NotImplementedError``.
    """
    mode = cfg.pos_encoding_ext
    if mode == ROPE:
        if getattr(cfg, "rope_theta", None) is None:
            cfg.rope_theta = 10000.0
        return ROPE
    if mode == LEARNABLE:
        # Implemented in Phase 6 (src/model/ablations.apply_learnable_pos_encoding), applied
        # by build_model after construction. Nothing to configure on the HF config here.
        return LEARNABLE
    raise ValueError(f"unknown pos_encoding_ext: {mode!r}")
