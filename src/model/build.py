"""Step 2.2 — baseline model builder.

`build_model(cfg)` instantiates a small `OlmoeForCausalLM` from a `SmallMoEConfig`.
SwiGLU experts and RoPE come natively from HF; we only enforce our policies:
strict bf16 (always, every device) and router-logits output (for routing metrics).
"""

from __future__ import annotations

import torch
from transformers import OlmoeForCausalLM

from src.model.config import SmallMoEConfig, load_model_config

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(cfg: SmallMoEConfig, device: str | None = None) -> torch.dtype:
    """Device-aware compute dtype.

    Policy: bf16 is the target for the *real* runs (GPU) and inference, per the spec
    ("bfloat16 throughout training and inference"). On CPU, bf16/fp16 have no native
    matmul and are emulated ~67x slower (measured), so local CPU dev/tests use fp32.

      * CUDA  → honor `dtype_ext` (bf16; fall back to fp16 if bf16 unsupported)
      * CPU   → fp32 always (bf16 emulation is impractical for training)
    """
    device = device or _auto_device()
    want = _DTYPE_MAP[cfg.dtype_ext]
    if str(device).startswith("cuda"):
        # Only consult GPU bf16 support when a real CUDA device is present (a CPU-only
        # box can't introspect it). Fall back to fp16 on pre-Ampere GPUs lacking bf16.
        if want is torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            return torch.float16
        return want
    return torch.float32  # CPU / MPS dev fallback


def build_model(cfg: SmallMoEConfig, device: str | None = None) -> OlmoeForCausalLM:
    """Build a small `OlmoeForCausalLM` with our policies applied.

    - `output_router_logits=True` so routing stats (load, entropy) can be computed (2.3).
    - move to `device` and cast to the device-aware dtype (bf16 on GPU, fp32 on CPU).
    """
    cfg.output_router_logits = True
    device = device or _auto_device()
    dtype = resolve_dtype(cfg, device)
    model = OlmoeForCausalLM(cfg).to(device=device, dtype=dtype)

    # apply ablation variants based on config flags (defaults are no-ops: swiglu + rope)
    if getattr(cfg, "expert_activation_ext", "swiglu") == "gelu":
        from src.model.ablations import apply_gelu_experts

        apply_gelu_experts(model)
    if getattr(cfg, "pos_encoding_ext", "rope") == "learnable":
        from src.model.ablations import apply_learnable_pos_encoding

        apply_learnable_pos_encoding(model)
    return model


def build_model_from_yaml(path) -> OlmoeForCausalLM:
    """Convenience: YAML path -> built model."""
    return build_model(load_model_config(path))


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
