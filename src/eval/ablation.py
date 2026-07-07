"""Phase 6 — ablation runner.

Trains a small model per variant on the same data/seed and collects comparable metrics:
final CE / perplexity, routing load-balance, and modality specialization. Used to fill the
ablation table for the report. Kept small + synthetic-friendly so the matrix runs locally;
the same runner scales to the real GPU config by passing a larger base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.data.sources import synthetic_modality, take
from src.data.tokenizer import MODALITIES
from src.eval.routing_analysis import analyze_routing
from src.model.config import SmallMoEConfig
from src.model.model import SmallMoE
from src.training.data import (
    MixtureIterableDataset,
    build_dataloader,
    synthetic_source_factory,
)
from src.training.trainer import Trainer, TrainConfig
from src.utils.seed import set_seed

WEIGHTS = {"language": 0.4, "code": 0.25, "math": 0.2, "logic": 0.15}

# Compact base model for fast local ablations (overridden per variant).
BASE_MODEL = dict(
    hidden_size=128, intermediate_size=256, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=4, num_experts=8,
    num_experts_per_tok=2, max_position_embeddings=64, dtype_ext="float32",
)


@dataclass
class VariantResult:
    name: str
    ce: float
    perplexity: float
    load_balance: float          # routing load entropy / max (1.0 = perfectly balanced)
    specialization: float        # modality-expert L1-from-uniform
    total_params: int
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {"variant": self.name, "ce": round(self.ce, 3),
                "perplexity": round(self.perplexity, 1),
                "load_balance": round(self.load_balance, 3),
                "specialization": round(self.specialization, 3),
                "params_M": round(self.total_params / 1e6, 2), **self.extra}


def run_variant(
    name: str,
    tokenizer,
    model_overrides: dict | None = None,
    train_overrides: dict | None = None,
    steps: int = 60,
    seed: int = 0,
    max_len: int = 48,
    output_dir: str = "checkpoints/_ablation",
) -> VariantResult:
    """Build + train one variant on synthetic data; return its comparable metrics."""
    set_seed(seed)
    model_cfg = SmallMoEConfig(vocab_size=len(tokenizer), **{**BASE_MODEL, **(model_overrides or {})})
    model = SmallMoE(model_cfg, device="cpu")

    ds = MixtureIterableDataset(synthetic_source_factory(MODALITIES, 5000), WEIGHTS,
                                tokenizer, max_len=max_len, seed=seed, max_examples=steps * 4 + 40)
    dl = build_dataloader(ds, tokenizer, batch_size=4, max_len=max_len)

    tc = TrainConfig(max_steps=steps, warmup_steps=max(5, steps // 10), lr=3e-3,
                     grad_accum_steps=1, log_every=max(5, steps // 6),
                     eval_every=10_000, save_every=10_000, precision="float32",
                     **(train_overrides or {}))
    trainer = Trainer(tc, model, tokenizer, dl, f"{output_dir}/{name}", max_len=max_len)
    history = trainer.train()

    loss_rows = [h for h in history if "loss/ce" in h]
    ce = loss_rows[-1]["loss/ce"] if loss_rows else float("nan")
    load_balance = loss_rows[-1].get("routing/load_balance", 0.0) if loss_rows else 0.0

    examples = {m: take(synthetic_modality(m, 24), 24) for m in MODALITIES}
    analysis = analyze_routing(model, tokenizer, examples, max_len=max_len, max_examples=24)

    total = sum(p.numel() for p in model.model.parameters())
    return VariantResult(
        name=name, ce=ce, perplexity=math.exp(min(ce, 20)),
        load_balance=load_balance, specialization=analysis.specialization_score(),
        total_params=total,
    )


# The default ablation matrix (each entry: display name + model/train overrides).
ABLATION_MATRIX = [
    ("baseline (top2, 8E, swiglu, rope)", {}, {}),
    ("top1_gating", {"num_experts_per_tok": 1}, {}),
    ("16_experts", {"num_experts": 16}, {}),
    ("gelu_expert", {"expert_activation_ext": "gelu"}, {}),
    ("learnable_pe", {"pos_encoding_ext": "learnable"}, {}),
    ("no_load_balance", {"router_aux_loss_coef": 0.0}, {}),
]


def run_matrix(tokenizer, steps: int = 60, matrix=None) -> list[VariantResult]:
    results = []
    for name, m_over, t_over in (matrix or ABLATION_MATRIX):
        results.append(run_variant(name, tokenizer, m_over, t_over, steps=steps))
    return results


def results_to_markdown(results: list[VariantResult]) -> str:
    header = "| variant | CE | ppl | load_balance | specialization | params(M) |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = "".join(
        f"| {r.name} | {r.ce:.3f} | {r.perplexity:.1f} | {r.load_balance:.3f} "
        f"| {r.specialization:.3f} | {r.total_params/1e6:.2f} |\n"
        for r in results
    )
    return "# Ablation results\n\n" + header + rows
