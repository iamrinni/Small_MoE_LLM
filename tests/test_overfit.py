"""Step 2.7 — Phase-2 EXIT GATE: a tiny model must overfit (memorize) a single tiny batch.

This is the sanity gate before any real training: if the model + loss + autograd are wired
correctly, a tiny network can drive the LM cross-entropy to ~0 on one fixed batch.

Precision-aware: we assert the **CE component** collapses (the LM memorizes). The *total*
loss keeps a small floor from the load-balance aux term (coef * aux ~ 0.01), which is
expected and correct — so we check CE, not total.
"""

import torch

from src.model.config import SmallMoEConfig
from src.model.model import SmallMoE
from src.utils.seed import set_seed


def _tiny_smallmoe() -> SmallMoE:
    cfg = SmallMoEConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        dtype_ext="float32",   # gate runs on CPU → fp32
    )
    return SmallMoE(cfg, device="cpu")


def test_overfits_tiny_batch():
    set_seed(0)
    model = _tiny_smallmoe()
    model.model.train()

    # one fixed tiny batch the model must memorize
    ids = torch.randint(0, model.cfg.vocab_size, (2, 8))

    opt = torch.optim.AdamW(model.model.parameters(), lr=1e-2)

    initial_ce = float(model(input_ids=ids, labels=ids).loss_breakdown.ce)

    final_ce = initial_ce
    for _ in range(200):
        out = model(input_ids=ids, labels=ids, collect_routing=False)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        final_ce = float(out.loss_breakdown.ce)

    # CE must collapse from ~log(vocab) toward 0 → model memorized the batch.
    assert final_ce < 0.1, f"overfit gate failed: CE {initial_ce:.3f} -> {final_ce:.3f}"
    assert final_ce < initial_ce * 0.1
