"""Step 4.4 tests — Trainer: loss decreases over steps, logs metrics, checkpoints & resumes."""

from pathlib import Path

import pytest

from src.data.tokenizer import MODALITIES, build_tokenizer
from src.model.config import SmallMoEConfig
from src.model.model import SmallMoE
from src.training.data import (
    MixtureIterableDataset,
    build_dataloader,
    synthetic_source_factory,
)
from src.training.logger import MetricLogger
from src.training.trainer import Trainer, TrainConfig

WEIGHTS = {"language": 0.4, "code": 0.25, "math": 0.2, "logic": 0.15}


@pytest.fixture(scope="module")
def tok():
    try:
        return build_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


def _small_model(tok):
    cfg = SmallMoEConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, num_experts=8,
                         num_experts_per_tok=2, vocab_size=len(tok),
                         max_position_embeddings=64, dtype_ext="float32")
    return SmallMoE(cfg, device="cpu")


def _dataloader(tok, max_examples=400):
    ds = MixtureIterableDataset(synthetic_source_factory(MODALITIES, 5000), WEIGHTS,
                                tok, max_len=48, seed=0, max_examples=max_examples)
    return build_dataloader(ds, tok, batch_size=4, max_len=48)


def test_training_reduces_loss(tok, tmp_path: Path):
    cfg = TrainConfig(max_steps=40, warmup_steps=5, lr=3e-3, grad_accum_steps=1,
                      log_every=5, eval_every=10_000, save_every=10_000, precision="float32")
    with MetricLogger(tmp_path, backend="none") as logger:
        trainer = Trainer(cfg, _small_model(tok), tok, _dataloader(tok), tmp_path,
                          logger=logger, max_len=48)
        history = trainer.train()

    losses = [h["loss/total"] for h in history if "loss/total" in h]
    assert len(losses) >= 3
    assert losses[-1] < losses[0], f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_metrics_logged(tok, tmp_path: Path):
    cfg = TrainConfig(max_steps=10, warmup_steps=2, lr=3e-3, log_every=5,
                      eval_every=10_000, save_every=10_000, precision="float32")
    trainer = Trainer(cfg, _small_model(tok), tok, _dataloader(tok, 200), tmp_path, max_len=48)
    history = trainer.train()
    logged = history[0]
    assert "lr" in logged and "throughput/tok_per_s" in logged
    assert "routing/gate_entropy" in logged            # routing metrics present
    assert "train/perplexity" in logged


def test_checkpoint_saved_and_resumable(tok, tmp_path: Path):
    cfg = TrainConfig(max_steps=10, warmup_steps=2, lr=3e-3, log_every=5,
                      eval_every=10_000, save_every=10_000, precision="float32")
    trainer = Trainer(cfg, _small_model(tok), tok, _dataloader(tok, 200), tmp_path, max_len=48)
    trainer.train()
    final = tmp_path / "final"
    assert final.exists() and (final / "trainer_state.json").exists()

    # a fresh trainer can load the state and recover the step count
    trainer2 = Trainer(cfg, _small_model(tok), tok, _dataloader(tok, 200), tmp_path, max_len=48)
    trainer2.load_checkpoint(final)
    assert trainer2.step == 10


def test_save_total_limit_prunes_old_checkpoints(tok, tmp_path: Path):
    """Only the N most recent step_* checkpoints are kept on disk."""
    cfg = TrainConfig(max_steps=10, warmup_steps=2, lr=3e-3, log_every=100,
                      eval_every=10_000, save_every=2, save_total_limit=2, precision="float32")
    trainer = Trainer(cfg, _small_model(tok), tok, _dataloader(tok, 400), tmp_path, max_len=48)
    trainer.train()
    step_ckpts = sorted(p.name for p in tmp_path.glob("step_*") if p.is_dir())
    assert len(step_ckpts) == 2, f"expected 2 kept, got {step_ckpts}"
    assert (tmp_path / "final").exists()          # final is always kept


def test_per_modality_validation(tok, tmp_path: Path):
    cfg = TrainConfig(max_steps=12, warmup_steps=2, lr=3e-3, log_every=100,
                      eval_every=6, save_every=10_000, precision="float32")
    val_sources = {m: iter(_synth(m)) for m in MODALITIES}
    trainer = Trainer(cfg, _small_model(tok), tok, _dataloader(tok, 200), tmp_path,
                      val_sources=val_sources, max_len=48)
    val = trainer.validate()
    assert any(k.endswith("/perplexity") for k in val)
    assert "val/mean_ce" in val


def _synth(m):
    from src.data.sources import synthetic_modality
    return synthetic_modality(m, 50)
