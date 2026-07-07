"""Step 4.3 — experiment logging.

A single `MetricLogger` fans metrics out to:
  * a **JSONL** file (always; the spec accepts "local JSON") — one line per log call,
  * **TensorBoard** (if `backend` in {tensorboard, both}),
  * **Weights & Biases** (if `backend` in {wandb, both} and wandb is importable).

Metrics are flat `{name: float}` dicts (loss components, lr, grad-norm, routing stats,
throughput), so the same call feeds all backends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricLogger:
    def __init__(
        self,
        output_dir: str | Path,
        backend: str = "tensorboard",
        json_path: str | Path | None = None,
        wandb_project: str | None = None,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend

        self.json_path = Path(json_path) if json_path else self.output_dir / "metrics.jsonl"
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self._json = self.json_path.open("a", encoding="utf-8")

        self._tb = None
        if backend in ("tensorboard", "both"):
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))

        self._wandb = None
        if backend in ("wandb", "both"):
            try:
                import wandb

                wandb.init(project=wandb_project or "small-moe-llm", name=run_name,
                           config=config, dir=str(self.output_dir))
                self._wandb = wandb
            except Exception as e:  # missing/unconfigured wandb shouldn't crash training
                print(f"[logger] wandb disabled: {e}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        record = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        self._json.write(json.dumps(record) + "\n")
        self._json.flush()
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, float(v), step)
        if self._wandb is not None:
            self._wandb.log(dict(metrics), step=step)

    def close(self) -> None:
        self._json.close()
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
