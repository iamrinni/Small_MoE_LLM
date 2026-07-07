"""Step 4.3 tests — MetricLogger writes JSONL (and TensorBoard files) without crashing."""

import json
from pathlib import Path

from src.training.logger import MetricLogger


def test_jsonl_written(tmp_path: Path):
    with MetricLogger(tmp_path, backend="none") as lg:
        lg.log({"loss/total": 2.5, "lr": 1e-3}, step=0)
        lg.log({"loss/total": 2.1, "lr": 9e-4}, step=1)

    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["step"] == 0 and rec["loss/total"] == 2.5


def test_custom_json_path(tmp_path: Path):
    jp = tmp_path / "logs" / "run.jsonl"
    with MetricLogger(tmp_path, backend="none", json_path=jp) as lg:
        lg.log({"a": 1.0}, step=0)
    assert jp.exists()


def test_tensorboard_backend_creates_files(tmp_path: Path):
    with MetricLogger(tmp_path, backend="tensorboard") as lg:
        lg.log({"loss/total": 1.0}, step=0)
    assert (tmp_path / "tb").exists()
    assert any((tmp_path / "tb").iterdir())        # event file written
