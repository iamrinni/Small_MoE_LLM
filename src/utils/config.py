"""Lightweight YAML config loading + merging.

Configs are plain dicts loaded from ``configs/*.yaml``. Keeping this dependency-free
(beyond PyYAML) makes a clean-environment run trivial to reproduce: one YAML file fully
describes a model or a training run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a single YAML config file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def merge_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key=value`` / ``a.b=value`` CLI overrides onto a (copied) config dict.

    Values are parsed as YAML scalars, so ``lr=3e-4``, ``n_experts=8`` and
    ``use_rope=true`` are typed correctly.
    """
    out = {**config}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Attach the standard ``--config`` / ``--set`` arguments shared by all entry points."""
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set training.lr=1e-4 model.n_experts=16",
    )
