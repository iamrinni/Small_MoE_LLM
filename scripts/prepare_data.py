"""Step 3.6 — dataset preparation.

Streams a small, capped subsample of each modality's real HF dataset and caches it as
JSONL under ``data/raw/<modality>.jsonl`` (raw data is gitignored). Downstream training/eval
read these local files, so the network is hit only once.

Usage:
    python scripts/prepare_data.py --config configs/train_small.yaml
    python scripts/prepare_data.py --config configs/train_small.yaml --set data.max_samples_per_modality=200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # allow `import src...` when run as a script

from src.data.mixture import normalize_weights  # noqa: E402
from src.data.sources import DEFAULT_SPECS, load_modality, save_examples  # noqa: E402
from src.utils.config import add_config_args, load_config, merge_overrides  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + cache modality subsamples.")
    add_config_args(parser)
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "data" / "raw"))
    args = parser.parse_args()

    cfg = merge_overrides(load_config(args.config), args.set)
    data_cfg = cfg.get("data", {})
    weights = normalize_weights(data_cfg["mixture"])
    cap = int(data_cfg.get("max_samples_per_modality", 20000))

    # keep the HF cache inside the project so everything is self-contained
    os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / "data" / "hf_cache"))

    out_dir = Path(args.out_dir)
    print(f"Preparing data → {out_dir} (cap={cap}/modality)")
    print(f"Mixture weights: {weights}")

    summary: dict[str, int] = {}
    for modality in weights:
        spec = DEFAULT_SPECS[modality]
        print(f"\n[{modality}] streaming {spec.hf_id} (subset={spec.subset}, split={spec.split}) …")
        path = out_dir / f"{modality}.jsonl"
        n = save_examples(load_modality(modality, cap=cap), path)
        summary[modality] = n
        print(f"[{modality}] saved {n} examples → {path}")

    print("\n=== Summary ===")
    for m, n in summary.items():
        print(f"  {m:10s} {n:6d} examples")
    print(f"  TOTAL      {sum(summary.values()):6d}")


if __name__ == "__main__":
    main()
