"""Phase 6 — run the ablation matrix and write a comparison table + plot.

    python scripts/run_ablations.py --steps 80 --out report/figures

Trains each variant (baseline, top-1, 16 experts, GeLU, learnable-PE, no-load-balance) on the
same synthetic data/seed and reports CE, perplexity, routing balance, and specialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.tokenizer import build_tokenizer  # noqa: E402
from src.eval.ablation import results_to_markdown, run_matrix  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--out", default=str(REPO_ROOT / "report" / "figures"))
    args = ap.parse_args()

    tokenizer = build_tokenizer()
    print(f"[ablation] running matrix ({args.steps} steps/variant) …")
    results = run_matrix(tokenizer, steps=args.steps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_results.json").write_text(json.dumps([r.row() for r in results], indent=2))
    (out / "ablation_results.md").write_text(results_to_markdown(results))

    print("\n" + results_to_markdown(results))
    _plot(results, out / "ablation_comparison.png")
    print(f"[ablation] wrote {out}/ablation_results.(json|md), ablation_comparison.png")


def _plot(results, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.name for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (attr, title) in zip(
        axes, [("perplexity", "Perplexity (↓)"), ("load_balance", "Load balance (↑)"),
               ("specialization", "Specialization (↑)")]
    ):
        vals = [getattr(r, attr) for r in results]
        ax.barh(names, vals, color="tab:blue")
        ax.set_title(title)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
