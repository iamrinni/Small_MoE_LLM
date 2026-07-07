"""Plot training metrics from a JSONL log → PNG figures (Phase-7 visualization).

    python scripts/plot_metrics.py --log logs/local_small_metrics.jsonl --out report/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if key in r and "step" in r:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "report" / "figures"))
    args = ap.parse_args()

    rows = load_rows(Path(args.log))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1) loss components
    ax = axes[0, 0]
    for key, label in [("loss/total", "total"), ("loss/ce", "CE"), ("loss/aux", "aux (routing)")]:
        xs, ys = series(rows, key)
        if xs:
            ax.plot(xs, ys, label=label, marker="o", ms=3)
    ax.set_title("Loss"); ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)

    # 2) perplexity (log scale)
    ax = axes[0, 1]
    xs, ys = series(rows, "train/perplexity")
    if xs:
        ax.plot(xs, ys, color="tab:red", marker="o", ms=3)
        ax.set_yscale("log")
    ax.set_title("Train perplexity"); ax.set_xlabel("step"); ax.set_ylabel("ppl (log)"); ax.grid(alpha=0.3)

    # 3) learning rate
    ax = axes[1, 0]
    xs, ys = series(rows, "lr")
    if xs:
        ax.plot(xs, ys, color="tab:green", marker="o", ms=3)
    ax.set_title("Learning rate (warmup + cosine)"); ax.set_xlabel("step"); ax.set_ylabel("lr"); ax.grid(alpha=0.3)

    # 4) routing health: gate & load entropy + balance
    ax = axes[1, 1]
    for key, label in [("routing/gate_entropy", "gate entropy"), ("routing/load_entropy", "load entropy")]:
        xs, ys = series(rows, key)
        if xs:
            ax.plot(xs, ys, label=label, marker="o", ms=3)
    ax.set_title("Routing entropy (max = log 8 = 2.08)"); ax.set_xlabel("step"); ax.set_ylabel("nats")
    ax.axhline(2.079, ls="--", c="gray", alpha=0.6, label="max (balanced)")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    p1 = out / "training_curves.png"
    fig.savefig(p1, dpi=120)
    print(f"saved {p1}")

    # 5) final per-expert load bar chart (last row with per-expert loads)
    expert_rows = [r for r in rows if any(k.startswith("routing/load/expert_") for k in r)]
    if expert_rows:
        last = expert_rows[-1]
        experts = sorted((k for k in last if k.startswith("routing/load/expert_")),
                         key=lambda k: int(k.rsplit("_", 1)[1]))
        loads = [last[k] for k in experts]
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(range(len(loads)), loads, color="tab:purple")
        ax2.axhline(1 / len(loads), ls="--", c="gray", label="uniform")
        ax2.set_title(f"Per-expert load @ step {last['step']}")
        ax2.set_xlabel("expert"); ax2.set_ylabel("fraction of tokens"); ax2.legend()
        fig2.tight_layout()
        p2 = out / "expert_load.png"
        fig2.savefig(p2, dpi=120)
        print(f"saved {p2}")


if __name__ == "__main__":
    sys.exit(main())
