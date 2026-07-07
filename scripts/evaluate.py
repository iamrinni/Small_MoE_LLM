"""Step 5.4 — evaluation entry point.

Runs the applicable task metrics + MoE routing analysis on a checkpoint (or a fresh model
for smoke), then writes `results.json` and `summary.md`.

    python scripts/evaluate.py --config configs/local_small.yaml --checkpoint checkpoints/local-small/final
    python scripts/evaluate.py --config configs/smoke.yaml            # fresh (untrained) model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.sources import iter_local_examples, synthetic_modality, take  # noqa: E402
from src.data.tokenizer import MODALITIES, build_tokenizer  # noqa: E402
from src.eval.routing_analysis import analyze_routing, save_heatmap  # noqa: E402
from src.eval.tasks import eval_code, eval_logic, eval_math  # noqa: E402
from src.model.config import load_model_config  # noqa: E402
from src.model.model import SmallMoE  # noqa: E402
from src.utils.config import add_config_args, load_config, merge_overrides  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _load_examples(data_cfg: dict, n: int) -> dict:
    data_dir = REPO_ROOT / data_cfg.get("data_dir", "data/raw")
    source = data_cfg.get("source", "auto")
    have_local = all((data_dir / f"{m}.jsonl").exists() for m in MODALITIES)
    if source == "synthetic" or (source == "auto" and not have_local):
        print("[eval] using SYNTHETIC eval examples")
        return {m: take(synthetic_modality(m, n), n) for m in MODALITIES}
    print(f"[eval] using LOCAL eval examples from {data_dir}")
    return {m: take(iter_local_examples(data_dir / f"{m}.jsonl"), n) for m in MODALITIES}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate SmallMoE.")
    add_config_args(ap)
    ap.add_argument("--checkpoint", default=None, help="checkpoint dir (else fresh model)")
    ap.add_argument("--n_examples", type=int, default=50)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = merge_overrides(load_config(args.config), args.set)
    set_seed(cfg.get("run", {}).get("seed", 42))
    data_cfg = cfg.get("data", {})
    max_len = data_cfg.get("max_seq_len", 256)

    tokenizer = build_tokenizer(data_cfg.get("tokenizer", "allenai/OLMoE-1B-7B-0924"))

    if args.checkpoint:
        print(f"[eval] loading checkpoint {args.checkpoint}")
        model = SmallMoE.from_pretrained(args.checkpoint)
    else:
        print("[eval] no checkpoint → fresh (untrained) model (pipeline smoke)")
        model_cfg = load_model_config(args.config)
        model_cfg.vocab_size = max(model_cfg.vocab_size, len(tokenizer))
        model = SmallMoE(model_cfg)

    examples = _load_examples(data_cfg, args.n_examples)

    results = {}
    results["math"] = eval_math(model, tokenizer, examples["math"])
    results["logic"] = eval_logic(model, tokenizer, examples["logic"])
    results["code"] = eval_code(model, tokenizer, examples["code"])

    routing = analyze_routing(model, tokenizer, examples, max_len=max_len,
                              max_examples=args.n_examples)
    results["routing"] = routing.to_dict()

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.checkpoint) if args.checkpoint else REPO_ROOT / "checkpoints" / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    save_heatmap(routing, out_dir / "routing_heatmap.png")
    (out_dir / "summary.md").write_text(_summary_md(results))

    print("\n=== RESULTS ===")
    print(f"  math  exact_match : {results['math'].get('exact_match', 0):.3f} (n={results['math']['n']})")
    print(f"  logic accuracy    : {results['logic'].get('accuracy', 0):.3f} (n={results['logic']['n']})")
    print(f"  code  codebleu    : {results['code'].get('codebleu', 0):.3f} (n={results['code']['n']})")
    print(f"  routing spec_score: {routing.specialization_score():.3f}")
    print(f"[eval] wrote {out_dir}/results.json, summary.md, routing_heatmap.png")


def _summary_md(r: dict) -> str:
    return (
        "# Evaluation summary\n\n"
        "| Task | Metric | Value | n |\n|---|---|---|---|\n"
        f"| Math | exact_match | {r['math'].get('exact_match', 0):.3f} | {r['math']['n']} |\n"
        f"| Logic | accuracy | {r['logic'].get('accuracy', 0):.3f} | {r['logic']['n']} |\n"
        f"| Code | codebleu | {r['code'].get('codebleu', 0):.3f} | {r['code']['n']} |\n"
        f"| MoE | specialization_score | {r['routing']['specialization_score']:.3f} | - |\n"
    )


if __name__ == "__main__":
    main()
