#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import dump_json, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate eval and training artifacts into final report.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--training_curves", required=False, default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    summary = load_json(eval_dir / "summary.json", default={})
    model_metrics = load_json(eval_dir / "model_metrics.json", default={})

    curves = {}
    if args.training_curves:
        curves = load_json(args.training_curves, default={})

    winner = None
    if isinstance(summary, dict):
        winner = summary.get("winner")

    winner_metrics = {}
    if winner and isinstance(model_metrics, dict):
        wm = model_metrics.get(winner, {})
        if isinstance(wm, dict):
            winner_metrics = wm

    report = {
        "headline": {
            "winner": winner,
            "avg_exact_match_clean": winner_metrics.get("summary", {}).get("avg_exact_match_clean"),
            "avg_exact_match_noisy": winner_metrics.get("summary", {}).get("avg_exact_match_noisy"),
            "step_f1": winner_metrics.get("step_metrics", {}).get("step_f1"),
            "step_auroc": winner_metrics.get("step_metrics", {}).get("step_auroc"),
        },
        "ranking": summary.get("ranking", []) if isinstance(summary, dict) else [],
        "model_metrics": model_metrics,
        "training_curves": curves,
    }

    dump_json(report, args.output)
    print(f"[ok] wrote report: {args.output}")


if __name__ == "__main__":
    main()
