#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import now_ts
from train.reliastep_train_lib import run_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ReLiaStep-GRPO reference implementation.")
    p.add_argument("--model", required=True)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--train_prefs", required=True)
    p.add_argument("--noise_features", required=True)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--rollouts_per_prompt", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--grpo_updates", type=int, default=300)
    p.add_argument("--em_every", type=int, default=25)
    p.add_argument("--em_iters", type=int, default=5)
    p.add_argument("--lora_sync_every", type=int, default=20)
    p.add_argument("--uncertainty_lambda", type=float, default=0.35)
    p.add_argument("--outcome_anchor_weight", type=float, default=0.2)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[warn] ignored unknown args: {unknown}")

    ckpt = run_training(
        method="reliastep",
        model=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        updates=args.grpo_updates,
        prefs_path=args.train_prefs,
        features_path=args.noise_features,
        use_reliability=True,
        uncertainty_lambda=args.uncertainty_lambda,
        outcome_anchor_weight=args.outcome_anchor_weight,
        extra_args={
            "qlora": args.qlora,
            "lora_r": args.lora_r,
            "global_batch_size": args.global_batch_size,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "max_new_tokens": args.max_new_tokens,
            "em_every": args.em_every,
            "em_iters": args.em_iters,
            "lora_sync_every": args.lora_sync_every,
            "timestamp": now_ts(),
        },
    )
    print(f"[ok] trained reliastep checkpoint at {args.output_dir}")
    print(f"[metric] quality_score={ckpt.get('quality_score', 0):.4f}")


if __name__ == "__main__":
    main()
