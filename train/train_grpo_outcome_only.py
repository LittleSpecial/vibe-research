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
    p = argparse.ArgumentParser(description="Train outcome-only GRPO baseline.")
    p.add_argument("--model", required=True)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--train_data", required=True)
    p.add_argument("--reuse_rollouts", required=True)
    p.add_argument("--global_batch_size", type=int, default=32)
    p.add_argument("--rollouts_per_prompt", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--grpo_updates", type=int, default=160)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[warn] ignored unknown args: {unknown}")

    ckpt = run_training(
        method="outcome_only",
        model=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        updates=args.grpo_updates,
        rollouts_path=args.reuse_rollouts,
        use_reliability=False,
        uncertainty_lambda=0.0,
        outcome_anchor_weight=0.0,
        extra_args={
            "qlora": args.qlora,
            "lora_r": args.lora_r,
            "global_batch_size": args.global_batch_size,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "max_new_tokens": args.max_new_tokens,
            "train_data": args.train_data,
            "timestamp": now_ts(),
        },
    )
    print(f"[ok] trained outcome baseline at {args.output_dir}")
    print(f"[metric] quality_score={ckpt.get('quality_score', 0):.4f}")


if __name__ == "__main__":
    main()
