#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import dump_json, load_jsonl, mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact training curves from train logs.")
    parser.add_argument("--run_dir", required=True, help="checkpoint directory with train_log.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    logs = load_jsonl(run_dir / "train_log.jsonl")
    if not logs:
        raise SystemExit(f"train_log.jsonl not found or empty in {run_dir}")

    updates = [int(x.get("update", i + 1)) for i, x in enumerate(logs)]
    rewards = [float(x.get("train_reward", 0.0)) for x in logs]
    losses = [float(x.get("policy_loss", 0.0)) for x in logs]

    payload = {
        "n_updates": len(logs),
        "updates": updates,
        "train_reward": rewards,
        "policy_loss": losses,
        "summary": {
            "reward_start": rewards[0],
            "reward_end": rewards[-1],
            "reward_mean": mean(rewards),
            "loss_start": losses[0],
            "loss_end": losses[-1],
            "loss_mean": mean(losses),
        },
    }
    dump_json(payload, args.output)
    print(f"[ok] wrote training curves: {args.output}")


if __name__ == "__main__":
    main()
