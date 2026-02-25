#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import dump_json, parse_int_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Check total GPU-hours against limit.")
    parser.add_argument("--item_gpu_hours", required=True, help="comma-separated numbers")
    parser.add_argument("--gpu_limit", type=float, required=True)
    parser.add_argument("--wall_clock_hours", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    items = parse_int_list(args.item_gpu_hours)
    total = sum(items)
    limit = float(args.gpu_limit) * float(args.wall_clock_hours)
    payload = {
        "items": items,
        "estimated_total_gpu_hours": total,
        "limit_gpu_hours": limit,
        "pass": total <= limit,
    }
    dump_json(payload, args.output)
    print(f"[ok] budget check: total={total:.2f} limit={limit:.2f} pass={payload['pass']}")


if __name__ == "__main__":
    main()
