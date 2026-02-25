#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import clamp, dump_json, load_jsonl


def to_sign(v: object) -> int:
    try:
        return 1 if int(v) == 1 else -1
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer latent annotator reliability from preference labels.")
    parser.add_argument("--prefs", required=True)
    parser.add_argument("--features", required=False, default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.prefs)
    if not rows:
        raise SystemExit(f"no prefs found: {args.prefs}")

    stats: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        aid = str(r.get("annotator_id", "unknown"))
        label = to_sign(r.get("label", 0))
        gold = to_sign(r.get("gold_pref", r.get("label", 0)))
        stats[aid].append(1 if label == gold else 0)

    post = {}
    for aid, vals in stats.items():
        n = len(vals)
        rel = clamp((sum(vals) + 1.0) / (n + 2.0), 0.05, 0.99)
        post[aid] = {"mean": rel, "count": n}

    dump_json({"posterior": post}, args.output)
    print(f"[ok] wrote posterior reliability: {args.output}")


if __name__ == "__main__":
    main()
