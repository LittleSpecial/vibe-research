#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import (
    difficulty_to_level,
    dump_json,
    load_jsonl,
    parse_csv_list,
    set_seed,
    stable_hash_float,
    try_dump_table,
)


def build_rows(data: list[dict], features: list[str]) -> list[dict]:
    out: list[dict] = []
    for row in data:
        pid = str(row.get("id", ""))
        difficulty = str(row.get("difficulty", "medium"))
        diff_level = difficulty_to_level(difficulty)
        # Pre-materialize a few nominal step indices to support step-level joins.
        max_step = 6 if diff_level >= 3 else 4
        for step_idx in range(max_step):
            base = {
                "problem_id": pid,
                "difficulty": difficulty,
                "difficulty_level": diff_level,
                "step_index": step_idx,
            }
            if "step_index" in features:
                base["feat_step_index_norm"] = step_idx / max(1, max_step - 1)
            if "repetition" in features:
                base["feat_repetition"] = stable_hash_float(f"rep::{pid}::{step_idx}")
            if "length" in features:
                base["feat_length"] = 0.35 + stable_hash_float(f"len::{pid}::{step_idx}") * 0.65
            if "difficulty" in features:
                base["feat_difficulty"] = float(diff_level)
            if "source" in features:
                base["feat_source"] = int(stable_hash_float(f"src::{pid}") * 5)
            out.append(base)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured-noise feature table from train set.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", default="step_index,repetition,length,difficulty,source")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = load_jsonl(args.input)
    if not rows:
        raise SystemExit(f"no rows found in {args.input}")

    feature_names = parse_csv_list(args.features)
    out_rows = build_rows(rows, feature_names)
    try_dump_table(out_rows, args.output)
    dump_json(
        {
            "input": args.input,
            "output": args.output,
            "rows": len(out_rows),
            "features": feature_names,
            "format": "jsonl-table",
        },
        f"{args.output}.meta.json",
    )
    print(f"[ok] wrote {len(out_rows)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
