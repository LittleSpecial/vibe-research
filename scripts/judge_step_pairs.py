#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import (
    clamp,
    difficulty_to_level,
    dump_json,
    dump_jsonl,
    load_json,
    load_jsonl,
    parse_csv_list,
    repetition_score,
    set_seed,
    try_load_table,
)


def load_judges(path: str) -> list[dict]:
    payload = load_json(path, default={})
    if isinstance(payload, dict) and isinstance(payload.get("personas"), list):
        return [x for x in payload["personas"] if isinstance(x, dict)]

    # Optional fallback for YAML, if installed.
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        if isinstance(y, dict) and isinstance(y.get("personas"), list):
            return [x for x in y["personas"] if isinstance(x, dict)]
    except Exception:
        pass

    raise SystemExit(f"failed to parse judges file: {path}")


def score_step_text(step: str) -> float:
    tokens = [w for w in step.lower().split() if w]
    if not tokens:
        return 0.0
    digits = sum(ch.isdigit() for ch in step)
    return 0.35 + 0.2 * min(1.0, digits / 6.0) + 0.45 * (1.0 - repetition_score(step))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create multi-annotator step-pair preferences.")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--judges", required=True)
    parser.add_argument("--judge_server", default="")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    rollouts = load_jsonl(args.rollouts)
    if not rollouts:
        raise SystemExit(f"no rollout rows found: {args.rollouts}")
    personas = load_judges(args.judges)
    feature_rows = try_load_table(args.features)

    feat_map: dict[tuple[str, int], dict] = {}
    for row in feature_rows:
        pid = str(row.get("problem_id", ""))
        step_idx = int(row.get("step_index", 0))
        feat_map[(pid, step_idx)] = row

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rollouts:
        grouped[str(r.get("problem_id", ""))].append(r)

    out: list[dict] = []
    pair_count = 0
    for pid, group in grouped.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda x: int(x.get("rollout_id", 0)))
        a, b = group[0], group[1]
        steps_a = a.get("steps", []) if isinstance(a.get("steps"), list) else []
        steps_b = b.get("steps", []) if isinstance(b.get("steps"), list) else []
        if not steps_a and not steps_b:
            continue

        gold_pref = 1 if int(a.get("is_correct", 0)) >= int(b.get("is_correct", 0)) else 0
        max_step = max(len(steps_a), len(steps_b))
        difficulty = str(a.get("difficulty", "medium"))
        diff_level = difficulty_to_level(difficulty)

        for step_idx in range(max_step):
            step_a = str(steps_a[step_idx]) if step_idx < len(steps_a) else ""
            step_b = str(steps_b[step_idx]) if step_idx < len(steps_b) else ""

            feat = feat_map.get((pid, step_idx), {})
            base = {
                "problem_id": pid,
                "pair_id": f"{pid}::s{step_idx}",
                "step_index": step_idx,
                "difficulty": difficulty,
                "difficulty_level": diff_level,
                "a_rollout_id": int(a.get("rollout_id", 0)),
                "b_rollout_id": int(b.get("rollout_id", 1)),
                "candidate_a": step_a,
                "candidate_b": step_b,
                "gold_pref": gold_pref,
                "features": {
                    "length_delta": len(step_a) - len(step_b),
                    "repetition_delta": repetition_score(step_a) - repetition_score(step_b),
                    "position_norm": step_idx / max(1, max_step - 1),
                    "source_delta": 0,
                    "feat_step_index_norm": float(feat.get("feat_step_index_norm", 0.0) or 0.0),
                    "feat_repetition": float(feat.get("feat_repetition", 0.0) or 0.0),
                    "feat_length": float(feat.get("feat_length", 0.0) or 0.0),
                    "feat_difficulty": float(feat.get("feat_difficulty", diff_level) or diff_level),
                    "feat_source": float(feat.get("feat_source", 0.0) or 0.0),
                },
            }

            step_quality = score_step_text(step_a) - score_step_text(step_b)

            for j in personas:
                annotator_id = str(j.get("id", "judge"))
                reliability = float(j.get("reliability", 0.7))
                bias = j.get("bias", {}) if isinstance(j.get("bias"), dict) else {}
                length_bias = float(bias.get("length", 0.0))
                pos_bias = float(bias.get("position", 0.0))
                rep_bias = float(bias.get("repetition", 0.0))
                src_bias = float(bias.get("source", 0.0))

                pref_logit = (
                    (1.0 if gold_pref == 1 else -1.0) * 1.15
                    + step_quality
                    + length_bias * clamp((len(step_a) - len(step_b)) / 60.0, -1.0, 1.0)
                    + pos_bias * ((step_idx / max(1, max_step - 1)) - 0.5)
                    + rep_bias * (repetition_score(step_a) - repetition_score(step_b))
                    + src_bias * float(base["features"]["feat_source"]) / 5.0
                )
                label = 1 if pref_logit >= 0 else 0

                # Reliability controls probability of flipping final decision.
                if rng.random() > clamp(reliability, 0.05, 0.99):
                    label = 1 - label

                row = {
                    **base,
                    "annotator_id": annotator_id,
                    "annotator_kind": j.get("kind", "llm"),
                    "annotator_model": j.get("model", "unknown"),
                    "annotator_reliability_truth": reliability,
                    "label": int(label),
                }
                out.append(row)
            pair_count += 1

    dump_jsonl(out, args.output)
    dump_json(
        {
            "rollouts": args.rollouts,
            "judges": args.judges,
            "features": args.features,
            "output": args.output,
            "pair_count": pair_count,
            "records": len(out),
            "annotators": sorted({str(x.get("annotator_id", "")) for x in out}),
        },
        f"{args.output}.meta.json",
    )
    print(f"[ok] wrote step preferences: {args.output} ({len(out)} records)")


if __name__ == "__main__":
    main()
