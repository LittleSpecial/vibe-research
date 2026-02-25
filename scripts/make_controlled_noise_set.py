#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import clamp, dump_json, dump_jsonl, load_jsonl, parse_csv_list, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled structured-noise preference set.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--truth_output", required=True)
    parser.add_argument("--annotator_reliabilities", default="0.95,0.8,0.65,0.55")
    parser.add_argument("--bias_channels", default="step_index,repetition,length,difficulty,source")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    rows = load_jsonl(args.input)
    if not rows:
        raise SystemExit(f"no preference rows found: {args.input}")

    rels = [clamp(float(x), 0.05, 0.99) for x in parse_csv_list(args.annotator_reliabilities)]
    if not rels:
        rels = [0.95, 0.8, 0.65, 0.55]

    annotators = sorted({str(r.get("annotator_id", "")) for r in rows if str(r.get("annotator_id", "")).strip()})
    if not annotators:
        annotators = [f"annotator_{i+1}" for i in range(len(rels))]

    truth = {
        "annotator_truth": {
            aid: rels[i % len(rels)] for i, aid in enumerate(annotators)
        },
        "bias_channels": parse_csv_list(args.bias_channels),
        "seed": args.seed,
    }

    out: list[dict] = []
    for row in rows:
        item = dict(row)
        aid = str(item.get("annotator_id", ""))
        if not aid:
            aid = annotators[0]
            item["annotator_id"] = aid
        rel = float(truth["annotator_truth"].get(aid, rels[0]))

        base_label = int(item.get("gold_pref", item.get("label", 0)))
        label = base_label
        if rng.random() > rel:
            label = 1 - base_label

        item["label"] = int(label)
        item["annotator_reliability_truth"] = rel
        out.append(item)

    dump_jsonl(out, args.output)
    dump_json(truth, args.truth_output)
    print(f"[ok] wrote controlled prefs: {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()
