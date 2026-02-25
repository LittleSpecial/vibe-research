#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import dump_json, load_json, spearman_corr


def main() -> None:
    parser = argparse.ArgumentParser(description="Score reliability recovery vs synthetic truth.")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--posterior", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    truth = load_json(args.truth, default={})
    post = load_json(args.posterior, default={})

    tmap = truth.get("annotator_truth", {}) if isinstance(truth, dict) else {}
    pmap = post.get("posterior", {}) if isinstance(post, dict) else {}

    common = sorted(set(tmap.keys()) & set(pmap.keys()))
    true_vals = [float(tmap[k]) for k in common]
    pred_vals = [float(pmap[k].get("mean", 0.0)) for k in common]

    spearman = spearman_corr(true_vals, pred_vals) if common else 0.0
    payload = {
        "n_common": len(common),
        "annotators": common,
        "spearman": spearman,
        "truth": {k: tmap[k] for k in common},
        "posterior": {k: pmap[k] for k in common},
    }
    dump_json(payload, args.output)
    print(f"[ok] reliability recovery spearman={spearman:.4f}")


if __name__ == "__main__":
    main()
