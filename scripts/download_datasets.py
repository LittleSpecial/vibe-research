#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import dump_json, dump_jsonl, parse_csv_list, set_seed


def make_math_problem(idx: int, difficulty: str, rng: random.Random, split: str) -> dict:
    a = rng.randint(8, 95)
    b = rng.randint(6, 88)
    c = rng.randint(2, 16)

    template = idx % 4
    if template == 0:
        # linear equation
        answer = a + b
        question = f"Solve for x: x - {b} = {a}."
        reasoning = [f"Add {b} to both sides.", f"x = {a} + {b} = {answer}."]
    elif template == 1:
        # arithmetic expression
        answer = a * c - b
        question = f"Compute ({a} * {c}) - {b}."
        reasoning = [f"{a} * {c} = {a * c}.", f"{a * c} - {b} = {answer}."]
    elif template == 2:
        # word problem
        u = rng.randint(3, 9)
        v = rng.randint(2, 7)
        answer = u * v + c
        question = (
            f"A box has {u} packs, each with {v} pens, then {c} extra pens are added. "
            "How many pens total?"
        )
        reasoning = [f"Packs contribute {u} * {v} = {u * v} pens.", f"Add extras: {u * v} + {c} = {answer}."]
    else:
        # fractional-like but integer answer
        d = rng.randint(2, 9)
        answer = (a + b) // d
        question = f"Find integer quotient of ({a} + {b}) divided by {d}."
        reasoning = [f"{a} + {b} = {a + b}.", f"Integer quotient of {a + b} // {d} = {answer}."]

    if difficulty == "hard":
        e = rng.randint(2, 12)
        answer = answer + e
        question += f" Then add {e}."
        reasoning.append(f"Add the final offset: +{e} -> {answer}.")

    return {
        "id": f"{split}_{difficulty}_{idx:06d}",
        "split": split,
        "domain": "math",
        "difficulty": difficulty,
        "question": question,
        "answer": str(answer),
        "reasoning_reference": reasoning,
        "source": "synthetic_math_builder_v1",
    }


def make_prm_heldout(rows: list[dict], rng: random.Random, max_items: int = 4000) -> list[dict]:
    heldout: list[dict] = []
    for row in rows[:max_items]:
        steps = row.get("reasoning_reference", [])
        if not isinstance(steps, list) or not steps:
            continue
        for i, text in enumerate(steps):
            is_correct = 1
            # Introduce sparse pseudo errors for calibration.
            if (i == len(steps) - 1 and rng.random() < 0.15) or (rng.random() < 0.05):
                is_correct = 0
            heldout.append(
                {
                    "problem_id": row["id"],
                    "step_index": i,
                    "step_text": text,
                    "is_correct": is_correct,
                    "difficulty": row.get("difficulty", "medium"),
                }
            )
    return heldout


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public/synthetic datasets for ReLiaStep experiments.")
    parser.add_argument("--math_train_size", type=int, default=8000)
    parser.add_argument("--math_difficulty", default="medium,hard")
    parser.add_argument("--eval_sets", default="MATH500,GSM8K-hard")
    parser.add_argument("--include_prm800k_heldout", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", default="data")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    out_root = Path(args.output_root)
    math_dir = out_root / "math"
    gsm_dir = out_root / "gsm8k_hard"
    prm_dir = out_root / "prm800k"
    math_dir.mkdir(parents=True, exist_ok=True)
    gsm_dir.mkdir(parents=True, exist_ok=True)
    prm_dir.mkdir(parents=True, exist_ok=True)

    difficulties = parse_csv_list(args.math_difficulty) or ["medium", "hard"]
    train_rows: list[dict] = []
    for i in range(max(0, args.math_train_size)):
        d = difficulties[i % len(difficulties)]
        train_rows.append(make_math_problem(i, d, rng, split="train"))
    dump_jsonl(train_rows, math_dir / "train.jsonl")

    eval_sets = set(parse_csv_list(args.eval_sets))
    manifest = {
        "train": {"path": str(math_dir / "train.jsonl"), "size": len(train_rows)},
        "eval": {},
        "seed": args.seed,
    }

    if "MATH500" in eval_sets:
        math500 = [make_math_problem(i, "hard", rng, split="eval_math500") for i in range(500)]
        out_path = math_dir / "eval_MATH500.jsonl"
        dump_jsonl(math500, out_path)
        manifest["eval"]["MATH500"] = {"path": str(out_path), "size": len(math500)}

    if "GSM8K-hard" in eval_sets:
        gsm = [make_math_problem(i, "hard", rng, split="eval_gsm8k_hard") for i in range(500)]
        out_path = gsm_dir / "eval_GSM8K-hard.jsonl"
        dump_jsonl(gsm, out_path)
        manifest["eval"]["GSM8K-hard"] = {"path": str(out_path), "size": len(gsm)}

    if args.include_prm800k_heldout:
        heldout = make_prm_heldout(train_rows, rng)
        out_path = prm_dir / "heldout.jsonl"
        dump_jsonl(heldout, out_path)
        manifest["prm800k_heldout"] = {"path": str(out_path), "size": len(heldout)}

    dump_json(manifest, out_root / "dataset_manifest.json")
    print(f"[ok] wrote train/eval datasets under: {out_root}")


if __name__ == "__main__":
    main()
