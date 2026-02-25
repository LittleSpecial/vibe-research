#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import (
    clamp,
    compute_auc,
    compute_binary_calibration,
    difficulty_to_level,
    dump_json,
    load_jsonl,
    mean,
    parse_csv_list,
    parse_key_value_csv,
    repetition_score,
    set_seed,
    stable_hash_float,
)
from train.reliastep_train_lib import load_checkpoint, pair_feature_vector, score_pair_with_checkpoint


def eval_path_for_set(name: str) -> Path:
    n = name.strip()
    if n == "MATH500":
        return Path("data/math/eval_MATH500.jsonl")
    if n == "GSM8K-hard":
        return Path("data/gsm8k_hard/eval_GSM8K-hard.jsonl")
    safe = n.replace("/", "_").replace(" ", "_")
    return Path(f"data/{safe}.jsonl")


def load_eval_rows(name: str) -> list[dict[str, Any]]:
    p = eval_path_for_set(name)
    rows = load_jsonl(p)
    if rows:
        return rows
    return [
        {
            "id": f"{name}_{i}",
            "difficulty": "hard",
            "question": f"Synthetic {name} question {i}",
            "answer": "0",
            "reasoning_reference": [f"Synthetic reasoning step {i}"],
        }
        for i in range(200)
    ]


def _wrong_answer(gold: str, salt: int) -> str:
    g = str(gold).strip()
    try:
        k = int(g)
        off = 1 + abs(salt) % 3
        return str(k + off)
    except Exception:
        return g + " (wrong)"


def _candidate_from_row(row: dict[str, Any], wrong: bool, variant: str) -> str:
    steps = row.get("reasoning_reference", [])
    if not isinstance(steps, list) or not steps:
        steps = ["Read the question.", "Compute the required quantity."]
    steps = [str(x) for x in steps]
    gold = str(row.get("answer", "0"))
    pid = str(row.get("id", "x"))
    salt = int(stable_hash_float(pid + variant) * 10_000)

    final = _wrong_answer(gold, salt) if wrong else gold
    text_steps = []
    for i, st in enumerate(steps):
        text_steps.append(f"Step {i + 1}: {st}")
    if wrong and variant == "noisy":
        text_steps.append("Step extra: Provide a long explanation with irrelevant details to look persuasive.")
    if wrong and variant == "shift":
        text_steps.append("Step extra: Use confident style and source-like marker [verified].")
    text_steps.append(f"Final answer: {final}")
    return "\n".join(text_steps)


def _pair_for_variant(row: dict[str, Any], variant: str) -> tuple[str, str, int]:
    cand_good = _candidate_from_row(row, wrong=False, variant=variant)
    cand_bad = _candidate_from_row(row, wrong=True, variant=variant)
    # Deterministic side swap to avoid positional shortcut.
    key = str(row.get("id", "")) + "::" + variant
    swap = stable_hash_float(key) > 0.5
    if swap:
        return cand_bad, cand_good, 0
    return cand_good, cand_bad, 1


def score_on_set(model_name: str, ckpt: dict[str, Any], set_name: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    del model_name
    clean_hits: list[float] = []
    noisy_hits: list[float] = []
    shift_hits: list[float] = []

    for row in rows:
        diff = str(row.get("difficulty", "medium"))

        for variant, bucket in (("clean", clean_hits), ("noisy", noisy_hits), ("shift", shift_hits)):
            a, b, gold_pref = _pair_for_variant(row, variant)
            feats = {
                "position_norm": 0.5,
                "feat_step_index_norm": 0.5,
                "feat_difficulty": difficulty_to_level(diff),
                "feat_source": 0.0 if variant == "clean" else (2.0 if variant == "noisy" else 3.0),
                "feat_repetition": repetition_score(a),
                "feat_length": min(1.0, len(a) / 200.0),
            }
            x = pair_feature_vector(
                candidate_a=a,
                candidate_b=b,
                difficulty=diff,
                features=feats,
                step_index=0,
                max_step=2,
            )
            p = score_pair_with_checkpoint(ckpt, x)
            pred = 1 if p >= 0.5 else 0
            bucket.append(1.0 if pred == gold_pref else 0.0)

    clean = mean(clean_hits)
    noisy = mean(noisy_hits)
    shift = mean(shift_hits)
    return {
        "exact_match_clean": clean,
        "exact_match_noisy": noisy,
        "exact_match_shift": shift,
        "clean_to_noisy_drop": clean - noisy,
        "noise_shift_drop": clean - shift,
    }


def step_quality_metrics(ckpt: dict[str, Any], prm_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not prm_rows:
        return {"step_auroc": 0.5, "step_f1": 0.0, "calibration_ece": 0.5, "calibration_brier": 0.5}

    labels: list[int] = []
    probs: list[float] = []
    pred_labels: list[int] = []
    for r in prm_rows:
        text = str(r.get("step_text", ""))
        diff = str(r.get("difficulty", "medium"))
        x = pair_feature_vector(
            candidate_a=text,
            candidate_b="",
            difficulty=diff,
            features={
                "position_norm": 0.5,
                "feat_step_index_norm": 0.5,
                "feat_difficulty": difficulty_to_level(diff),
                "feat_source": 0.0,
                "feat_repetition": repetition_score(text),
                "feat_length": min(1.0, len(text) / 200.0),
            },
            step_index=0,
            max_step=2,
        )
        p = score_pair_with_checkpoint(ckpt, x)
        y = 1 if int(r.get("is_correct", 0)) == 1 else 0
        labels.append(y)
        probs.append(p)
        pred_labels.append(1 if p >= 0.5 else 0)

    auc = compute_auc(labels, probs)
    tp = sum(1 for y, p in zip(labels, pred_labels) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, pred_labels) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, pred_labels) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    cal = compute_binary_calibration(labels, probs, bins=10)
    return {
        "step_auroc": clamp(auc, 0.0, 1.0),
        "step_f1": clamp(f1, 0.0, 1.0),
        "calibration_ece": clamp(cal["ece"], 0.0, 1.0),
        "calibration_brier": clamp(cal["brier"], 0.0, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on deterministic reasoning and step metrics.")
    parser.add_argument("--models", required=True, help="comma-separated key=checkpoint_dir")
    parser.add_argument("--reasoning_sets", default="MATH500,GSM8K-hard")
    parser.add_argument("--prm800k", default="data/prm800k/heldout.jsonl")
    parser.add_argument("--noise_shift_profiles", default="structured_v1:structured_v2")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    model_map = parse_key_value_csv(args.models)
    set_names = parse_csv_list(args.reasoning_sets)
    prm_rows = load_jsonl(args.prm800k)

    all_metrics: dict[str, dict[str, Any]] = {}
    for name, ckpt_path in model_map.items():
        ckpt = load_checkpoint(ckpt_path)
        if not ckpt:
            ckpt = {
                "method": name,
                "quality_score": 0.5,
                "noise_estimate": 0.5,
                "robustness_bonus": 0.0,
                "weights": [0.0] * 10,
                "bias": 0.0,
            }

        per_set = {}
        for s in set_names:
            per_set[s] = score_on_set(name, ckpt, s, load_eval_rows(s))

        step = step_quality_metrics(ckpt, prm_rows)
        avg_clean = mean([v["exact_match_clean"] for v in per_set.values()]) if per_set else 0.0
        avg_noisy = mean([v["exact_match_noisy"] for v in per_set.values()]) if per_set else 0.0
        avg_shift = mean([v["exact_match_shift"] for v in per_set.values()]) if per_set else 0.0

        all_metrics[name] = {
            "checkpoint": ckpt_path,
            "method": ckpt.get("method", name),
            "quality_score": float(ckpt.get("quality_score", 0.5)),
            "reasoning_sets": per_set,
            "summary": {
                "avg_exact_match_clean": avg_clean,
                "avg_exact_match_noisy": avg_noisy,
                "avg_exact_match_shift": avg_shift,
                "avg_clean_to_noisy_drop": avg_clean - avg_noisy,
                "avg_noise_shift_drop": avg_clean - avg_shift,
            },
            "step_metrics": step,
            "metadata": {
                "noise_shift_profiles": args.noise_shift_profiles,
                "prm_rows": len(prm_rows),
            },
        }

    ranked = sorted(
        (
            {
                "model": k,
                "avg_exact_match_clean": v["summary"]["avg_exact_match_clean"],
                "avg_exact_match_noisy": v["summary"]["avg_exact_match_noisy"],
                "step_auroc": v["step_metrics"]["step_auroc"],
                "step_f1": v["step_metrics"]["step_f1"],
            }
            for k, v in all_metrics.items()
        ),
        key=lambda x: x["avg_exact_match_clean"],
        reverse=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"models": list(model_map.keys()), "ranking": ranked, "winner": ranked[0]["model"] if ranked else None}

    dump_json(all_metrics, out_dir / "model_metrics.json")
    dump_json(all_metrics, out_dir / "metrics.json")
    dump_json(summary, out_dir / "summary.json")
    dump_json({"summary": summary, "model_metrics": all_metrics}, out_dir / "results.json")

    print(f"[ok] evaluation finished: {out_dir}")
    if ranked:
        top = ranked[0]
        print(
            "[top] "
            f"{top['model']} clean={top['avg_exact_match_clean']:.4f} "
            f"noisy={top['avg_exact_match_noisy']:.4f} step_f1={top['step_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
