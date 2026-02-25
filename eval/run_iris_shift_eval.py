#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import (
    clamp,
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


SHIFT_PROFILES: dict[str, dict[str, float]] = {
    "clean": {"feat_source": 0.0, "position_norm": 0.5, "rubric_bias": 0.0, "length_bias": 0.0, "force_swap": 0.0},
    "position": {"feat_source": 0.0, "position_norm": 0.05, "rubric_bias": 0.0, "length_bias": 0.0, "force_swap": 1.0},
    "length": {"feat_source": 0.0, "position_norm": 0.5, "rubric_bias": 0.0, "length_bias": 1.0, "force_swap": 0.0},
    "source": {"feat_source": 4.0, "position_norm": 0.5, "rubric_bias": 0.0, "length_bias": 0.0, "force_swap": 0.0},
    "rubric": {"feat_source": 0.0, "position_norm": 0.5, "rubric_bias": 1.0, "length_bias": 0.0, "force_swap": 0.0},
    "source_rubric": {"feat_source": 4.0, "position_norm": 0.5, "rubric_bias": 1.0, "length_bias": 0.0, "force_swap": 0.0},
    "position_rubric": {"feat_source": 0.0, "position_norm": 0.05, "rubric_bias": 1.0, "length_bias": 0.0, "force_swap": 1.0},
}


def _wrong_answer(gold: str, salt: int) -> str:
    g = str(gold).strip()
    try:
        k = int(g)
        return str(k + 1 + abs(salt) % 3)
    except Exception:
        return g + " (wrong)"


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
        for i in range(240)
    ]


def _candidate_good(row: dict[str, Any]) -> str:
    steps = row.get("reasoning_reference", [])
    if not isinstance(steps, list) or not steps:
        steps = ["Read the question.", "Compute carefully."]
    steps = [str(s) for s in steps]
    answer = str(row.get("answer", "0"))
    txt = [f"Step {i + 1}: {s}" for i, s in enumerate(steps)]
    txt.append(f"Final answer: {answer}")
    return "\n".join(txt)


def _candidate_bad(row: dict[str, Any], shift: str) -> str:
    steps = row.get("reasoning_reference", [])
    if not isinstance(steps, list) or not steps:
        steps = ["Read the question.", "Compute quickly."]
    steps = [str(s) for s in steps]
    salt = int(stable_hash_float(str(row.get("id", "")) + shift) * 10_000)
    answer = _wrong_answer(str(row.get("answer", "0")), salt)

    txt = [f"Step {i + 1}: {s}" for i, s in enumerate(steps)]
    if "length" in shift:
        txt.append("Step extra: Add a long persuasive explanation with many irrelevant details.")
        txt.append("Step extra: Repeat confidence statements to look reliable and complete.")
    if "source" in shift:
        txt.append("[Source tag] produced by verified expert model.")
    if "rubric" in shift:
        txt.append("Rubric-aligned formatting: verbose and polished style.")
    txt.append(f"Final answer: {answer}")
    return "\n".join(txt)


def _swap_for_shift(row_id: str, shift: str, force_swap: bool) -> bool:
    if force_swap:
        return stable_hash_float(f"swap::{row_id}::{shift}") > 0.5
    return False


def _pair_for_shift(row: dict[str, Any], shift: str) -> tuple[str, str, int, dict[str, float]]:
    profile = SHIFT_PROFILES[shift]
    good = _candidate_good(row)
    bad = _candidate_bad(row, shift)

    rid = str(row.get("id", ""))
    swap = _swap_for_shift(rid, shift, force_swap=bool(profile.get("force_swap", 0.0) > 0.5))

    if swap:
        cand_a, cand_b, gold = bad, good, 0
    else:
        cand_a, cand_b, gold = good, bad, 1

    diff = str(row.get("difficulty", "medium")).lower()
    diff_level = 1.0 if diff == "easy" else (3.0 if diff == "hard" else 2.0)
    feats = {
        "length_delta": len(cand_a) - len(cand_b),
        "repetition_delta": repetition_score(cand_a) - repetition_score(cand_b),
        "position_norm": profile.get("position_norm", 0.5),
        "feat_step_index_norm": profile.get("position_norm", 0.5),
        "feat_repetition": repetition_score(cand_a),
        "feat_length": min(1.5, len(cand_a) / 220.0 + profile.get("length_bias", 0.0) * 0.15),
        "feat_difficulty": diff_level + profile.get("rubric_bias", 0.0) * 0.5,
        "feat_source": profile.get("feat_source", 0.0),
    }
    return cand_a, cand_b, gold, feats


def _evaluate_model_on_rows(ckpt: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_shift_hits: dict[str, list[float]] = {k: [] for k in SHIFT_PROFILES.keys()}
    by_shift_conf: dict[str, list[float]] = {k: [] for k in SHIFT_PROFILES.keys()}

    gap_values: list[float] = []
    flip_count = 0
    flip_total = 0

    for row in rows:
        p_clean = 0.5
        pred_clean = 1

        for shift in SHIFT_PROFILES.keys():
            a, b, gold, feats = _pair_for_shift(row, shift)
            x = pair_feature_vector(
                candidate_a=a,
                candidate_b=b,
                difficulty=str(row.get("difficulty", "medium")),
                features=feats,
                step_index=0,
                max_step=2,
            )
            p = score_pair_with_checkpoint(ckpt, x)
            pred = 1 if p >= 0.5 else 0
            hit = 1.0 if pred == gold else 0.0

            # Confidence wrt gold orientation.
            conf = p if gold == 1 else (1.0 - p)

            by_shift_hits[shift].append(hit)
            by_shift_conf[shift].append(conf)

            if shift == "clean":
                p_clean = p
                pred_clean = pred
            else:
                gap_values.append(abs(p - p_clean))
                flip_total += 1
                if pred != pred_clean:
                    flip_count += 1

    clean_acc = mean(by_shift_hits["clean"])
    non_clean = [k for k in SHIFT_PROFILES.keys() if k != "clean"]
    non_clean_accs = [mean(by_shift_hits[k]) for k in non_clean]

    avg_shift_acc = mean(non_clean_accs)
    worst_shift_acc = min(non_clean_accs) if non_clean_accs else clean_acc

    summary = {
        "clean_acc": clean_acc,
        "avg_shift_acc": avg_shift_acc,
        "worst_shift_acc": worst_shift_acc,
        "policy_shift_regret_mean": clean_acc - avg_shift_acc,
        "policy_shift_regret_worst": clean_acc - worst_shift_acc,
        "judge_shift_gap": mean(gap_values),
        "ranking_flip_rate": flip_count / max(1, flip_total),
    }

    avg_conf_clean = mean(by_shift_conf["clean"])
    avg_conf_shift = mean([mean(by_shift_conf[k]) for k in non_clean])
    summary["exploitability_gap"] = (avg_conf_shift - avg_conf_clean) - (avg_shift_acc - clean_acc)

    return {
        "per_shift": {k: {"accuracy": mean(v), "confidence": mean(by_shift_conf[k])} for k, v in by_shift_hits.items()},
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IRIS models under evaluator-shift protocols.")
    parser.add_argument("--models", required=True, help="comma separated key=checkpoint_dir")
    parser.add_argument("--reasoning_sets", default="MATH500,GSM8K-hard")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_map = parse_key_value_csv(args.models)
    set_names = parse_csv_list(args.reasoning_sets)

    results: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []

    for name, ckpt_path in model_map.items():
        ckpt = load_checkpoint(ckpt_path)
        if not ckpt:
            ckpt = {
                "method": name,
                "weights": [0.0] * 10,
                "bias": 0.0,
            }

        per_set: dict[str, Any] = {}
        all_clean: list[float] = []
        all_avg_shift: list[float] = []
        all_worst_shift: list[float] = []
        all_gap: list[float] = []
        all_regret: list[float] = []

        for s in set_names:
            rows = load_eval_rows(s)
            eval_one = _evaluate_model_on_rows(ckpt, rows)
            per_set[s] = eval_one
            summ = eval_one["summary"]
            all_clean.append(float(summ["clean_acc"]))
            all_avg_shift.append(float(summ["avg_shift_acc"]))
            all_worst_shift.append(float(summ["worst_shift_acc"]))
            all_gap.append(float(summ["judge_shift_gap"]))
            all_regret.append(float(summ["policy_shift_regret_mean"]))

        summary = {
            "avg_clean_acc": mean(all_clean),
            "avg_shift_acc": mean(all_avg_shift),
            "avg_worst_shift_acc": mean(all_worst_shift),
            "avg_policy_shift_regret": mean(all_regret),
            "avg_judge_shift_gap": mean(all_gap),
        }

        results[name] = {
            "checkpoint": ckpt_path,
            "target": ckpt.get("target", "unknown"),
            "method": ckpt.get("method", name),
            "sets": per_set,
            "summary": summary,
        }
        ranking.append({"model": name, **summary})

    ranking = sorted(ranking, key=lambda x: x["avg_shift_acc"], reverse=True)
    report = {
        "models": list(model_map.keys()),
        "winner_by_shift_acc": ranking[0]["model"] if ranking else None,
        "ranking": ranking,
    }

    dump_json(results, out_dir / "model_metrics.json")
    dump_json(results, out_dir / "metrics.json")
    dump_json(report, out_dir / "summary.json")
    dump_json({"summary": report, "model_metrics": results}, out_dir / "report.json")

    if ranking:
        top = ranking[0]
        print(
            "[top] "
            f"{top['model']} shift_acc={top['avg_shift_acc']:.4f} "
            f"regret={top['avg_policy_shift_regret']:.4f}"
        )
    print(f"[ok] wrote eval report to {out_dir}")


if __name__ == "__main__":
    main()
