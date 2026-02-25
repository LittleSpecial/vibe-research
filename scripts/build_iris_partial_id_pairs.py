#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import clamp, dump_json, dump_jsonl, load_jsonl, mean, set_seed, tokenize_words


SHIFT_NAMES = ["clean", "position", "length", "source", "rubric"]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _to_sign(label: Any) -> float:
    try:
        return 1.0 if int(label) == 1 else -1.0
    except Exception:
        return -1.0


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    total_w = sum(weights)
    if total_w <= 1e-9:
        return mean(values)
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _weighted_var(values: list[float], weights: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _weighted_mean(values, weights)
    total_w = sum(weights)
    if total_w <= 1e-9:
        return 0.0
    return sum(w * (v - m) ** 2 for v, w in zip(values, weights)) / total_w


def _token_jaccard(a: str, b: str) -> float:
    sa = set(tokenize_words(a))
    sb = set(tokenize_words(b))
    if not sa and not sb:
        return 1.0
    inter = len(sa.intersection(sb))
    uni = len(sa.union(sb))
    return inter / max(1, uni)


def _shift_score(row: dict[str, Any], shift: str) -> float:
    feats = row.get("features") if isinstance(row.get("features"), dict) else {}
    label_sign = _to_sign(row.get("label", 0))

    pos_norm = clamp(_safe_float(feats.get("position_norm", 0.5)), 0.0, 1.0)
    len_delta = clamp(_safe_float(feats.get("length_delta", 0.0)) / 180.0, -1.5, 1.5)
    rep_delta = clamp(_safe_float(feats.get("repetition_delta", 0.0)), -1.0, 1.0)
    src_norm = clamp(_safe_float(feats.get("feat_source", 0.0)) / 5.0, 0.0, 1.0)
    diff_norm = clamp(_safe_float(feats.get("feat_difficulty", 2.0)) / 3.0, 0.0, 1.5)

    base = label_sign
    if shift == "clean":
        return base
    if shift == "position":
        return base + 0.55 * (0.5 - pos_norm)
    if shift == "length":
        return base + 0.50 * len_delta
    if shift == "source":
        return base + 0.45 * (src_norm - 0.5)
    if shift == "rubric":
        return base + 0.30 * (diff_norm - 0.5) + 0.20 * rep_delta
    return base


def build_partial_id_pairs(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    zeta0: float,
    alpha: float,
    beta: float,
    gamma: float,
    eta_scale: float,
    min_jaccard: float,
    max_disagreement: float,
    generic_lambda: float,
    calibrate_offset: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("pair_id", "")).strip()
        if not key:
            pid = str(row.get("problem_id", ""))
            step = int(row.get("step_index", 0) or 0)
            key = f"{pid}::s{step}"
            row["pair_id"] = key
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    provisional: list[dict[str, Any]] = []
    dropped = 0
    purity_pass_count = 0
    ql_values: list[float] = []
    ql_raw_values: list[float] = []
    clean_scores: list[float] = []
    shift_gap_values: list[float] = []

    for pair_id, group in grouped.items():
        if not group:
            continue
        first = group[0]
        cand_a = str(first.get("candidate_a", ""))
        cand_b = str(first.get("candidate_b", ""))

        weights = [clamp(_safe_float(r.get("annotator_reliability_truth", 0.7)), 0.05, 0.99) for r in group]
        score_table: dict[str, list[float]] = {s: [] for s in SHIFT_NAMES}
        for row in group:
            for shift in SHIFT_NAMES:
                score_table[shift].append(_shift_score(row, shift))

        agg_score = {s: _weighted_mean(score_table[s], weights) for s in SHIFT_NAMES}
        clean = agg_score["clean"]
        clean_var = _weighted_var(score_table["clean"], weights)
        clean_std = math.sqrt(max(0.0, clean_var))

        disagreement = clamp(clean_var / 1.0, 0.0, 1.0)
        format_drift = clamp(abs(_safe_float((first.get("features") or {}).get("length_delta", 0.0))) / 180.0, 0.0, 1.0)
        token_jaccard = _token_jaccard(cand_a, cand_b)

        purity_pass = (token_jaccard >= min_jaccard) and (disagreement <= max_disagreement)
        if mode == "hard" and not purity_pass:
            dropped += 1
            continue
        if purity_pass:
            purity_pass_count += 1

        if mode == "hard":
            zeta = zeta0
        else:
            zeta = alpha * (1.0 - token_jaccard) + beta * disagreement + gamma * format_drift
            zeta = clamp(zeta, 0.0, 1.25)

        eta = eta_scale * clean_std

        upper_terms: list[float] = []
        shift_gap = 0.0
        for shift in SHIFT_NAMES:
            if shift == "clean":
                continue
            delta_r = agg_score[shift] - clean
            upper_terms.append(delta_r + zeta + eta)
            shift_gap = max(shift_gap, abs(delta_r))

        u_bound = max(0.0, max(upper_terms) if upper_terms else 0.0)
        q_l_raw = clean - u_bound
        q_generic = clean - generic_lambda * math.sqrt(max(1e-8, clean_var))

        item = {
            "pair_id": pair_id,
            "problem_id": str(first.get("problem_id", "")),
            "step_index": int(first.get("step_index", 0) or 0),
            "difficulty": str(first.get("difficulty", "medium")),
            "candidate_a": cand_a,
            "candidate_b": cand_b,
            "gold_pref": int(first.get("gold_pref", 0) or 0),
            "n_annotators": len(group),
            "purity_pass": bool(purity_pass),
            "purity": {
                "token_jaccard": token_jaccard,
                "disagreement": disagreement,
                "format_drift": format_drift,
            },
            "r_scores": agg_score,
            "r_clean": clean,
            "zeta": zeta,
            "eta": eta,
            "u_bound": u_bound,
            "max_shift_gap": shift_gap,
            "q_l_raw": q_l_raw,
            "q_generic": q_generic,
            "features": first.get("features", {}),
        }
        provisional.append(item)
        ql_raw_values.append(q_l_raw)
        clean_scores.append(clean)
        shift_gap_values.append(shift_gap)

    offset = 0.0
    if provisional and calibrate_offset:
        offset = mean(clean_scores) - mean(ql_raw_values)

    for item in provisional:
        q_l = float(item["q_l_raw"]) + offset
        q_generic = float(item["q_generic"])
        clean = float(item["r_clean"])
        item["q_l"] = q_l
        item["label"] = 1 if q_l >= 0 else 0
        item["label_generic"] = 1 if q_generic >= 0 else 0
        item["label_raw"] = 1 if clean >= 0 else 0
        item["weight"] = clamp(abs(q_l), 0.05, 3.0)
        item["weight_generic"] = clamp(abs(q_generic), 0.05, 3.0)
        item["weight_raw"] = 1.0
        out.append(item)
        ql_values.append(q_l)

    summary = {
        "pairs_total": len(grouped),
        "pairs_output": len(out),
        "pairs_dropped": dropped,
        "purity_pass_rate": purity_pass_count / max(1, len(grouped)),
        "mode": mode,
        "zeta_config": {
            "zeta0": zeta0,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "eta_scale": eta_scale,
            "min_jaccard": min_jaccard,
            "max_disagreement": max_disagreement,
            "calibrate_offset": calibrate_offset,
            "offset_applied": offset,
        },
        "stats": {
            "q_l_raw_mean": mean(ql_raw_values),
            "q_l_mean": mean(ql_values),
            "q_l_min": min(ql_values) if ql_values else 0.0,
            "q_l_max": max(ql_values) if ql_values else 0.0,
            "r_clean_mean": mean(clean_scores),
            "max_shift_gap_mean": mean(shift_gap_values),
        },
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IRIS partial-identification training pairs from judged preferences.")
    parser.add_argument("--input", required=True, help="step preference jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--mode", choices=["hard", "soft"], default="soft")
    parser.add_argument("--zeta0", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.55)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--eta_scale", type=float, default=0.65)
    parser.add_argument("--min_jaccard", type=float, default=0.05)
    parser.add_argument("--max_disagreement", type=float, default=0.65)
    parser.add_argument("--generic_lambda", type=float, default=0.55)
    parser.add_argument("--no_calibrate_offset", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = load_jsonl(args.input)
    if not rows:
        raise SystemExit(f"no rows found: {args.input}")

    pairs, summary = build_partial_id_pairs(
        rows,
        mode=args.mode,
        zeta0=args.zeta0,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        eta_scale=args.eta_scale,
        min_jaccard=args.min_jaccard,
        max_disagreement=args.max_disagreement,
        generic_lambda=args.generic_lambda,
        calibrate_offset=not args.no_calibrate_offset,
    )

    if not pairs:
        raise SystemExit("partial-ID builder produced 0 rows; relax purity/zeta settings")

    dump_jsonl(pairs, args.output)
    summary_path = args.summary or f"{args.output}.summary.json"
    dump_json(summary, summary_path)
    print(f"[ok] wrote {len(pairs)} IRIS partial-ID pairs to {args.output}")
    print(f"[ok] summary: {summary_path}")


if __name__ == "__main__":
    main()
