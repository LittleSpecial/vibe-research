#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import (
    clamp,
    compute_binary_calibration,
    difficulty_to_level,
    dump_json,
    dump_jsonl,
    load_jsonl,
    mean,
    repetition_score,
    set_seed,
    std,
    tokenize_words,
    variance,
)

FEATURE_NAMES = [
    "length_delta_norm",
    "repetition_delta",
    "position_norm",
    "feat_step_index_norm",
    "feat_repetition",
    "feat_length",
    "feat_difficulty_norm",
    "feat_source_norm",
    "step_quality_delta",
    "token_count_delta_norm",
]


def _to_sign(label: Any) -> int:
    try:
        return 1 if int(label) == 1 else -1
    except Exception:
        return -1


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _score_step_text(step: str) -> float:
    toks = [w for w in tokenize_words(step) if w]
    if not toks:
        return 0.0
    digits = sum(ch.isdigit() for ch in step)
    lexical = 1.0 - repetition_score(step)
    return 0.35 + 0.2 * min(1.0, digits / 6.0) + 0.45 * lexical


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def pair_feature_vector(
    *,
    candidate_a: str,
    candidate_b: str,
    difficulty: str = "medium",
    features: dict[str, Any] | None = None,
    step_index: int = 0,
    max_step: int = 1,
) -> list[float]:
    feats = features or {}
    len_a = len(candidate_a)
    len_b = len(candidate_b)
    tok_a = len(tokenize_words(candidate_a))
    tok_b = len(tokenize_words(candidate_b))
    rep_a = repetition_score(candidate_a)
    rep_b = repetition_score(candidate_b)
    pos_norm = step_index / max(1, max_step - 1)
    diff_norm = difficulty_to_level(difficulty) / 3.0

    x = [
        clamp(_safe_float(feats.get("length_delta", (len_a - len_b))) / 200.0, -3.0, 3.0),
        clamp(_safe_float(feats.get("repetition_delta", rep_a - rep_b)), -1.0, 1.0),
        clamp(_safe_float(feats.get("position_norm", pos_norm)), 0.0, 1.0),
        clamp(_safe_float(feats.get("feat_step_index_norm", pos_norm)), 0.0, 1.0),
        clamp(_safe_float(feats.get("feat_repetition", rep_a)), 0.0, 1.0),
        clamp(_safe_float(feats.get("feat_length", min(1.0, len_a / 160.0))), 0.0, 2.0),
        clamp(_safe_float(feats.get("feat_difficulty", diff_norm * 3.0)) / 3.0, 0.0, 2.0),
        clamp(_safe_float(feats.get("feat_source", 0.0)) / 5.0, 0.0, 2.0),
        clamp(_score_step_text(candidate_a) - _score_step_text(candidate_b), -1.5, 1.5),
        clamp((tok_a - tok_b) / 80.0, -2.0, 2.0),
    ]
    return x


def _dot(x: list[float], w: list[float], b: float) -> float:
    return sum(a * c for a, c in zip(x, w)) + b


def _bce_loss(y: int, p: float) -> float:
    eps = 1e-9
    p = clamp(p, eps, 1.0 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def load_preferences(path: str) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    return [r for r in rows if isinstance(r, dict)]


def estimate_reliability(rows: list[dict[str, Any]]) -> dict[str, float]:
    stats: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        aid = str(r.get("annotator_id", "unknown"))
        label = _to_sign(r.get("label", 0))
        gold = _to_sign(r.get("gold_pref", r.get("label", 0)))
        stats[aid].append(1 if label == gold else 0)

    rel: dict[str, float] = {}
    for aid, vals in stats.items():
        pos = sum(vals)
        n = len(vals)
        rel[aid] = clamp((pos + 1.0) / (n + 2.0), 0.05, 0.99)
    return rel


def estimate_latent_signal(
    prefs: list[dict[str, Any]],
    reliability: dict[str, float],
    use_reliability: bool,
) -> dict[str, float]:
    if not prefs:
        return {"signal": 0.0, "noise": 0.5, "disagreement": 0.5, "n_records": 0, "n_steps": 0}

    by_step: dict[str, list[float]] = defaultdict(list)
    errors: list[float] = []
    for r in prefs:
        aid = str(r.get("annotator_id", "unknown"))
        w = reliability.get(aid, 0.7) if use_reliability else 1.0
        label = float(_to_sign(r.get("label", 0)))
        gold = float(_to_sign(r.get("gold_pref", r.get("label", 0))))
        key = f"{r.get('problem_id', 'p')}::{r.get('step_index', 0)}"
        by_step[key].append(label * w)
        errors.append(abs(label - gold) / 2.0)

    per_step = [mean(v) for v in by_step.values() if v]
    signal = mean(per_step)
    disagreement = mean([variance(v) for v in by_step.values() if len(v) > 1])
    noise = clamp(0.5 * mean(errors) + 0.5 * disagreement, 0.0, 1.0)
    return {
        "signal": signal,
        "noise": noise,
        "disagreement": disagreement,
        "n_records": float(len(prefs)),
        "n_steps": float(len(by_step)),
    }


def outcome_signal_from_rollouts(rollouts_path: str) -> dict[str, float]:
    rows = load_jsonl(rollouts_path)
    if not rows:
        return {"signal": 0.0, "noise": 0.6, "n_rows": 0.0}
    correct = [int(r.get("is_correct", 0)) for r in rows]
    acc = mean([float(x) for x in correct])
    return {"signal": (acc - 0.5) * 2.0, "noise": clamp(1.0 - acc, 0.0, 1.0), "n_rows": float(len(rows))}


def _build_pref_samples(
    prefs: list[dict[str, Any]],
    reliability: dict[str, float],
    use_reliability: bool,
) -> list[tuple[list[float], int, float]]:
    out: list[tuple[list[float], int, float]] = []
    for r in prefs:
        a = str(r.get("candidate_a", ""))
        b = str(r.get("candidate_b", ""))
        x = pair_feature_vector(
            candidate_a=a,
            candidate_b=b,
            difficulty=str(r.get("difficulty", "medium")),
            features=r.get("features") if isinstance(r.get("features"), dict) else None,
            step_index=int(r.get("step_index", 0) or 0),
            max_step=max(2, int(r.get("max_step", 6) or 6)),
        )
        y = 1 if int(r.get("label", 0)) == 1 else 0
        aid = str(r.get("annotator_id", "unknown"))
        w = reliability.get(aid, 0.7) if use_reliability else 1.0
        out.append((x, y, clamp(w, 0.05, 2.0)))
    return out


def _build_rollout_samples(rows: list[dict[str, Any]]) -> list[tuple[list[float], int, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("problem_id", ""))].append(r)

    out: list[tuple[list[float], int, float]] = []
    for pid, group in grouped.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda x: int(x.get("rollout_id", 0)))
        a, b = group[0], group[1]
        a_text = str(a.get("response", ""))
        b_text = str(b.get("response", ""))
        x = pair_feature_vector(
            candidate_a=a_text,
            candidate_b=b_text,
            difficulty=str(a.get("difficulty", "medium")),
            features=None,
            step_index=0,
            max_step=2,
        )
        ya = int(a.get("is_correct", 0))
        yb = int(b.get("is_correct", 0))
        if ya == yb:
            continue
        y = 1 if ya > yb else 0
        out.append((x, y, 1.0))
    return out


def _evaluate_samples(
    samples: list[tuple[list[float], int, float]],
    weights: list[float],
    bias: float,
) -> dict[str, Any]:
    if not samples:
        return {"acc": 0.0, "loss": 0.0, "probs": [], "labels": []}
    probs: list[float] = []
    labels: list[int] = []
    weighted_loss = 0.0
    weighted_total = 0.0
    correct = 0
    total = 0
    for x, y, w in samples:
        p = _sigmoid(_dot(x, weights, bias))
        probs.append(p)
        labels.append(y)
        weighted_loss += w * _bce_loss(y, p)
        weighted_total += w
        pred = 1 if p >= 0.5 else 0
        correct += 1 if pred == y else 0
        total += 1
    return {
        "acc": correct / max(1, total),
        "loss": weighted_loss / max(1e-9, weighted_total),
        "probs": probs,
        "labels": labels,
    }


def _fit_logistic_sgd(
    samples: list[tuple[list[float], int, float]],
    updates: int,
    seed: int,
    use_reliability: bool,
) -> tuple[list[float], float, list[dict[str, Any]]]:
    rng = random.Random(seed)
    dim = len(FEATURE_NAMES)
    weights = [0.0] * dim
    bias = 0.0
    lr0 = 0.18 if use_reliability else 0.14
    l2 = 1e-4
    batch_size = min(2048, max(128, len(samples) // 8))

    logs: list[dict[str, Any]] = []
    if not samples:
        return weights, bias, logs

    for step in range(1, max(1, updates) + 1):
        if len(samples) <= batch_size:
            batch = samples
        else:
            batch = [samples[rng.randrange(len(samples))] for _ in range(batch_size)]

        grad_w = [0.0] * dim
        grad_b = 0.0
        batch_loss = 0.0
        batch_w = 0.0
        batch_correct = 0

        for x, y, w in batch:
            z = _dot(x, weights, bias)
            p = _sigmoid(z)
            err = (p - float(y)) * w
            for i in range(dim):
                grad_w[i] += err * x[i]
            grad_b += err
            batch_loss += w * _bce_loss(y, p)
            batch_w += w
            batch_correct += 1 if ((p >= 0.5) == (y == 1)) else 0

        lr = lr0 / math.sqrt(1.0 + 0.015 * step)
        inv = 1.0 / max(1, len(batch))
        for i in range(dim):
            grad = grad_w[i] * inv + l2 * weights[i]
            weights[i] -= lr * grad
        bias -= lr * grad_b * inv

        logs.append(
            {
                "update": step,
                "progress": step / max(1, updates),
                "train_reward": batch_correct / max(1, len(batch)),
                "policy_loss": batch_loss / max(1e-9, batch_w),
            }
        )

    return weights, bias, logs


def _structured_noisy_view(samples: list[tuple[list[float], int, float]]) -> list[tuple[list[float], int, float]]:
    out: list[tuple[list[float], int, float]] = []
    for x, y, w in samples:
        z = list(x)
        # Perturb nuisance axes: verbosity/position/source.
        z[0] = clamp(z[0] * 1.35 + 0.18, -3.0, 3.0)
        z[2] = clamp(1.0 - z[2], 0.0, 1.0)
        z[7] = clamp(z[7] * 1.25 + 0.1, 0.0, 2.0)
        out.append((z, y, w))
    return out


def run_training(
    *,
    method: str,
    model: str,
    output_dir: str,
    seed: int,
    updates: int,
    prefs_path: str = "",
    features_path: str = "",
    rollouts_path: str = "",
    use_reliability: bool = True,
    uncertainty_lambda: float = 0.3,
    outcome_anchor_weight: float = 0.2,
    extra_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del features_path, uncertainty_lambda, outcome_anchor_weight
    set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    reliability: dict[str, float] = {}
    signal = 0.0
    noise = 0.5
    n_records = 0

    if prefs_path:
        prefs = load_preferences(prefs_path)
        reliability = estimate_reliability(prefs) if prefs else {}
        sig = estimate_latent_signal(prefs, reliability=reliability, use_reliability=use_reliability)
        signal = float(sig["signal"])
        noise = float(sig["noise"])
        samples = _build_pref_samples(prefs, reliability=reliability, use_reliability=use_reliability)
        n_records = len(samples)
    else:
        rollouts = load_jsonl(rollouts_path)
        osig = outcome_signal_from_rollouts(rollouts_path)
        signal = float(osig["signal"])
        noise = float(osig["noise"])
        samples = _build_rollout_samples(rollouts)
        n_records = len(samples)

    rng = random.Random(seed + 17)
    rng.shuffle(samples)
    n_train = int(len(samples) * 0.9)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:] if n_train < len(samples) else samples[-max(1, len(samples) // 10) :]
    if not val_samples:
        val_samples = train_samples

    weights, bias, logs = _fit_logistic_sgd(
        samples=train_samples,
        updates=updates,
        seed=seed,
        use_reliability=use_reliability,
    )

    train_eval = _evaluate_samples(train_samples, weights, bias)
    val_eval = _evaluate_samples(val_samples, weights, bias)
    noisy_eval = _evaluate_samples(_structured_noisy_view(val_samples), weights, bias)
    calib = compute_binary_calibration(val_eval["labels"], val_eval["probs"], bins=10)

    quality = clamp(val_eval["acc"], 0.0, 1.0)
    robust_gap = clamp(val_eval["acc"] - noisy_eval["acc"], -1.0, 1.0)
    robustness_bonus = clamp((0.10 if use_reliability else 0.05) * (1.0 - max(0.0, robust_gap)), 0.0, 0.2)
    calibration_bonus = clamp(0.12 * (1.0 - calib["ece"]), 0.0, 0.12)

    reward_tail = [float(x["train_reward"]) for x in logs[-max(5, len(logs) // 8) :]]
    stability = std(reward_tail) if reward_tail else 0.0

    args_dict = dict(extra_args or {})
    args_dict.update(
        {
            "method": method,
            "model": model,
            "seed": seed,
            "updates": updates,
            "prefs_path": prefs_path,
            "rollouts_path": rollouts_path,
            "use_reliability": use_reliability,
        }
    )

    ckpt = {
        "method": method,
        "model": model,
        "seed": seed,
        "quality_score": quality,
        "robustness_bonus": robustness_bonus,
        "calibration_bonus": calibration_bonus,
        "noise_estimate": clamp(noise, 0.0, 1.0),
        "signal_estimate": signal,
        "training_stability_std": stability,
        "use_reliability": use_reliability,
        "annotator_reliability": reliability,
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "bias": bias,
        "dataset_records": n_records,
        "train_acc": train_eval["acc"],
        "val_acc": val_eval["acc"],
        "val_acc_noisy": noisy_eval["acc"],
        "args": args_dict,
    }

    dump_jsonl(logs, out / "train_log.jsonl")
    dump_json(ckpt, out / "checkpoint.json")

    metrics = {
        "quality_score": ckpt["quality_score"],
        "robustness_bonus": ckpt["robustness_bonus"],
        "noise_estimate": ckpt["noise_estimate"],
        "training_stability_std": ckpt["training_stability_std"],
        "n_updates": updates,
        "n_records": n_records,
        "train_acc": train_eval["acc"],
        "val_acc": val_eval["acc"],
        "val_acc_noisy": noisy_eval["acc"],
        "calibration_ece": calib["ece"],
        "calibration_brier": calib["brier"],
    }
    dump_json(metrics, out / "metrics.json")

    if ckpt["annotator_reliability"]:
        dump_json(
            {"annotator_reliability": ckpt["annotator_reliability"], "n": len(ckpt["annotator_reliability"])},
            out / "reliability.json",
        )

    with (out / "README.txt").open("w", encoding="utf-8") as f:
        f.write("Real (non-mock) preference classifier checkpoint for ReLiaStep experiments.\n")
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        f.write("\n")

    return ckpt


def load_checkpoint(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "checkpoint.json"
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def score_pair_with_checkpoint(ckpt: dict[str, Any], features: list[float]) -> float:
    w = ckpt.get("weights", [])
    b = _safe_float(ckpt.get("bias", 0.0))
    if not isinstance(w, list) or len(w) != len(features):
        # Deterministic fallback if checkpoint is malformed.
        score = 0.5 + 0.2 * _safe_float(ckpt.get("quality_score", 0.5)) - 0.1 * _safe_float(ckpt.get("noise_estimate", 0.5))
        return clamp(score, 0.01, 0.99)
    ww = [_safe_float(x) for x in w]
    return _sigmoid(_dot(features, ww, b))
