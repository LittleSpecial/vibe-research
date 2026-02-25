#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import clamp, dump_json, dump_jsonl, load_jsonl, mean, set_seed
from train.reliastep_train_lib import FEATURE_NAMES, pair_feature_vector


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _dot(x: list[float], w: list[float], b: float) -> float:
    return sum(a * c for a, c in zip(x, w)) + b


def _bce(y: int, p: float) -> float:
    eps = 1e-9
    pp = clamp(p, eps, 1.0 - eps)
    return -(y * math.log(pp) + (1 - y) * math.log(1 - pp))


def _build_samples(rows: list[dict[str, Any]], target: str) -> list[tuple[list[float], int, float]]:
    out: list[tuple[list[float], int, float]] = []
    for row in rows:
        a = str(row.get("candidate_a", ""))
        b = str(row.get("candidate_b", ""))
        diff = str(row.get("difficulty", "medium"))
        feats = row.get("features") if isinstance(row.get("features"), dict) else {}

        x = pair_feature_vector(
            candidate_a=a,
            candidate_b=b,
            difficulty=diff,
            features=feats,
            step_index=int(row.get("step_index", 0) or 0),
            max_step=max(2, int(row.get("max_step", 6) or 6)),
        )

        if target == "q_l":
            y = 1 if int(row.get("label", 0)) == 1 else 0
            w = _safe_float(row.get("weight", 1.0), 1.0)
        elif target == "q_generic":
            y = 1 if int(row.get("label_generic", 0)) == 1 else 0
            w = _safe_float(row.get("weight_generic", 1.0), 1.0)
        elif target == "raw":
            y = 1 if int(row.get("label_raw", 0)) == 1 else 0
            w = _safe_float(row.get("weight_raw", 1.0), 1.0)
        else:
            raise ValueError(f"unsupported target: {target}")

        out.append((x, y, clamp(w, 0.05, 4.0)))
    return out


def _evaluate(samples: list[tuple[list[float], int, float]], weights: list[float], bias: float) -> dict[str, Any]:
    if not samples:
        return {"acc": 0.0, "loss": 0.0, "margin": 0.0}

    corr = 0
    wloss = 0.0
    wsum = 0.0
    probs: list[float] = []
    labels: list[int] = []

    for x, y, w in samples:
        p = _sigmoid(_dot(x, weights, bias))
        pred = 1 if p >= 0.5 else 0
        corr += 1 if pred == y else 0
        wloss += w * _bce(y, p)
        wsum += w
        probs.append(p)
        labels.append(y)

    margins = [abs(p - 0.5) * 2.0 for p in probs]
    return {
        "acc": corr / max(1, len(samples)),
        "loss": wloss / max(1e-9, wsum),
        "margin": mean(margins),
    }


def _train(
    samples: list[tuple[list[float], int, float]],
    updates: int,
    seed: int,
) -> tuple[list[float], float, list[dict[str, Any]]]:
    dim = len(FEATURE_NAMES)
    w = [0.0] * dim
    b = 0.0

    if not samples:
        return w, b, []

    rng = random.Random(seed)
    lr0 = 0.16
    l2 = 1e-4
    batch_size = min(2048, max(128, len(samples) // 6))

    logs: list[dict[str, Any]] = []
    for step in range(1, max(1, updates) + 1):
        if len(samples) <= batch_size:
            batch = samples
        else:
            batch = [samples[rng.randrange(len(samples))] for _ in range(batch_size)]

        gw = [0.0] * dim
        gb = 0.0
        b_loss = 0.0
        b_w = 0.0
        b_hit = 0

        for x, y, ww in batch:
            p = _sigmoid(_dot(x, w, b))
            err = (p - float(y)) * ww
            for i in range(dim):
                gw[i] += err * x[i]
            gb += err
            b_loss += ww * _bce(y, p)
            b_w += ww
            b_hit += 1 if ((p >= 0.5) == (y == 1)) else 0

        inv = 1.0 / max(1, len(batch))
        lr = lr0 / math.sqrt(1.0 + 0.012 * step)
        for i in range(dim):
            grad = gw[i] * inv + l2 * w[i]
            w[i] -= lr * grad
        b -= lr * gb * inv

        if step <= 5 or step % 10 == 0 or step == updates:
            logs.append(
                {
                    "update": step,
                    "train_reward": b_hit / max(1, len(batch)),
                    "policy_loss": b_loss / max(1e-9, b_w),
                    "lr": lr,
                }
            )

    return w, b, logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IRIS shift-robust pair scorer with q_L / baseline targets.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--target", choices=["q_l", "q_generic", "raw"], default="q_l")
    parser.add_argument("--updates", type=int, default=320)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = load_jsonl(args.pairs)
    if not rows:
        raise SystemExit(f"no rows found in {args.pairs}")

    samples = _build_samples(rows, target=args.target)
    if not samples:
        raise SystemExit("no trainable samples built from input pairs")

    rng = random.Random(args.seed + 37)
    rng.shuffle(samples)
    n_train = int(clamp(args.train_ratio, 0.5, 0.98) * len(samples))
    train_samples = samples[:n_train]
    val_samples = samples[n_train:] if n_train < len(samples) else samples[-max(32, len(samples) // 10) :]
    if not val_samples:
        val_samples = train_samples

    weights, bias, logs = _train(train_samples, updates=args.updates, seed=args.seed)
    train_eval = _evaluate(train_samples, weights, bias)
    val_eval = _evaluate(val_samples, weights, bias)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "method": "iris_partial_id_dpo",
        "target": args.target,
        "seed": args.seed,
        "updates": args.updates,
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "bias": bias,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "train_acc": train_eval["acc"],
        "val_acc": val_eval["acc"],
        "val_margin": val_eval["margin"],
    }

    metrics = {
        "target": args.target,
        "train_acc": train_eval["acc"],
        "train_loss": train_eval["loss"],
        "val_acc": val_eval["acc"],
        "val_loss": val_eval["loss"],
        "val_margin": val_eval["margin"],
        "n_updates": args.updates,
        "n_samples": len(samples),
    }

    dump_json(ckpt, out / "checkpoint.json")
    dump_json(metrics, out / "metrics.json")
    dump_jsonl(logs, out / "train_log.jsonl")

    with (out / "README.txt").open("w", encoding="utf-8") as f:
        f.write("IRIS partial-ID pair scorer checkpoint.\n")
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        f.write("\n")

    print(f"[ok] trained {args.target} model at {out}")
    print(f"[metric] val_acc={val_eval['acc']:.4f} val_margin={val_eval['margin']:.4f}")


if __name__ == "__main__":
    main()
