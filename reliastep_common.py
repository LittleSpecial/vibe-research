from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path, default: Any | None = None) -> Any:
    p = Path(path)
    if not p.exists():
        return {} if default is None else default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {} if default is None else default


def dump_json(data: Any, path: str | Path) -> None:
    p = ensure_parent(path)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def dump_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def try_load_table(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []

    if p.suffix.lower() in {".json", ".jsonl", ".parquet"}:
        rows = load_jsonl(p)
        if rows:
            return rows
        data = load_json(p, default=[])
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

    if p.suffix.lower() == ".csv":
        import csv

        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(r) for r in reader]

    return []


def try_dump_table(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = ensure_parent(path)
    # Keep dependency-free behavior: write JSONL regardless of extension.
    # Consumer scripts read JSONL from this path via `try_load_table`.
    dump_jsonl(rows, p)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softplus(x: float) -> float:
    if x > 20:
        return x
    return math.log1p(math.exp(x))


def parse_key_value_csv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid key=value item: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def stable_hash_float(text: str) -> float:
    # Small deterministic pseudo-random scalar in [0, 1).
    h = 0
    for ch in text:
        h = (h * 131 + ord(ch)) % 1_000_000_007
    return (h % 10_000) / 10_000.0


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def repetition_score(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    uniq = len(set(words))
    return clamp(1.0 - (uniq / max(1, len(words))), 0.0, 1.0)


def guess_source_id(text: str) -> int:
    key = text.lower()
    if "qwen" in key:
        return 1
    if "llama" in key:
        return 2
    if "mistral" in key:
        return 3
    if "verifier" in key or "anchor" in key:
        return 0
    return 4


def difficulty_to_level(d: str) -> int:
    d = (d or "").strip().lower()
    if d in {"easy", "l1"}:
        return 1
    if d in {"medium", "l2"}:
        return 2
    if d in {"hard", "l3"}:
        return 3
    return 2


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def variance(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def std(xs: list[float]) -> float:
    return math.sqrt(max(0.0, variance(xs)))


def pearson_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    if denx <= 1e-12 or deny <= 1e-12:
        return 0.0
    return clamp(num / (denx * deny), -1.0, 1.0)


def rankdata(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_corr(rx, ry)


def compute_binary_calibration(target: list[int], prob: list[float], bins: int = 10) -> dict[str, float]:
    if not target or len(target) != len(prob):
        return {"ece": 0.0, "brier": 0.0}

    brier = mean([(p - float(y)) ** 2 for y, p in zip(target, prob)])
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        idx = [i for i, p in enumerate(prob) if (lo <= p < hi) or (b == bins - 1 and p == hi)]
        if not idx:
            continue
        acc = mean([float(target[i]) for i in idx])
        conf = mean([prob[i] for i in idx])
        ece += abs(acc - conf) * (len(idx) / len(prob))
    return {"ece": clamp(ece, 0.0, 1.0), "brier": clamp(brier, 0.0, 1.0)}


def compute_auc(target: list[int], score: list[float]) -> float:
    pos = [(s, t) for s, t in zip(score, target) if t == 1]
    neg = [(s, t) for s, t in zip(score, target) if t == 0]
    if not pos or not neg:
        return 0.5

    better = 0.0
    total = float(len(pos) * len(neg))
    for sp, _ in pos:
        for sn, _ in neg:
            if sp > sn:
                better += 1.0
            elif sp == sn:
                better += 0.5
    return clamp(better / total, 0.0, 1.0)


def parse_int_list(text: str) -> list[float]:
    out: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out


def add_standard_seed_arg(parser: argparse.ArgumentParser, default: int = 42) -> None:
    parser.add_argument("--seed", type=int, default=default)


def now_ts() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()
