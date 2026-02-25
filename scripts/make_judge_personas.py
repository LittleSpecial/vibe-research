#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_personas(anchor: str, llm_personas: int) -> dict:
    personas = [
        {
            "id": f"{anchor}_anchor",
            "kind": "anchor",
            "model": anchor,
            "reliability": 0.95,
            "bias": {"length": 0.0, "position": 0.0, "repetition": 0.0, "source": 0.0},
        }
    ]

    bias_bank = [
        {"length": 0.20, "position": 0.05, "repetition": -0.10, "source": 0.10},
        {"length": -0.15, "position": -0.10, "repetition": 0.20, "source": 0.05},
        {"length": 0.05, "position": 0.25, "repetition": -0.05, "source": -0.10},
        {"length": -0.25, "position": 0.15, "repetition": 0.10, "source": 0.15},
    ]

    for i in range(max(0, llm_personas)):
        b = bias_bank[i % len(bias_bank)]
        personas.append(
            {
                "id": f"judge_persona_{i + 1}",
                "kind": "llm",
                "model": ["Qwen", "Llama", "Mistral", "Mixtral"][i % 4],
                "reliability": round(0.62 + 0.08 * (i % 3), 3),
                "bias": b,
            }
        )

    return {"anchor": anchor, "personas": personas}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-annotator judge personas config.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor", default="verifier")
    parser.add_argument("--llm_personas", type=int, default=3)
    args = parser.parse_args()

    payload = build_personas(anchor=args.anchor, llm_personas=args.llm_personas)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML subset, so downstream YAML parsers can still read it.
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote judge personas: {out}")


if __name__ == "__main__":
    main()
