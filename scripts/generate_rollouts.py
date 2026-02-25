#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliastep_common import dump_json, dump_jsonl, load_jsonl, set_seed


def maybe_discover_model(server: str, timeout: float = 2.0) -> str | None:
    url = server.rstrip("/") + "/v1/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        return None

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if isinstance(first, dict):
        mid = first.get("id")
        if isinstance(mid, str) and mid.strip():
            return mid
    return None


def _resolve_device(torch_mod: Any, requested: str) -> str:
    req = requested.strip().lower()
    if req in {"cuda", "gpu"}:
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    if req in {"cpu"}:
        return "cpu"
    # auto
    return "cuda" if torch_mod.cuda.is_available() else "cpu"


def _resolve_dtype(torch_mod: Any, device: str, requested: str) -> Any:
    req = requested.strip().lower()
    if req == "float32":
        return torch_mod.float32
    if req == "float16":
        return torch_mod.float16
    if req == "bfloat16":
        return torch_mod.bfloat16
    # auto
    if device == "cuda":
        if hasattr(torch_mod.cuda, "is_bf16_supported") and torch_mod.cuda.is_bf16_supported():
            return torch_mod.bfloat16
        return torch_mod.float16
    return torch_mod.float32


def _chat_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You solve math problems step by step. End with `Final answer: ...`.",
        },
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return (
        "System: You solve math problems step by step. End with `Final answer: ...`.\n"
        f"User: {question}\nAssistant:"
    )


class LocalHFGenerator:
    def __init__(self, model_name: str, device: str, dtype: str):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"local_hf backend requires torch+transformers: {type(e).__name__}: {e}") from e

        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.dtype = _resolve_dtype(torch, self.device, dtype)
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        if self.device == "cuda":
            self.model.to("cuda")

        print(
            f"[info] local_hf loaded model={model_name} device={self.device} dtype={str(self.dtype).replace('torch.', '')}",
            flush=True,
        )

    def complete(self, question: str, max_new_tokens: int, seed: int) -> str:
        prompt = _chat_prompt(self.tokenizer, question)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        if seed >= 0:
            # Keep deterministic behavior without passing `generator`, because
            # some trust_remote_code models reject that kwarg in generate().
            self.torch.manual_seed(seed)
            if self.device == "cuda":
                self.torch.cuda.manual_seed_all(seed)

        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        if out is None or len(out) == 0:
            raise ValueError("empty generation output")
        prompt_len = int(inputs["input_ids"].shape[-1])
        seq = out[0]
        if seq is None or len(seq) == 0:
            raise ValueError("empty generation sequence")
        gen_ids = seq[prompt_len:] if len(seq) > prompt_len else seq
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not text:
            text = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        return text


def request_completion(server: str, model: str, question: str, max_new_tokens: int, seed: int) -> str:
    url = server.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You solve math problems step by step. End with `Final answer: ...`.",
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
        "max_tokens": max_new_tokens,
        "seed": seed,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30.0) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("empty choices")
    first = choices[0]
    msg = first.get("message", {}) if isinstance(first, dict) else {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty content")
    return content.strip()


def synthetic_rollout(problem: dict, rollout_id: int, rng: random.Random) -> tuple[str, list[str], str, bool]:
    gold = str(problem.get("answer", "0"))
    question = str(problem.get("question", ""))

    # Default: rollout 0 is mostly correct; others carry controlled errors.
    should_be_correct = rollout_id == 0 or rng.random() < 0.35

    steps = [
        "Identify known quantities from the question.",
        "Set up the required arithmetic transformation.",
        "Compute the intermediate value carefully.",
    ]

    final = gold
    if not should_be_correct:
        try:
            final = str(int(gold) + (1 if rollout_id % 2 == 0 else -1))
        except ValueError:
            final = gold + " (approx)"
        steps[2] = "Compute the intermediate value quickly (possible arithmetic slip)."

    steps.append(f"Final answer: {final}")
    text = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
    is_correct = str(final).strip() == str(gold).strip()

    # Add a tiny question-dependent lexical variation.
    if "integer quotient" in question.lower() and rng.random() < 0.2:
        text += "\nCheck integer floor behavior before concluding."
        steps.insert(2, "Double-check quotient semantics for integer division.")

    return text, steps, final, is_correct


def split_steps(text: str) -> list[str]:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        return [text.strip()] if text.strip() else []
    steps = []
    for line in lines:
        if ":" in line and line.lower().startswith("step"):
            _, rhs = line.split(":", 1)
            steps.append(rhs.strip())
        else:
            steps.append(line)
    return steps


def extract_final_answer(text: str) -> str:
    marker = "final answer:"
    lower = text.lower()
    idx = lower.rfind(marker)
    if idx >= 0:
        tail = text[idx + len(marker) :].strip()
        if tail:
            lines = [x.strip() for x in tail.splitlines() if x.strip()]
            if lines:
                return lines[0]
    # fallback: last non-empty line
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return lines[-1] if lines else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate on-policy rollouts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--n_rollouts", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--backend", choices=["auto", "server", "local_hf", "synthetic"], default="auto")
    parser.add_argument("--server", default="")
    parser.add_argument("--local_model", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    problems = load_jsonl(args.input)
    if not problems:
        raise SystemExit(f"no input rows: {args.input}")

    model_id = None
    if args.server:
        model_id = maybe_discover_model(args.server)

    backend = args.backend
    backend_from_auto = backend == "auto"
    if backend == "auto":
        if args.server and model_id:
            backend = "server"
        elif args.local_model.strip():
            backend = "local_hf"
        else:
            backend = "synthetic"

    local_model = args.local_model.strip()
    local_gen: LocalHFGenerator | None = None
    if backend == "local_hf":
        if not local_model:
            raise SystemExit("backend=local_hf requires --local_model")
        try:
            local_gen = LocalHFGenerator(model_name=local_model, device=args.device, dtype=args.dtype)
        except Exception as e:  # noqa: BLE001
            if backend_from_auto:
                print(f"[warn] local_hf init failed in auto mode, fallback synthetic: {e}", flush=True)
                backend = "synthetic"
                local_gen = None
            else:
                raise SystemExit(f"failed to init local_hf backend: {e}") from e

    print(
        f"[info] rollout backend={backend} server={args.server or '-'} model={model_id or local_model or 'synthetic'}",
        flush=True,
    )

    rows: list[dict] = []
    source_counts = defaultdict(int)
    fallback_counts = defaultdict(int)
    local_warn_limit = 12

    for pi, p in enumerate(problems):
        q = str(p.get("question", ""))
        gold = str(p.get("answer", ""))
        for rid in range(max(1, args.n_rollouts)):
            text = ""
            steps: list[str] = []
            final = ""
            correct = False
            source = "synthetic_policy"

            local_seed = args.seed + pi * 31 + rid

            if backend == "server" and model_id and args.server:
                try:
                    text = request_completion(
                        server=args.server,
                        model=model_id,
                        question=q,
                        max_new_tokens=args.max_new_tokens,
                        seed=local_seed,
                    )
                    steps = split_steps(text)
                    final = extract_final_answer(text)
                    correct = str(final).strip() == gold.strip()
                    source = model_id
                except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
                    fallback_counts["server_to_synth"] += 1
                    print(f"[warn] server completion failed, fallback synthetic: {type(e).__name__}", flush=True)
                    text, steps, final, correct = synthetic_rollout(p, rid, rng)
            elif backend == "local_hf" and local_gen is not None:
                try:
                    text = local_gen.complete(
                        question=q,
                        max_new_tokens=args.max_new_tokens,
                        seed=local_seed,
                    )
                    steps = split_steps(text)
                    final = extract_final_answer(text)
                    correct = str(final).strip() == gold.strip()
                    source = f"local_hf::{local_model}"
                except Exception as e:  # noqa: BLE001
                    fallback_counts["local_to_synth"] += 1
                    if fallback_counts["local_to_synth"] <= local_warn_limit:
                        print(f"[warn] local_hf failed, fallback synthetic: {type(e).__name__}: {e}", flush=True)
                    elif fallback_counts["local_to_synth"] == local_warn_limit + 1:
                        print("[warn] further local_hf fallback logs are suppressed", flush=True)
                    text, steps, final, correct = synthetic_rollout(p, rid, rng)
            else:
                text, steps, final, correct = synthetic_rollout(p, rid, rng)

            row = {
                "problem_id": p.get("id", f"row_{pi}"),
                "rollout_id": rid,
                "question": q,
                "gold_answer": gold,
                "final_answer": final,
                "is_correct": int(bool(correct)),
                "difficulty": p.get("difficulty", "medium"),
                "source_model": source,
                "response": text,
                "steps": steps,
            }
            rows.append(row)
            source_counts[source] += 1

    dump_jsonl(rows, args.output)
    dump_json(
        {
            "input": args.input,
            "output": args.output,
            "n_rows": len(rows),
            "n_problems": len(problems),
            "n_rollouts_per_problem": max(1, args.n_rollouts),
            "backend": backend,
            "source_counts": dict(source_counts),
            "fallback_counts": dict(fallback_counts),
        },
        f"{args.output}.meta.json",
    )
    print(f"[ok] wrote rollouts: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
