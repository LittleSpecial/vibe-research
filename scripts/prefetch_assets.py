#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download


@dataclass
class PrefetchItem:
    repo_id: str
    repo_type: str
    endpoint: str
    status: str
    duration_sec: float
    local_path: str
    error: str = ""


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def try_prefetch(repo_id: str, repo_type: str, endpoints: List[str], cache_dir: str, workers: int) -> PrefetchItem:
    last_error = ""
    for ep in endpoints:
        os.environ["HF_ENDPOINT"] = ep
        t0 = time.time()
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                cache_dir=cache_dir,
                max_workers=workers,
            )
            return PrefetchItem(
                repo_id=repo_id,
                repo_type=repo_type,
                endpoint=ep,
                status="ok",
                duration_sec=time.time() - t0,
                local_path=local_path,
            )
        except Exception as e:  # noqa: BLE001
            last_error = f"{type(e).__name__}: {e}"
            print(f"[warn] {repo_type}:{repo_id} via {ep} failed: {last_error}", flush=True)

    return PrefetchItem(
        repo_id=repo_id,
        repo_type=repo_type,
        endpoint=endpoints[-1] if endpoints else "",
        status="failed",
        duration_sec=0.0,
        local_path="",
        error=last_error,
    )


def materialize_local_dataset(output_root: str, seed: int) -> None:
    cmd = (
        "python scripts/download_datasets.py "
        "--math_train_size 8000 "
        "--math_difficulty medium,hard "
        "--eval_sets MATH500,GSM8K-hard "
        "--include_prm800k_heldout "
        f"--seed {seed} "
        f"--output_root {output_root}"
    )
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"dataset materialization failed: rc={rc}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prefetch model/dataset assets with CN mirror first.")
    p.add_argument("--hf-home", default=".cache/huggingface")
    p.add_argument("--endpoint-primary", default="https://hf-mirror.com")
    p.add_argument("--endpoint-fallback", default="https://huggingface.co")
    p.add_argument(
        "--models",
        default="Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct",
    )
    p.add_argument(
        "--datasets",
        default="hendrycks/competition_math,openai/gsm8k,tasksource/PRM800K,mbpp,openai_humaneval",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--materialize-local-data", action="store_true")
    p.add_argument("--local-data-dir", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary-out", default="runs/.prefetch/summary.json")
    args = p.parse_args()

    hf_home = Path(args.hf_home)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home.resolve())

    endpoints = []
    for ep in [args.endpoint_primary, args.endpoint_fallback]:
        ep = ep.strip()
        if ep and ep not in endpoints:
            endpoints.append(ep)

    out_items: List[PrefetchItem] = []

    for model in parse_csv(args.models):
        print(f"[info] prefetch model: {model}", flush=True)
        item = try_prefetch(
            repo_id=model,
            repo_type="model",
            endpoints=endpoints,
            cache_dir=str(hf_home),
            workers=max(1, args.workers),
        )
        out_items.append(item)
        print(
            f"[info] model {model} -> {item.status} via {item.endpoint} ({item.duration_sec:.1f}s)",
            flush=True,
        )

    for ds in parse_csv(args.datasets):
        print(f"[info] prefetch dataset: {ds}", flush=True)
        item = try_prefetch(
            repo_id=ds,
            repo_type="dataset",
            endpoints=endpoints,
            cache_dir=str(hf_home),
            workers=max(1, args.workers),
        )
        out_items.append(item)
        print(
            f"[info] dataset {ds} -> {item.status} via {item.endpoint} ({item.duration_sec:.1f}s)",
            flush=True,
        )

    local_data_status = "skipped"
    if args.materialize_local_data:
        print("[info] materializing local data/ from project script", flush=True)
        materialize_local_dataset(output_root=args.local_data_dir, seed=args.seed)
        local_data_status = "ok"

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_home": str(hf_home.resolve()),
        "endpoints": endpoints,
        "items": [asdict(x) for x in out_items],
        "local_data_status": local_data_status,
        "ok_count": sum(1 for x in out_items if x.status == "ok"),
        "fail_count": sum(1 for x in out_items if x.status != "ok"),
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] summary: {out}", flush=True)
    print(f"[ok] done: ok={summary['ok_count']} fail={summary['fail_count']}", flush=True)


if __name__ == "__main__":
    main()
