from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import re

from .config import Settings
from .llm_client import ResponsesClient


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:48] or "run"


def _read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class ResearchCycleRunner:
    def __init__(self, repo_root: Path, settings: Settings):
        self.repo_root = repo_root
        self.settings = settings

        provider = settings.provider()
        model = settings.model
        self.client = ResponsesClient(
            base_url=provider["base_url"],
            model=model["model"],
            reasoning_effort=model.get("model_reasoning_effort", "high"),
        )

    def run_cycle(self, topic: str, dry_run: bool = False) -> Path:
        max_gpu_hours = float(self.settings.research.get("max_gpu_hours_per_run", 12))
        if max_gpu_hours <= 0:
            raise ValueError("max_gpu_hours_per_run must be > 0")

        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(topic)}"
        run_dir = self.repo_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            (run_dir / "DRY_RUN.txt").write_text("dry run only\n", encoding="utf-8")
            return run_dir

        idea_prompt = _read_prompt(self.repo_root / "prompts" / "ideation.md")
        idea_md = self.client.complete(
            system_prompt="You are a careful and pragmatic ML research assistant.",
            user_prompt=f"{idea_prompt}\n\nTopic seed: {topic}",
        )
        (run_dir / "idea.md").write_text(idea_md, encoding="utf-8")

        planning_prompt = _read_prompt(self.repo_root / "prompts" / "planning.md")
        plan_raw = self.client.complete(
            system_prompt="Return valid JSON only. No markdown wrappers.",
            user_prompt=f"{planning_prompt}\n\nIdea:\n{idea_md}",
        )
        plan = json.loads(plan_raw)
        total_gpu = sum(float(x.get("est_gpu_hours", 0.0)) for x in plan)
        if total_gpu > max_gpu_hours:
            raise RuntimeError(
                f"Plan exceeds max_gpu_hours_per_run: {total_gpu:.2f} > {max_gpu_hours:.2f}"
            )
        (run_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        impl_prompt = _read_prompt(self.repo_root / "prompts" / "implementation.md")
        impl_doc = self.client.complete(
            system_prompt="Be concrete. Prefer short, executable outputs.",
            user_prompt=f"{impl_prompt}\n\nPlan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}",
        )
        (run_dir / "implementation.md").write_text(impl_doc, encoding="utf-8")

        meta = {
            "run_id": run_id,
            "topic": topic,
            "max_gpu_hours_per_run": max_gpu_hours,
            "status": "planned",
            "created_at": datetime.now().isoformat(),
        }
        (run_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return run_dir
