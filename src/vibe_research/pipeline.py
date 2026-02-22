from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import Settings
from .literature import (
    LiteraturePaper,
    download_pdfs,
    papers_to_digest_markdown,
    save_papers_json,
    search_literature,
)
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
            max_output_tokens=int(model.get("max_output_tokens", 1400)),
        )

    def run_cycle(
        self,
        topic: str,
        dry_run: bool = False,
        interactive: bool = False,
        agent_count: int = 4,
        feedback_timeout: int = 0,
    ) -> Path:
        max_gpu_hours = float(self.settings.research.get("max_gpu_hours_per_run", 12))
        if max_gpu_hours <= 0:
            raise ValueError("max_gpu_hours_per_run must be > 0")
        max_gpus = self._max_gpus_per_run()
        self._validate_api_budget_config()

        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(topic)}"
        run_dir = self.repo_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (self.repo_root / "runs" / "LATEST_RUN").write_text(run_id, encoding="utf-8")

        self._write_feedback_templates(run_dir)
        literature_enabled = self._literature_enabled()
        total_steps = (4 if agent_count > 1 else 3) + (1 if literature_enabled else 0)
        self._write_status(
            run_dir,
            run_id=run_id,
            topic=topic,
            state="running",
            stage="init",
            step=0,
            total_steps=total_steps,
            message="initialized run directory",
        )
        project_dir = self._init_project_layout(run_id=run_id, topic=topic, run_dir=run_dir)
        self._sync_project_artifacts(run_dir=run_dir, project_dir=project_dir, run_id=run_id, topic=topic)

        if dry_run:
            (run_dir / "DRY_RUN.txt").write_text("dry run only\n", encoding="utf-8")
            self._write_status(
                run_dir,
                run_id=run_id,
                topic=topic,
                state="dry_run",
                stage="dry_run",
                step=0,
                total_steps=total_steps,
                message="dry-run completed",
            )
            self._sync_project_artifacts(run_dir=run_dir, project_dir=project_dir, run_id=run_id, topic=topic)
            return run_dir

        try:
            current_step = 0
            literature_context = ""
            literature_digest = ""
            top_papers: list[LiteraturePaper] = []
            if literature_enabled:
                literature_digest, top_papers = self._prepare_literature(topic=topic, run_dir=run_dir)
                current_step += 1
                self._write_status(
                    run_dir,
                    run_id=run_id,
                    topic=topic,
                    state="running",
                    stage="literature",
                    step=current_step,
                    total_steps=total_steps,
                    message="searching papers and synthesizing literature gaps",
                )
                literature_context = self._run_reviewable_stage(
                    run_dir=run_dir,
                    stage="literature",
                    interactive=interactive,
                    feedback_timeout=feedback_timeout,
                    generator=lambda feedback_text: self._generate_literature_review(
                        run_dir=run_dir,
                        topic=topic,
                        digest_markdown=literature_digest,
                        max_gpu_hours=max_gpu_hours,
                        max_gpus=max_gpus,
                        feedback_text=feedback_text,
                    ),
                    output_path=run_dir / "literature" / "review.md",
                )
                self._maybe_archive_literature_pdfs(
                    run_dir=run_dir,
                    papers=top_papers,
                    run_id=run_id,
                )
                self._sync_project_artifacts(run_dir=run_dir, project_dir=project_dir, run_id=run_id, topic=topic)

            debate_notes: list[tuple[str, str]] = []
            if agent_count > 1:
                current_step += 1
                self._write_status(
                    run_dir,
                    run_id=run_id,
                    topic=topic,
                    state="running",
                    stage="ideation_agents",
                    step=current_step,
                    total_steps=total_steps,
                    message="running multi-agent ideation debate",
                )
                debate_notes = self._run_agent_debate(
                    topic=topic,
                    run_dir=run_dir,
                    agent_count=agent_count,
                    max_gpu_hours=max_gpu_hours,
                    max_gpus=max_gpus,
                    literature_context=literature_context,
                )
                self._append_progress(run_dir, f"generated {len(debate_notes)} agent notes")
                self._sync_project_artifacts(run_dir=run_dir, project_dir=project_dir, run_id=run_id, topic=topic)

            current_step += 1
            self._write_status(
                run_dir,
                run_id=run_id,
                topic=topic,
                state="running",
                stage="idea",
                step=current_step,
                total_steps=total_steps,
                message="generating synthesized idea",
            )
            idea_md = self._run_reviewable_stage(
                run_dir=run_dir,
                stage="idea",
                interactive=interactive,
                feedback_timeout=feedback_timeout,
                generator=lambda feedback_text: self._generate_idea(
                    run_dir=run_dir,
                    topic=topic,
                    debate_notes=debate_notes,
                    max_gpu_hours=max_gpu_hours,
                    max_gpus=max_gpus,
                    literature_context=literature_context,
                    feedback_text=feedback_text,
                ),
                output_path=run_dir / "idea.md",
            )
            self._sync_project_artifacts(run_dir=run_dir, project_dir=project_dir, run_id=run_id, topic=topic)

            current_step += 1
            self._write_status(
                run_dir,
                run_id=run_id,
                topic=topic,
                state="running",
                stage="planning",
                step=current_step,
                total_steps=total_steps,
                message="building executable experiment plan",
            )
            plan_raw = self._run_reviewable_stage(
                run_dir=run_dir,
                stage="planning",
                interactive=interactive,
                feedback_timeout=feedback_timeout,
                generator=lambda feedback_text: self._generate_plan_raw(
                    run_dir=run_dir,
                    idea_md=idea_md,
                    max_gpu_hours=max_gpu_hours,
                    max_gpus=max_gpus,
                    literature_context=literature_context,
                    feedback_text=feedback_text,
                ),
                output_path=run_dir / "plan.raw.txt",
            )
            plan = self._parse_plan(plan_raw)
            self._validate_plan_budget(plan=plan, max_gpu_hours=max_gpu_hours, max_gpus=max_gpus)
            (run_dir / "plan.json").write_text(
                json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._sync_project_artifacts(run_dir=run_dir, project_dir=project_dir, run_id=run_id, topic=topic)

            current_step += 1
            self._write_status(
                run_dir,
                run_id=run_id,
                topic=topic,
                state="running",
                stage="implementation",
                step=current_step,
                total_steps=total_steps,
                message="generating runnable implementation",
            )
            impl_doc = self._run_reviewable_stage(
                run_dir=run_dir,
                stage="implementation",
                interactive=interactive,
                feedback_timeout=feedback_timeout,
                generator=lambda feedback_text: self._generate_implementation(
                    run_dir=run_dir,
                    plan=plan,
                    max_gpu_hours=max_gpu_hours,
                    max_gpus=max_gpus,
                    literature_context=literature_context,
                    feedback_text=feedback_text,
                ),
                output_path=run_dir / "implementation.md",
            )
            self._materialize_experiment_script(run_dir=run_dir, implementation_doc=impl_doc)

            meta = {
                "run_id": run_id,
                "topic": topic,
                "max_gpu_hours_per_run": max_gpu_hours,
                "max_gpu_cards_per_run": max_gpus,
                "max_api_usd_per_day": float(self.settings.research.get("max_api_usd_per_day", 0)),
                "project_dir": f"projects/{run_id}",
                "status": "planned",
                "created_at": datetime.now().isoformat(),
            }
            (run_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            self._write_status(
                run_dir,
                run_id=run_id,
                topic=topic,
                state="completed",
                stage="completed",
                step=total_steps,
                total_steps=total_steps,
                message="idea/plan/implementation generated",
            )
            self._sync_project_artifacts(run_dir=run_dir, project_dir=project_dir, run_id=run_id, topic=topic)
            return run_dir
        except Exception as e:
            self._write_status(
                run_dir,
                run_id=run_id,
                topic=topic,
                state="failed",
                stage="failed",
                step=0,
                total_steps=total_steps,
                message=str(e),
            )
            self._sync_project_artifacts(run_dir=run_dir, project_dir=project_dir, run_id=run_id, topic=topic)
            raise

    def _run_agent_debate(
        self,
        topic: str,
        run_dir: Path,
        agent_count: int,
        max_gpu_hours: float,
        max_gpus: int,
        literature_context: str,
    ) -> list[tuple[str, str]]:
        roles = self._agent_roles()
        selected = roles[: max(1, min(agent_count, len(roles)))]

        ideation_prompt = _read_prompt(self.repo_root / "prompts" / "ideation.md")
        feedback_text = self._collect_feedback(run_dir, "ideation_agents")

        out_dir = run_dir / "agents"
        out_dir.mkdir(parents=True, exist_ok=True)

        notes: list[tuple[str, str]] = []
        for idx, (role_name, role_goal) in enumerate(selected, start=1):
            user_prompt = (
                f"{ideation_prompt}\n\n"
                f"Role: {role_name}\n"
                f"Role objective: {role_goal}\n"
                f"Topic seed: {topic}\n"
                f"Compute envelope: up to {max_gpus}xA100; max wall-clock {max_gpu_hours:.1f}h per run.\n"
            )
            if literature_context.strip():
                user_prompt += (
                    "\nLiterature review context (mandatory; cite paper IDs when you claim novelty):\n"
                    f"{literature_context}\n"
                )
            if feedback_text:
                user_prompt += f"\nExtra human feedback:\n{feedback_text}\n"

            note = self._call_llm(
                run_dir=run_dir,
                system_prompt="You are one specialized research agent in a multi-agent discussion.",
                user_prompt=user_prompt,
            )
            notes.append((role_name, note))
            (out_dir / f"{idx:02d}_{role_name}.md").write_text(note, encoding="utf-8")
            self._append_progress(run_dir, f"agent note completed: {role_name}")

        return notes

    def _agent_roles(self) -> list[tuple[str, str]]:
        fallback = [
            (
                "pi_vision_agent",
                "Propose a high-upside core hypothesis and contribution framing that could survive top-tier review.",
            ),
            (
                "methodology_agent",
                "Design technically rigorous RL/LLM methods and identify the key implementation details.",
            ),
            (
                "experiment_engineer_agent",
                "Build an execution-first plan with strong baselines/ablations that fits the configured A100 constraints.",
            ),
            (
                "reviewer_redteam_agent",
                "Act like a strict reviewer: expose weaknesses, confounders, missing controls, and ways to falsify the claim.",
            ),
        ]

        cfg = self.settings.agents
        raw_roles = cfg.get("roles", [])
        if not isinstance(raw_roles, list) or not raw_roles:
            return fallback

        roles: list[tuple[str, str]] = []
        for i, item in enumerate(raw_roles, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", f"agent_{i}")).strip() or f"agent_{i}"
            goal = str(item.get("goal", "")).strip()
            if goal:
                roles.append((name, goal))

        return roles or fallback

    def _generate_idea(
        self,
        run_dir: Path,
        topic: str,
        debate_notes: list[tuple[str, str]],
        max_gpu_hours: float,
        max_gpus: int,
        literature_context: str,
        feedback_text: str,
    ) -> str:
        ideation_prompt = _read_prompt(self.repo_root / "prompts" / "ideation.md")
        notes_blob = "\n\n".join([f"[{name}]\n{text}" for name, text in debate_notes])
        user_prompt = (
            f"{ideation_prompt}\n\n"
            f"Topic seed: {topic}\n"
            f"Compute envelope: up to {max_gpus}xA100; max wall-clock {max_gpu_hours:.1f}h per run.\n"
            f"If using >1 GPU, mention the required parallel setup explicitly.\n"
        )
        if notes_blob:
            user_prompt += f"\nAgent discussion notes:\n{notes_blob}\n"
        if literature_context.strip():
            user_prompt += (
                "\nLiterature review context (must use to avoid trivial/known ideas):\n"
                f"{literature_context}\n"
            )
        if feedback_text:
            user_prompt += f"\nHuman feedback for revision:\n{feedback_text}\n"

        return self._call_llm(
            run_dir=run_dir,
            system_prompt=(
                "You are the lead research agent. Synthesize one strong RL/LLM idea with clear experiments."
            ),
            user_prompt=user_prompt,
        )

    def _generate_plan_raw(
        self,
        run_dir: Path,
        idea_md: str,
        max_gpu_hours: float,
        max_gpus: int,
        literature_context: str,
        feedback_text: str,
    ) -> str:
        planning_prompt = _read_prompt(self.repo_root / "prompts" / "planning.md")
        user_prompt = (
            f"{planning_prompt}\n\n"
            f"Budget constraint:\n"
            f"- Max wall-clock per run: {max_gpu_hours:.1f}h\n"
            f"- Hardware: up to {max_gpus}xA100\n\n"
            f"Idea:\n{idea_md}"
        )
        if literature_context.strip():
            user_prompt += (
                "\n\nLiterature novelty constraints:\n"
                f"{literature_context}\n"
            )
        if feedback_text:
            user_prompt += f"\n\nHuman feedback for revision:\n{feedback_text}\n"

        return self._call_llm(
            run_dir=run_dir,
            system_prompt="Return valid JSON only. No markdown wrappers.",
            user_prompt=user_prompt,
        )

    def _generate_implementation(
        self,
        run_dir: Path,
        plan: list[dict],
        max_gpu_hours: float,
        max_gpus: int,
        literature_context: str,
        feedback_text: str,
    ) -> str:
        impl_prompt = _read_prompt(self.repo_root / "prompts" / "implementation.md")
        user_prompt = (
            f"{impl_prompt}\n\n"
            f"Runtime environment:\n"
            f"- Linux aarch64\n"
            f"- Slurm cluster\n"
            f"- up to {max_gpus}xA100\n"
            f"- wall-clock <= {max_gpu_hours:.1f}h\n\n"
            f"Plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}"
        )
        if literature_context.strip():
            user_prompt += (
                "\n\nContext from literature review:\n"
                f"{literature_context}\n"
            )
        if feedback_text:
            user_prompt += f"\n\nHuman feedback for revision:\n{feedback_text}\n"

        return self._call_llm(
            run_dir=run_dir,
            system_prompt="Be concrete. Prefer short, executable outputs.",
            user_prompt=user_prompt,
        )

    def _literature_enabled(self) -> bool:
        return self._to_bool(self.settings.research.get("enable_literature_search", True))

    def _prepare_literature(self, topic: str, run_dir: Path) -> tuple[str, list[LiteraturePaper]]:
        lit_dir = run_dir / "literature"
        lit_dir.mkdir(parents=True, exist_ok=True)

        max_results = int(self.settings.research.get("literature_max_results", 24))
        top_k = int(self.settings.research.get("literature_top_k", 12))
        sources = self._literature_sources()
        papers = search_literature(topic=topic, max_results=max_results, sources=sources)
        save_papers_json(papers, lit_dir / "papers.json")

        digest = papers_to_digest_markdown(topic=topic, papers=papers, top_k=top_k)
        (lit_dir / "digest.md").write_text(digest, encoding="utf-8")
        source_counts: dict[str, int] = {}
        for p in papers:
            source_counts[p.source] = source_counts.get(p.source, 0) + 1
        source_blob = ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())) or "none"
        self._append_progress(
            run_dir,
            (
                "literature search done: "
                f"{len(papers)} papers retrieved from [{source_blob}] "
                f"(top_k={min(len(papers), top_k)})"
            ),
        )
        return digest, papers[: max(1, top_k)]

    def _generate_literature_review(
        self,
        run_dir: Path,
        topic: str,
        digest_markdown: str,
        max_gpu_hours: float,
        max_gpus: int,
        feedback_text: str,
    ) -> str:
        lit_prompt = _read_prompt(self.repo_root / "prompts" / "literature.md")
        user_prompt = (
            f"{lit_prompt}\n\n"
            f"Topic seed: {topic}\n"
            f"Compute envelope: up to {max_gpus}xA100; max wall-clock {max_gpu_hours:.1f}h per run.\n\n"
            f"Retrieved papers digest:\n{digest_markdown}\n"
        )
        if feedback_text:
            user_prompt += f"\nHuman feedback for revision:\n{feedback_text}\n"

        return self._call_llm(
            run_dir=run_dir,
            system_prompt=(
                "You are a strict top-conference scout. Produce non-trivial novelty gaps with explicit paper references."
            ),
            user_prompt=user_prompt,
        )

    def _maybe_archive_literature_pdfs(
        self,
        run_dir: Path,
        papers: list[LiteraturePaper],
        run_id: str,
    ) -> None:
        if not self._to_bool(self.settings.research.get("literature_download_pdfs", True)):
            return

        lit_dir = run_dir / "literature"
        pdf_dir = lit_dir / "pdfs"
        max_pdf_count = int(self.settings.research.get("literature_pdf_count", 6))
        downloaded = download_pdfs(
            papers=papers,
            out_dir=pdf_dir,
            max_count=max_pdf_count,
        )
        if not downloaded:
            self._append_progress(run_dir, "literature pdf archive skipped: no pdf downloaded")
            return

        self._append_progress(run_dir, f"literature pdf downloaded: {len(downloaded)} files")
        if not self._to_bool(self.settings.research.get("literature_remote_archive", True)):
            return

        host = str(self.settings.remote.get("default_host", "")).strip()
        remote_repo = str(self.settings.remote.get("remote_repo", "")).strip()
        if not host or not remote_repo:
            self._append_progress(run_dir, "literature remote archive skipped: remote host/repo not configured")
            return

        remote_rel = f"runs/{run_id}/literature/pdfs"
        ok = self._sync_dir_to_remote(local_dir=pdf_dir, host=host, remote_repo=remote_repo, remote_rel=remote_rel)
        if ok:
            self._append_progress(run_dir, f"literature pdf archived to remote: {host}:~/zx/{remote_repo}/{remote_rel}")
            if self._to_bool(self.settings.research.get("literature_delete_local_pdfs_after_sync", True)):
                shutil.rmtree(pdf_dir, ignore_errors=True)
                self._append_progress(run_dir, "literature local pdf removed after remote archive")
        else:
            self._append_progress(run_dir, "literature remote archive failed; kept local pdf cache")

    def _sync_dir_to_remote(self, local_dir: Path, host: str, remote_repo: str, remote_rel: str) -> bool:
        remote_prefix = f"{host}:~/zx/{remote_repo}/{remote_rel}/"
        try:
            subprocess.run(
                ["ssh", host, f"mkdir -p ~/zx/{remote_repo}/{remote_rel}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            cp = subprocess.run(
                ["rsync", "-az", str(local_dir) + "/", remote_prefix],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception:
            return False
        return cp.returncode == 0

    def _run_reviewable_stage(
        self,
        run_dir: Path,
        stage: str,
        interactive: bool,
        feedback_timeout: int,
        generator: Callable[[str], str],
        output_path: Path,
    ) -> str:
        revision_feedback = self._collect_feedback(run_dir, stage)
        text = generator(revision_feedback)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        self._append_progress(run_dir, f"stage completed: {stage}")

        if not interactive:
            return text

        while True:
            action, note = self._await_stage_action(run_dir, stage, feedback_timeout)
            if action == "revise":
                revision_feedback = note.strip()
                text = generator(revision_feedback)
                output_path.write_text(text, encoding="utf-8")
                self._append_progress(run_dir, f"stage revised: {stage}")
                continue
            return text

    def _parse_plan(self, plan_raw: str) -> list[dict]:
        try:
            parsed = json.loads(plan_raw)
        except json.JSONDecodeError:
            # best-effort JSON extraction from fenced content
            m = re.search(r"```json\s*(.*?)```", plan_raw, flags=re.DOTALL | re.IGNORECASE)
            if not m:
                raise
            parsed = json.loads(m.group(1).strip())

        if not isinstance(parsed, list):
            raise TypeError("plan is not a JSON array")
        return parsed

    @staticmethod
    def _validate_plan_budget(plan: list[dict], max_gpu_hours: float, max_gpus: int) -> None:
        total_gpu = sum(float(x.get("est_gpu_hours", 0.0)) for x in plan)
        budget = max_gpu_hours * max_gpus
        if total_gpu > budget:
            raise RuntimeError(
                "Plan exceeds total GPU-hours envelope: "
                f"{total_gpu:.2f} > {budget:.2f} ({max_gpus}xA100 * wall-clock budget)"
            )

    def _call_llm(self, run_dir: Path, system_prompt: str, user_prompt: str) -> str:
        self._check_api_budget_before_call()
        text = self.client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        usage = self.client.last_usage
        self._record_api_usage(run_dir=run_dir, usage=usage)
        return text

    def _max_gpus_per_run(self) -> int:
        raw = self.settings.research.get(
            "max_gpus_per_run",
            self.settings.remote.get("default_gpus", 4),
        )
        try:
            requested = int(raw)
        except (TypeError, ValueError):
            requested = 4
        if requested <= 0:
            raise ValueError("max_gpus_per_run must be > 0")

        remote_cap_raw = self.settings.remote.get("max_gpus", requested)
        try:
            remote_cap = int(remote_cap_raw)
        except (TypeError, ValueError):
            remote_cap = requested
        if remote_cap > 0:
            return min(requested, remote_cap)
        return requested

    def _max_api_usd_per_day(self) -> float:
        raw = self.settings.research.get("max_api_usd_per_day", 0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def _api_pricing(self) -> tuple[float, float, float]:
        def _f(name: str, default: float = 0.0) -> float:
            raw = self.settings.research.get(name, default)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        input_rate = _f("api_input_usd_per_1m_tokens", 0.0)
        output_rate = _f("api_output_usd_per_1m_tokens", 0.0)
        cached_input_rate = _f("api_cached_input_usd_per_1m_tokens", input_rate)
        return input_rate, output_rate, cached_input_rate

    def _validate_api_budget_config(self) -> None:
        limit = self._max_api_usd_per_day()
        if limit <= 0:
            return

        input_rate, output_rate, cached_rate = self._api_pricing()
        if input_rate <= 0 and output_rate <= 0 and cached_rate <= 0:
            raise ValueError(
                "max_api_usd_per_day > 0 but pricing is not configured. "
                "Set research.api_input_usd_per_1m_tokens and api_output_usd_per_1m_tokens."
            )

    def _check_api_budget_before_call(self) -> None:
        limit = self._max_api_usd_per_day()
        if limit <= 0:
            return

        usage = self._load_daily_usage()
        if float(usage.get("estimated_usd", 0.0)) >= limit:
            raise RuntimeError(
                f"API budget exhausted for today: {usage.get('estimated_usd', 0.0):.6f} >= {limit:.6f} USD"
            )

    def _record_api_usage(self, run_dir: Path, usage: dict[str, int]) -> None:
        input_tokens = max(0, int(usage.get("input_tokens", 0)))
        output_tokens = max(0, int(usage.get("output_tokens", 0)))
        total_tokens = max(0, int(usage.get("total_tokens", input_tokens + output_tokens)))
        cached_input_tokens = max(0, int(usage.get("cached_input_tokens", 0)))
        billable_input_tokens = max(0, input_tokens - cached_input_tokens)

        input_rate, output_rate, cached_rate = self._api_pricing()
        estimated_usd = (
            (billable_input_tokens / 1_000_000.0) * input_rate
            + (cached_input_tokens / 1_000_000.0) * cached_rate
            + (output_tokens / 1_000_000.0) * output_rate
        )

        run_usage_path = run_dir / "api_usage.json"
        run_usage = self._read_json_dict(run_usage_path)
        run_usage = {
            "calls": int(run_usage.get("calls", 0)) + 1,
            "input_tokens": int(run_usage.get("input_tokens", 0)) + input_tokens,
            "cached_input_tokens": int(run_usage.get("cached_input_tokens", 0)) + cached_input_tokens,
            "output_tokens": int(run_usage.get("output_tokens", 0)) + output_tokens,
            "total_tokens": int(run_usage.get("total_tokens", 0)) + total_tokens,
            "estimated_usd": float(run_usage.get("estimated_usd", 0.0)) + estimated_usd,
            "updated_at": datetime.now().isoformat(),
        }
        run_usage_path.write_text(json.dumps(run_usage, ensure_ascii=False, indent=2), encoding="utf-8")

        daily_usage_path = self._daily_usage_path()
        daily_usage = self._read_json_dict(daily_usage_path)
        daily_usage = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "calls": int(daily_usage.get("calls", 0)) + 1,
            "input_tokens": int(daily_usage.get("input_tokens", 0)) + input_tokens,
            "cached_input_tokens": int(daily_usage.get("cached_input_tokens", 0)) + cached_input_tokens,
            "output_tokens": int(daily_usage.get("output_tokens", 0)) + output_tokens,
            "total_tokens": int(daily_usage.get("total_tokens", 0)) + total_tokens,
            "estimated_usd": float(daily_usage.get("estimated_usd", 0.0)) + estimated_usd,
            "updated_at": datetime.now().isoformat(),
        }
        daily_usage_path.parent.mkdir(parents=True, exist_ok=True)
        daily_usage_path.write_text(json.dumps(daily_usage, ensure_ascii=False, indent=2), encoding="utf-8")

        self._append_progress(
            run_dir,
            (
                "api usage updated: "
                f"+in={input_tokens}, +out={output_tokens}, +cost=${estimated_usd:.6f}, "
                f"daily=${daily_usage['estimated_usd']:.6f}"
            ),
        )

        limit = self._max_api_usd_per_day()
        if limit > 0 and float(daily_usage["estimated_usd"]) > limit:
            raise RuntimeError(
                f"API budget exceeded for today: {daily_usage['estimated_usd']:.6f} > {limit:.6f} USD"
            )

    def _load_daily_usage(self) -> dict:
        return self._read_json_dict(self._daily_usage_path())

    def _daily_usage_path(self) -> Path:
        day = datetime.now().strftime("%Y-%m-%d")
        return self.repo_root / "runs" / ".budget" / f"{day}.json"

    def _init_project_layout(self, run_id: str, topic: str, run_dir: Path) -> Path:
        project_dir = self.repo_root / "projects" / run_id
        (project_dir / "exp").mkdir(parents=True, exist_ok=True)
        (project_dir / "idea").mkdir(parents=True, exist_ok=True)
        (project_dir / "writing").mkdir(parents=True, exist_ok=True)
        (project_dir / "FARS_MEMO").mkdir(parents=True, exist_ok=True)
        (project_dir / "EXPERIMENT_RESULTS").mkdir(parents=True, exist_ok=True)

        readme = project_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                (
                    f"# {run_id}\n\n"
                    f"- topic: {topic}\n"
                    f"- created_at: {datetime.now().isoformat()}\n"
                    f"- run_dir: runs/{run_id}\n"
                ),
                encoding="utf-8",
            )

        run_ptr = project_dir / "RUN_DIR"
        run_ptr.write_text(f"runs/{run_id}\n", encoding="utf-8")

        memo = project_dir / "FARS_MEMO" / "notes.md"
        if not memo.exists():
            memo.write_text("# Notes\n\n", encoding="utf-8")

        self._copy_if_exists(run_dir / "status.json", project_dir / "EXPERIMENT_RESULTS" / "status.json")
        return project_dir

    def _sync_project_artifacts(self, run_dir: Path, project_dir: Path, run_id: str, topic: str) -> None:
        idea_dir = project_dir / "idea"
        exp_dir = project_dir / "exp"
        memo_dir = project_dir / "FARS_MEMO"
        results_dir = project_dir / "EXPERIMENT_RESULTS"
        literature_dir = idea_dir / "literature"
        agents_dir = idea_dir / "agents"

        self._copy_if_exists(run_dir / "status.json", results_dir / "status.json")
        self._copy_if_exists(run_dir / "progress.log", results_dir / "progress.log")
        self._copy_if_exists(run_dir / "meta.json", results_dir / "meta.json")
        self._copy_if_exists(run_dir / "api_usage.json", results_dir / "api_usage.json")

        self._copy_if_exists(run_dir / "idea.md", idea_dir / "idea.md")
        self._copy_if_exists(run_dir / "plan.raw.txt", exp_dir / "plan.raw.txt")
        self._copy_if_exists(run_dir / "plan.json", exp_dir / "plan.json")
        self._copy_if_exists(run_dir / "plan.json", project_dir / "task_plan.json")
        self._copy_if_exists(run_dir / "implementation.md", exp_dir / "implementation.md")
        self._copy_if_exists(run_dir / "experiment.sh", exp_dir / "experiment.sh", executable=True)

        self._copy_if_exists(run_dir / "literature" / "review.md", literature_dir / "review.md")
        self._copy_if_exists(run_dir / "literature" / "digest.md", literature_dir / "digest.md")
        self._copy_if_exists(run_dir / "literature" / "papers.json", literature_dir / "papers.json")

        agents_src = run_dir / "agents"
        if agents_src.exists() and agents_src.is_dir():
            for p in sorted(agents_src.glob("*.md")):
                self._copy_if_exists(p, agents_dir / p.name)

        feedback_src = run_dir / "feedback"
        if feedback_src.exists() and feedback_src.is_dir():
            for p in sorted(feedback_src.glob("*.md")):
                self._copy_if_exists(p, memo_dir / "feedback" / p.name)

        manifest = {
            "run_id": run_id,
            "topic": topic,
            "project_dir": f"projects/{run_id}",
            "run_dir": f"runs/{run_id}",
            "synced_at": datetime.now().isoformat(),
        }
        (project_dir / "project_meta.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _copy_if_exists(src: Path, dst: Path, executable: bool = False) -> None:
        if not src.exists() or not src.is_file():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if executable:
            dst.chmod(0o755)

    def _literature_sources(self) -> list[str]:
        raw = self.settings.research.get("literature_sources", "arxiv")
        items: list[str]
        if isinstance(raw, (list, tuple)):
            items = [str(x).strip().lower() for x in raw]
        else:
            items = [x.strip().lower() for x in str(raw).split(",")]
        allowed = {"arxiv", "semantic_scholar", "semanticscholar", "openalex"}
        out: list[str] = []
        for x in items:
            if not x:
                continue
            if x not in allowed:
                continue
            if x == "semanticscholar":
                x = "semantic_scholar"
            if x not in out:
                out.append(x)
        return out or ["arxiv"]

    @staticmethod
    def _to_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        s = str(value).strip().lower()
        return s in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _read_json_dict(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_feedback_templates(self, run_dir: Path) -> None:
        feedback_dir = run_dir / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        template = (
            "Write feedback notes in this folder.\n\n"
            "Supported files:\n"
            "- global.md: applies to all following stages\n"
            "- literature.md / ideation_agents.md / idea.md / planning.md / implementation.md: stage-specific feedback\n"
            "- <stage>.approve: continue stage\n"
            "- <stage>.revise: regenerate stage using <stage>.md content\n"
        )
        (feedback_dir / "README.md").write_text(template, encoding="utf-8")

    def _collect_feedback(self, run_dir: Path, stage: str) -> str:
        feedback_dir = run_dir / "feedback"
        parts: list[str] = []
        for name in ("global.md", f"{stage}.md"):
            p = feedback_dir / name
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    parts.append(f"[{name}]\n{txt}")
        return "\n\n".join(parts)

    def _await_stage_action(self, run_dir: Path, stage: str, timeout_sec: int) -> tuple[str, str]:
        feedback_dir = run_dir / "feedback"
        approve = feedback_dir / f"{stage}.approve"
        revise = feedback_dir / f"{stage}.revise"
        note_file = feedback_dir / f"{stage}.md"

        self._write_status(
            run_dir,
            state="waiting_feedback",
            stage=stage,
            message=(
                f"waiting for feedback: create {stage}.approve or {stage}.revise under feedback/"
            ),
        )
        self._append_progress(run_dir, f"interactive gate opened: {stage}")

        start = time.time()
        while True:
            if revise.exists():
                revise.unlink(missing_ok=True)
                note = note_file.read_text(encoding="utf-8").strip() if note_file.exists() else ""
                self._write_status(run_dir, state="running", stage=stage, message="revision requested")
                return "revise", note
            if approve.exists():
                approve.unlink(missing_ok=True)
                self._write_status(run_dir, state="running", stage=stage, message="approved")
                return "approve", ""

            if timeout_sec > 0 and (time.time() - start) >= timeout_sec:
                self._write_status(
                    run_dir,
                    state="running",
                    stage=stage,
                    message=f"feedback timeout ({timeout_sec}s), auto-continue",
                )
                return "approve", ""

            time.sleep(2)

    @staticmethod
    def _append_progress(run_dir: Path, message: str) -> None:
        line = f"{datetime.now().isoformat()} | {message}\n"
        with (run_dir / "progress.log").open("a", encoding="utf-8") as f:
            f.write(line)

    @staticmethod
    def _write_status(run_dir: Path, **fields: object) -> None:
        status_path = run_dir / "status.json"
        payload: dict[str, object] = {}
        if status_path.exists():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        payload.update(fields)
        payload["updated_at"] = datetime.now().isoformat()
        status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _materialize_experiment_script(run_dir: Path, implementation_doc: str) -> None:
        m = re.search(r"```bash\s*(.*?)```", implementation_doc, flags=re.DOTALL | re.IGNORECASE)
        if m:
            body = m.group(1).strip()
            script = f"#!/usr/bin/env bash\nset -euo pipefail\n\n{body}\n"
        else:
            script = (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n\n"
                "echo 'No bash block found in implementation.md. Please fill this script manually.'\n"
                "exit 1\n"
            )
        out = run_dir / "experiment.sh"
        out.write_text(script, encoding="utf-8")
        out.chmod(0o755)
