from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

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
            max_output_tokens=int(model.get("max_output_tokens", 1400)),
        )

    def run_cycle(
        self,
        topic: str,
        dry_run: bool = False,
        interactive: bool = False,
        agent_count: int = 3,
        feedback_timeout: int = 0,
    ) -> Path:
        max_gpu_hours = float(self.settings.research.get("max_gpu_hours_per_run", 12))
        if max_gpu_hours <= 0:
            raise ValueError("max_gpu_hours_per_run must be > 0")

        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(topic)}"
        run_dir = self.repo_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (self.repo_root / "runs" / "LATEST_RUN").write_text(run_id, encoding="utf-8")

        self._write_feedback_templates(run_dir)
        total_steps = 4 if agent_count > 1 else 3
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
            return run_dir

        try:
            debate_notes: list[tuple[str, str]] = []
            if agent_count > 1:
                self._write_status(
                    run_dir,
                    run_id=run_id,
                    topic=topic,
                    state="running",
                    stage="ideation_agents",
                    step=1,
                    total_steps=total_steps,
                    message="running multi-agent ideation debate",
                )
                debate_notes = self._run_agent_debate(
                    topic=topic,
                    run_dir=run_dir,
                    agent_count=agent_count,
                    max_gpu_hours=max_gpu_hours,
                )
                self._append_progress(run_dir, f"generated {len(debate_notes)} agent notes")

            self._write_status(
                run_dir,
                run_id=run_id,
                topic=topic,
                state="running",
                stage="idea",
                step=(2 if agent_count > 1 else 1),
                total_steps=total_steps,
                message="generating synthesized idea",
            )
            idea_md = self._run_reviewable_stage(
                run_dir=run_dir,
                stage="idea",
                interactive=interactive,
                feedback_timeout=feedback_timeout,
                generator=lambda feedback_text: self._generate_idea(
                    topic=topic,
                    debate_notes=debate_notes,
                    max_gpu_hours=max_gpu_hours,
                    feedback_text=feedback_text,
                ),
                output_path=run_dir / "idea.md",
            )

            self._write_status(
                run_dir,
                run_id=run_id,
                topic=topic,
                state="running",
                stage="planning",
                step=(3 if agent_count > 1 else 2),
                total_steps=total_steps,
                message="building executable experiment plan",
            )
            plan_raw = self._run_reviewable_stage(
                run_dir=run_dir,
                stage="planning",
                interactive=interactive,
                feedback_timeout=feedback_timeout,
                generator=lambda feedback_text: self._generate_plan_raw(
                    idea_md=idea_md,
                    max_gpu_hours=max_gpu_hours,
                    feedback_text=feedback_text,
                ),
                output_path=run_dir / "plan.raw.txt",
            )
            plan = self._parse_plan(plan_raw)
            self._validate_plan_budget(plan=plan, max_gpu_hours=max_gpu_hours)
            (run_dir / "plan.json").write_text(
                json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            self._write_status(
                run_dir,
                run_id=run_id,
                topic=topic,
                state="running",
                stage="implementation",
                step=total_steps,
                total_steps=total_steps,
                message="generating runnable implementation",
            )
            impl_doc = self._run_reviewable_stage(
                run_dir=run_dir,
                stage="implementation",
                interactive=interactive,
                feedback_timeout=feedback_timeout,
                generator=lambda feedback_text: self._generate_implementation(
                    plan=plan,
                    max_gpu_hours=max_gpu_hours,
                    feedback_text=feedback_text,
                ),
                output_path=run_dir / "implementation.md",
            )
            self._materialize_experiment_script(run_dir=run_dir, implementation_doc=impl_doc)

            meta = {
                "run_id": run_id,
                "topic": topic,
                "max_gpu_hours_per_run": max_gpu_hours,
                "max_gpu_cards_per_run": 4,
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
            raise

    def _run_agent_debate(
        self,
        topic: str,
        run_dir: Path,
        agent_count: int,
        max_gpu_hours: float,
    ) -> list[tuple[str, str]]:
        roles = [
            ("novelty_agent", "Focus on novelty and publishability."),
            ("feasibility_agent", "Focus on whether this can be executed on 4xA100 within the time budget."),
            ("risk_agent", "Focus on failure modes, confounders, and reproducibility risks."),
            ("baseline_agent", "Focus on strong baselines and ablations needed for top-tier review."),
            ("efficiency_agent", "Focus on compute-efficient experiment design and staged validation."),
        ]
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
                f"Compute envelope: up to 4xA100; max wall-clock {max_gpu_hours:.1f}h per run.\n"
            )
            if feedback_text:
                user_prompt += f"\nExtra human feedback:\n{feedback_text}\n"

            note = self.client.complete(
                system_prompt="You are one specialized research agent in a multi-agent discussion.",
                user_prompt=user_prompt,
            )
            notes.append((role_name, note))
            (out_dir / f"{idx:02d}_{role_name}.md").write_text(note, encoding="utf-8")
            self._append_progress(run_dir, f"agent note completed: {role_name}")

        return notes

    def _generate_idea(
        self,
        topic: str,
        debate_notes: list[tuple[str, str]],
        max_gpu_hours: float,
        feedback_text: str,
    ) -> str:
        ideation_prompt = _read_prompt(self.repo_root / "prompts" / "ideation.md")
        notes_blob = "\n\n".join([f"[{name}]\n{text}" for name, text in debate_notes])
        user_prompt = (
            f"{ideation_prompt}\n\n"
            f"Topic seed: {topic}\n"
            f"Compute envelope: up to 4xA100; max wall-clock {max_gpu_hours:.1f}h per run.\n"
            f"If using >1 GPU, mention the required parallel setup explicitly.\n"
        )
        if notes_blob:
            user_prompt += f"\nAgent discussion notes:\n{notes_blob}\n"
        if feedback_text:
            user_prompt += f"\nHuman feedback for revision:\n{feedback_text}\n"

        return self.client.complete(
            system_prompt=(
                "You are the lead research agent. Synthesize one strong RL/LLM idea with clear experiments."
            ),
            user_prompt=user_prompt,
        )

    def _generate_plan_raw(self, idea_md: str, max_gpu_hours: float, feedback_text: str) -> str:
        planning_prompt = _read_prompt(self.repo_root / "prompts" / "planning.md")
        user_prompt = (
            f"{planning_prompt}\n\n"
            f"Budget constraint:\n"
            f"- Max wall-clock per run: {max_gpu_hours:.1f}h\n"
            f"- Hardware: up to 4xA100\n\n"
            f"Idea:\n{idea_md}"
        )
        if feedback_text:
            user_prompt += f"\n\nHuman feedback for revision:\n{feedback_text}\n"

        return self.client.complete(
            system_prompt="Return valid JSON only. No markdown wrappers.",
            user_prompt=user_prompt,
        )

    def _generate_implementation(self, plan: list[dict], max_gpu_hours: float, feedback_text: str) -> str:
        impl_prompt = _read_prompt(self.repo_root / "prompts" / "implementation.md")
        user_prompt = (
            f"{impl_prompt}\n\n"
            f"Runtime environment:\n"
            f"- Linux aarch64\n"
            f"- Slurm cluster\n"
            f"- up to 4xA100\n"
            f"- wall-clock <= {max_gpu_hours:.1f}h\n\n"
            f"Plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}"
        )
        if feedback_text:
            user_prompt += f"\n\nHuman feedback for revision:\n{feedback_text}\n"

        return self.client.complete(
            system_prompt="Be concrete. Prefer short, executable outputs.",
            user_prompt=user_prompt,
        )

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
    def _validate_plan_budget(plan: list[dict], max_gpu_hours: float) -> None:
        total_gpu = sum(float(x.get("est_gpu_hours", 0.0)) for x in plan)
        if total_gpu > max_gpu_hours * 4:
            raise RuntimeError(
                "Plan exceeds total GPU-hours envelope: "
                f"{total_gpu:.2f} > {max_gpu_hours * 4:.2f} (4xA100 * wall-clock budget)"
            )

    def _write_feedback_templates(self, run_dir: Path) -> None:
        feedback_dir = run_dir / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        template = (
            "Write feedback notes in this folder.\n\n"
            "Supported files:\n"
            "- global.md: applies to all following stages\n"
            "- idea.md / planning.md / implementation.md: stage-specific feedback\n"
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
