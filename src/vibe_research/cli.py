from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .config import load_settings
from .execution import run_experiment
from .pipeline import ResearchCycleRunner


def cmd_run_cycle(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    settings = load_settings(args.config)
    topic = args.topic or settings.research.get("default_topic", "rl+llm")
    agent_count = args.agent_count
    if agent_count <= 0:
        agent_count = int(settings.agents.get("default_count", 4))

    runner = ResearchCycleRunner(repo_root=repo_root, settings=settings)
    run_dir = runner.run_cycle(
        topic=topic,
        dry_run=args.dry_run,
        interactive=args.interactive,
        agent_count=agent_count,
        feedback_timeout=args.feedback_timeout,
    )
    print(run_dir)
    return 0


def cmd_run_experiment(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}", file=sys.stderr)
        return 2
    return run_experiment(run_dir)


def cmd_watch_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    status_file = run_dir / "status.json"
    progress_file = run_dir / "progress.log"

    if not run_dir.exists():
        print(f"run dir not found: {run_dir}", file=sys.stderr)
        return 2

    seen_sig = ""
    seen_progress_size = -1

    while True:
        if status_file.exists():
            try:
                status = json.loads(status_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {}
        else:
            status = {}

        sig = json.dumps(status, sort_keys=True, ensure_ascii=False)
        if sig != seen_sig:
            seen_sig = sig
            print(
                " | ".join(
                    [
                        str(status.get("updated_at", "?")),
                        f"state={status.get('state', '?')}",
                        f"stage={status.get('stage', '?')}",
                        f"step={status.get('step', '?')}/{status.get('total_steps', '?')}",
                        f"msg={status.get('message', '')}",
                    ]
                )
            )

        if progress_file.exists():
            size_now = progress_file.stat().st_size
            if size_now != seen_progress_size:
                seen_progress_size = size_now
                lines = progress_file.read_text(encoding="utf-8").splitlines()
                for line in lines[-args.progress_tail :]:
                    print(f"  {line}")

        done = status.get("state") in {"completed", "failed", "dry_run"}
        if not args.follow:
            break
        if done and args.until_done:
            break
        time.sleep(args.interval)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vibe-research")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_cycle = sub.add_parser("run-cycle", help="generate idea/plan/implementation artifacts")
    p_cycle.add_argument("--config", default="configs/local.toml")
    p_cycle.add_argument("--topic", default="")
    p_cycle.add_argument("--dry-run", action="store_true")
    p_cycle.add_argument("--interactive", action="store_true", help="pause after each stage for human approve/revise")
    p_cycle.add_argument("--agent-count", type=int, default=0, help="number of ideation agents (0=use config)")
    p_cycle.add_argument(
        "--feedback-timeout",
        type=int,
        default=0,
        help="seconds to wait for feedback in interactive mode (0 = wait forever)",
    )
    p_cycle.set_defaults(func=cmd_run_cycle)

    p_exp = sub.add_parser("run-experiment", help="execute run_dir/experiment.sh")
    p_exp.add_argument("--run-dir", required=True)
    p_exp.set_defaults(func=cmd_run_experiment)

    p_watch = sub.add_parser("watch-run", help="watch status/progress of a run directory")
    p_watch.add_argument("--run-dir", required=True)
    p_watch.add_argument("--follow", action="store_true", help="poll continuously")
    p_watch.add_argument("--until-done", action="store_true", help="exit when run reaches completed/failed")
    p_watch.add_argument("--interval", type=float, default=2.0)
    p_watch.add_argument("--progress-tail", type=int, default=6)
    p_watch.set_defaults(func=cmd_watch_run)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)
