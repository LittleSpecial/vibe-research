from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import load_settings
from .execution import run_experiment
from .pipeline import ResearchCycleRunner


def cmd_run_cycle(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    settings = load_settings(args.config)
    topic = args.topic or settings.research.get("default_topic", "rl+llm")

    runner = ResearchCycleRunner(repo_root=repo_root, settings=settings)
    run_dir = runner.run_cycle(topic=topic, dry_run=args.dry_run)
    print(run_dir)
    return 0


def cmd_run_experiment(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}", file=sys.stderr)
        return 2
    return run_experiment(run_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vibe-research")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_cycle = sub.add_parser("run-cycle", help="generate idea/plan/implementation artifacts")
    p_cycle.add_argument("--config", default="configs/local.toml")
    p_cycle.add_argument("--topic", default="")
    p_cycle.add_argument("--dry-run", action="store_true")
    p_cycle.set_defaults(func=cmd_run_cycle)

    p_exp = sub.add_parser("run-experiment", help="execute run_dir/experiment.sh")
    p_exp.add_argument("--run-dir", required=True)
    p_exp.set_defaults(func=cmd_run_experiment)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)
