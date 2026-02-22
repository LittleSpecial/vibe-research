# vibe-research

Minimal `idea -> plan -> implementation -> remote run` pipeline for RL + LLM research.

## Design
- Local (Mac): orchestration and model calls through `codex-lb` (`gpt-5.3-codex`).
- Remote (zw1): heavy training/evaluation only, under `~/zx/vibe-research`.
- Budget guardrails:
  - `max_gpu_hours_per_run = 12` (wall-clock)
  - max GPUs per job = `8` (default submission uses `4`)
- Multi-agent ideation:
  - default 3 agents: novelty / feasibility / risk
- Real-time observability:
  - `runs/<RUN_ID>/status.json`
  - `runs/<RUN_ID>/progress.log`

## Quick Start
```bash
cd /Users/zhaoxu/Developer/projects/vibe-research
./scripts/bootstrap_local.sh
source .venv/bin/activate

# 1) Generate one full research cycle locally (non-interactive)
vibe-research run-cycle --topic "RL + LLM topic"

# 2) Find latest run and watch status
RUN_ID=$(cat runs/LATEST_RUN)
./scripts/watch_local_run.sh "$RUN_ID"

# 3) Sync code + generated run to remote
./scripts/sync_to_zw1.sh zw1 vibe-research

# 4) (One-time) bootstrap remote env
./scripts/remote_bootstrap.sh zw1 vibe-research

# 5) Submit remote job (4 GPUs, 12h)
./scripts/submit_remote.sh zw1 vibe-research "$RUN_ID" 4 12

# 6) Watch remote status/logs
./scripts/watch_remote_run.sh zw1 vibe-research "$RUN_ID"
./scripts/tail_remote_logs.sh zw1 vibe-research
```

## Interactive Feedback Loop
Use interactive mode when you want to review/revise each stage.

```bash
vibe-research run-cycle --interactive --agent-count 4 --topic "RL + LLM topic"
```

During run, feedback files are under `runs/<RUN_ID>/feedback/`:
- `global.md`: applies to next stages.
- `idea.md`, `planning.md`, `implementation.md`: stage-specific revision notes.
- `idea.approve` / `planning.approve` / `implementation.approve`: continue.
- `idea.revise` / `planning.revise` / `implementation.revise`: regenerate stage using corresponding `.md`.

## Config
Main config: `configs/local.toml`.

Fast local iteration config: `configs/local_fast.toml`.

If `codex-lb` requires auth, set locally:
```bash
export OPENAI_API_KEY=...   # from your local auth bridge
```

## Notes
- This repo avoids touching your existing running experiments in other repos.
- Default run artifacts are under `runs/<RUN_ID>/`.
