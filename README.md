# vibe-research

Minimal `literature -> idea -> plan -> implementation -> remote run` pipeline for RL + LLM research.

## Design
- Local (Mac): orchestration and model calls through `codex-lb` (`gpt-5.3-codex`).
- Remote (zw1): heavy training/evaluation only, under `~/zx/vibe-research`.
- Budget guardrails:
  - `max_gpu_hours_per_run = 12` (wall-clock)
  - `max_gpus_per_run = 4` (clamped by `remote.max_gpus = 8`)
  - optional API guardrail: set `max_api_usd_per_day > 0` and pricing keys
    `api_input_usd_per_1m_tokens`, `api_output_usd_per_1m_tokens`
- Multi-agent ideation:
  - default 4 agents: `pi_vision_agent` / `methodology_agent` / `experiment_engineer_agent` / `reviewer_redteam_agent`
- Real-time observability:
  - `runs/<RUN_ID>/status.json`
  - `runs/<RUN_ID>/progress.log`
  - `runs/<RUN_ID>/literature/*` (paper retrieval + review)
- FARS-style per-idea project mirror:
  - `projects/<RUN_ID>/exp`
  - `projects/<RUN_ID>/idea`
  - `projects/<RUN_ID>/writing`
  - `projects/<RUN_ID>/FARS_MEMO`
  - `projects/<RUN_ID>/EXPERIMENT_RESULTS`

## Quick Start
```bash
cd /Users/zhaoxu/Developer/projects/vibe-research
./scripts/bootstrap_local.sh
source .venv/bin/activate

# 1) Start FARS-style web monitor (local dashboard)
./scripts/start_monitor_web.sh 8787 127.0.0.1 zw1 vibe-research
# open: http://127.0.0.1:8787

# 2) Start one full research cycle in background (interactive by default)
./scripts/start_cycle.sh configs/local.toml "RL + LLM topic" interactive 0 0

# 3) Find latest run and watch status in terminal
RUN_ID=$(cat runs/LATEST_RUN)
./scripts/monitor_local_fars.sh "$RUN_ID"

# 4) Sync code + generated run to remote
./scripts/sync_to_zw1.sh zw1 vibe-research

# 5) (One-time) bootstrap remote env
./scripts/remote_bootstrap.sh zw1 vibe-research

# 6) Submit remote job (4 GPUs, 12h)
./scripts/submit_remote.sh zw1 vibe-research "$RUN_ID" 4 12

# 7) FARS-style remote monitor (status + slurm + logs)
./scripts/monitor_remote_fars.sh zw1 vibe-research "$RUN_ID"
./scripts/tail_remote_logs.sh zw1 vibe-research
```

## IRIS v4.2 Run (Partial-ID under Evaluator Shift)
Generate a runnable IRIS experiment (`runs/<RUN_ID>/experiment.sh`), then sync and submit.

```bash
cd /Users/zhaoxu/Developer/projects/vibe-research
source .venv/bin/activate

# 1) Materialize smoke run
RUN_ID=$(./scripts/create_iris_run.sh smoke iris-v42-partial-id)
echo "$RUN_ID"

# 2) Local smoke (fast)
python -m vibe_research.cli run-experiment --run-dir "runs/$RUN_ID"

# 3) Sync + remote smoke submit (1 GPU, 1h)
./scripts/sync_to_zw1.sh zw1 vibe-research
./scripts/remote_bootstrap.sh zw1 vibe-research
./scripts/submit_remote.sh zw1 vibe-research "$RUN_ID" 1 1

# 4) After smoke success: materialize and submit formal run (4 GPU, 12h)
# formal defaults to real rollout backend: local_hf + Qwen2.5-1.5B-Instruct
RUN_ID_FORMAL=$(./scripts/create_iris_run.sh formal iris-v42-partial-id)
./scripts/sync_to_zw1.sh zw1 vibe-research
./scripts/submit_remote.sh zw1 vibe-research "$RUN_ID_FORMAL" 4 12

# If local_hf CUDA is unavailable on cluster, force automatic fallback:
./scripts/submit_remote.sh zw1 vibe-research "$RUN_ID_FORMAL" 4 12 \
  "LOCAL_HF_FALLBACK_TO_SYNTHETIC=1"
```

Detailed protocol and monitoring checklist:
- `docs/IRIS_V42_RUNBOOK.md`

Remote bootstrap now follows cluster module setup (Paracloud aarch64):
- `module purge`
- auto-pick available CUDA module (`compilers/cuda/*` preferred)
- auto-pick GCC module (`compilers/gcc/*`)
- optional python module (falls back to current PATH python)
- export `CUDA_HOME` from `nvcc` path
- create/refresh `.venv` and install `requirements.txt` + editable package

## Web Monitor
- Start: `./scripts/start_monitor_web.sh 8787 127.0.0.1 zw1 vibe-research`
- URL: `http://127.0.0.1:8787`
- Features:
  - Run list + live local status/progress.
  - Start run-cycle directly from web UI.
  - Submit interactive feedback (`approve` / `revise`) per stage.
  - Pull remote status/queue/logs via SSH skill script.

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

API budget notes:
- `max_api_usd_per_day = 0` means disabled.
- If `max_api_usd_per_day > 0`, pricing keys must be configured.
- Usage ledgers are written to:
  - `runs/<RUN_ID>/api_usage.json`
  - `runs/.budget/YYYY-MM-DD.json`

Literature notes:
- Every cycle starts with a literature retrieval + synthesis stage.
- Default retrieval sources: `arxiv`, `semantic_scholar` (optional: add `openalex` via `research.literature_sources`).
- Retrieved metadata is stored in `runs/<RUN_ID>/literature/papers.json`.
- Review synthesis is in `runs/<RUN_ID>/literature/review.md`.
- PDFs are downloaded (configurable) and then synced to remote under:
  - `~/zx/vibe-research/runs/<RUN_ID>/literature/pdfs/`
  - local copies can be auto-deleted after successful sync.

If `codex-lb` requires auth, set locally:
```bash
export OPENAI_API_KEY=...   # from your local auth bridge
```

## Notes
- This repo avoids touching your existing running experiments in other repos.
- Default run artifacts are under `runs/<RUN_ID>/`.
- A synchronized subproject view is also created under `projects/<RUN_ID>/` for each idea.
