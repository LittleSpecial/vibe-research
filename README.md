# vibe-research

Minimal `idea -> plan -> implementation -> remote run` pipeline for RL + LLM research.

## Design
- Local (Mac): orchestration and model calls through `codex-lb` (`gpt-5.3-codex`).
- Remote (zw1): heavy training/evaluation only, under `~/zx/vibe-research`.
- Budget guardrails:
  - `max_gpu_hours_per_run = 12`
  - optional API budget field (can be `0` when using local codex-lb)

## Quick Start
```bash
cd /Users/zhaoxu/Developer/projects/vibe-research
./scripts/bootstrap_local.sh

# 1) Generate one full research cycle locally
vibe-research run-cycle --topic "low-resource RLVR for LLM reasoning with public datasets"

# 2) Sync code + generated run to remote
./scripts/sync_to_zw1.sh zw1 vibe-research

# 3) (One-time) bootstrap remote env
./scripts/remote_bootstrap.sh zw1 vibe-research

# 4) Submit remote job
./scripts/submit_remote.sh zw1 vibe-research <RUN_ID> 4 12
```

## Config
Main config: `configs/local.toml`.

If `codex-lb` requires OpenAI auth, set locally:
```bash
export OPENAI_API_KEY=...   # from your local auth bridge
```

## Notes
- This repo intentionally avoids touching your existing running experiments.
- Default run artifacts are under `runs/<RUN_ID>/`.
