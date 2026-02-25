# IRIS v4.2 Runbook

This runbook covers end-to-end execution for:
- IRIS partial-ID pair construction
- q_L training vs raw/generic-pess baselines
- evaluator-shift evaluation

## 1) Local setup

```bash
cd /Users/zhaoxu/Developer/projects/vibe-research
./scripts/bootstrap_local.sh
source .venv/bin/activate
```

## 2) Smoke run (local then remote)

```bash
# materialize a smoke run
RUN_ID=$(./scripts/create_iris_run.sh smoke iris-v42)

# local smoke (synthetic rollout backend)
python -m vibe_research.cli run-experiment --run-dir "runs/$RUN_ID"

# sync and run smoke on zw1
./scripts/sync_to_zw1.sh zw1 vibe-research
./scripts/remote_bootstrap.sh zw1 vibe-research
./scripts/submit_remote.sh zw1 vibe-research "$RUN_ID" 1 1
```

## 3) Formal run (4xA100)

```bash
# formal defaults: local_hf backend + Qwen/Qwen2.5-1.5B-Instruct
RUN_ID_FORMAL=$(./scripts/create_iris_run.sh formal iris-v42)

./scripts/sync_to_zw1.sh zw1 vibe-research
./scripts/submit_remote.sh zw1 vibe-research "$RUN_ID_FORMAL" 4 12

# optional: pass extra env vars into sbatch --export
./scripts/submit_remote.sh zw1 vibe-research "$RUN_ID_FORMAL" 4 12 \
  "LOCAL_HF_FALLBACK_TO_SYNTHETIC=1"
```

Optional overrides:

```bash
# choose backend/model
ROLLOUT_BACKEND=local_hf ROLLOUT_MODEL=Qwen/Qwen2.5-3B-Instruct \
  python -m vibe_research.cli run-experiment --run-dir "runs/$RUN_ID_FORMAL"

# adjust scale
a=1200  # example
MATH_TRAIN_SIZE=$a MAX_NEW_TOKENS=128 UPDATES_MAIN=280 \
  python -m vibe_research.cli run-experiment --run-dir "runs/$RUN_ID_FORMAL"
```

## 4) Monitor

Local run:

```bash
python -m vibe_research.cli watch-run --run-dir "runs/$RUN_ID_FORMAL" --follow --until-done
```

Remote run:

```bash
./scripts/monitor_remote_fars.sh zw1 vibe-research "$RUN_ID_FORMAL"
./scripts/tail_remote_logs.sh zw1 vibe-research
```

Direct Slurm check:

```bash
~/.codex/skills/paracloud-zx-ssh-workflow/scripts/ssh_zx.sh zw1 --repo vibe-research \
  "squeue -u $USER -o '%.18i %.9P %.20j %.2t %.10M %.9l %R'"
```

## 5) Artifacts to inspect

- `runs/<RUN_ID>/results.json`
- `runs/<RUN_ID>/iris_pairs.summary.json`
- `runs/<RUN_ID>/eval/summary.json`
- `runs/<RUN_ID>/eval/model_metrics.json`
- `runs/<RUN_ID>/final/report.json`

## 6) Common failure fixes

1. Job completes too fast with no artifacts.
- Check `remote/slurm_run_experiment.sh` executes `bash runs/<RUN_ID>/experiment.sh`.

2. `local_hf` backend missing runtime.
- The generated `experiment.sh` auto-picks a Python containing `torch+transformers`.
- If still failing, set `PY_BIN=/home/bingxing2/apps/miniforge3/24.1.2/bin/python3`.

3. `local_hf` requires CUDA but CUDA is unavailable.
- Formal runs default to `LOCAL_HF_REQUIRE_CUDA=1`.
- To keep the run alive with synthetic fallback:
  `./scripts/submit_remote.sh ... "LOCAL_HF_FALLBACK_TO_SYNTHETIC=1"`

3. Partial-ID model collapses (q_L too negative).
- Check `iris_pairs.summary.json` for `q_l_mean` and `offset_applied`.
- Rebuild pairs with offset calibration enabled (default).
