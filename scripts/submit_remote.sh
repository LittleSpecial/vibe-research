#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-zw1}"
REMOTE_REPO="${2:-vibe-research}"
RUN_ID="${3:?RUN_ID required}"
GPUS="${4:-4}"
HOURS="${5:-12}"
EXPORT_VARS="${6:-}"

if [[ "$HOURS" -gt 12 ]]; then
  echo "Refuse: HOURS=$HOURS exceeds guardrail 12" >&2
  exit 2
fi
if [[ "$GPUS" -gt 8 ]]; then
  echo "Refuse: GPUS=$GPUS exceeds guardrail 8" >&2
  exit 2
fi

ZX_SSH_SCRIPT="${ZX_SSH_SCRIPT:-$HOME/.codex/skills/paracloud-zx-ssh-workflow/scripts/ssh_zx.sh}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR_LOCAL="$ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR_LOCAL"

SBATCH_EXPORT="--export=ALL"
if [[ -n "$EXPORT_VARS" ]]; then
  SBATCH_EXPORT="--export=ALL,${EXPORT_VARS}"
fi

CMD="mkdir -p logs runs/${RUN_ID} && sbatch --gpus=${GPUS} --time=${HOURS}:00:00 ${SBATCH_EXPORT} remote/slurm_run_experiment.sh runs/${RUN_ID}"
if ! OUT="$("$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "$CMD" 2>&1)"; then
  echo "$OUT" >&2
  cat > "$RUN_DIR_LOCAL/remote_submit_error.json" <<EOF
{
  "host": "$HOST",
  "remote_repo": "$REMOTE_REPO",
  "run_id": "$RUN_ID",
  "gpus": $GPUS,
  "hours": $HOURS,
  "export_vars": $(printf '%s' "$EXPORT_VARS" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'),
  "error": $(printf '%s' "$OUT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')
}
EOF
  echo "Recorded remote_submit_error.json at $RUN_DIR_LOCAL/remote_submit_error.json" >&2
  exit 1
fi
echo "$OUT"

JOB_ID="$(echo "$OUT" | grep -Eo '[0-9]+' | tail -n 1 || true)"
if [[ -n "$JOB_ID" ]]; then
  rm -f "$RUN_DIR_LOCAL/remote_submit_error.json"
  cat > "$RUN_DIR_LOCAL/remote_submit.json" <<EOF
{
  "host": "$HOST",
  "remote_repo": "$REMOTE_REPO",
  "run_id": "$RUN_ID",
  "job_id": "$JOB_ID",
  "gpus": $GPUS,
  "hours": $HOURS,
  "export_vars": $(printf '%s' "$EXPORT_VARS" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')
}
EOF
  echo "Recorded remote_submit.json with job_id=$JOB_ID at $RUN_DIR_LOCAL/remote_submit.json"
fi
