#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-zw1}"
REMOTE_REPO="${2:-vibe-research}"
RUN_ID="${3:?RUN_ID required}"
GPUS="${4:-4}"
HOURS="${5:-12}"

if [[ "$HOURS" -gt 12 ]]; then
  echo "Refuse: HOURS=$HOURS exceeds guardrail 12" >&2
  exit 2
fi
if [[ "$GPUS" -gt 8 ]]; then
  echo "Refuse: GPUS=$GPUS exceeds guardrail 8" >&2
  exit 2
fi

ZX_SSH_SCRIPT="${ZX_SSH_SCRIPT:-$HOME/.codex/skills/paracloud-zx-ssh-workflow/scripts/ssh_zx.sh}"
CMD="mkdir -p logs && sbatch --gpus=${GPUS} --time=${HOURS}:00:00 remote/slurm_run_experiment.sh runs/${RUN_ID}"
"$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "$CMD"
