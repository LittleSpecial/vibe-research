#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOST="${1:-zw1}"
REMOTE_REPO="${2:-vibe-research}"
RUN_ID="${3:-}"
if [[ -z "$RUN_ID" ]]; then
  if [[ -f runs/LATEST_RUN ]]; then
    RUN_ID="$(cat runs/LATEST_RUN)"
  else
    echo "Usage: $0 <host> <remote_repo> <RUN_ID>" >&2
    exit 2
  fi
fi

ZX_SSH_SCRIPT="${ZX_SSH_SCRIPT:-$HOME/.codex/skills/paracloud-zx-ssh-workflow/scripts/ssh_zx.sh}"
JOB_ID=""
if [[ -f "runs/$RUN_ID/remote_submit.json" ]]; then
  JOB_ID="$(python3 - <<PY
import json
from pathlib import Path
p=Path('runs/$RUN_ID/remote_submit.json')
try:
  d=json.loads(p.read_text())
  print(d.get('job_id',''))
except Exception:
  print('')
PY
)"
fi

while true; do
  if [[ -t 1 ]]; then
    clear
  fi
  echo "==== Vibe Research Remote Monitor ===="
  echo "Time:      $(date '+%F %T')"
  echo "Host:      $HOST"
  echo "Repo:      $REMOTE_REPO"
  echo "Run:       $RUN_ID"
  echo "Job ID:    ${JOB_ID:-unknown}"
  echo

  echo "--- local status mirror ---"
  if [[ -d "runs/$RUN_ID" ]]; then
    source .venv/bin/activate >/dev/null 2>&1 || true
    vibe-research watch-run --run-dir "runs/$RUN_ID" || true
  else
    echo "local run dir not found"
  fi

  echo
  echo "--- remote run status ---"
  "$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "python3 - <<'PY'
import json
from pathlib import Path
p=Path('runs/${RUN_ID}/status.json')
if p.exists():
    try:
        d=json.loads(p.read_text())
    except Exception:
        d={}
    print(f\"{d.get('updated_at','?')} | state={d.get('state','?')} | stage={d.get('stage','?')} | step={d.get('step','?')}/{d.get('total_steps','?')} | msg={d.get('message','')}\")
else:
    print('status.json not found')
PY" || true

  if [[ -n "$JOB_ID" ]]; then
    echo
    echo "--- slurm queue ---"
    "$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "squeue -j $JOB_ID -o '%.18i %.9P %.20j %.8u %.2t %.10M %.10l %.6D %R' 2>/dev/null | sed -n '1,5p'" || true
    echo
    echo "--- slurm job detail ---"
    "$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "scontrol show job $JOB_ID 2>/dev/null | tr ' ' '\n' | grep -E 'JobState=|RunTime=|TimeLimit=|NumNodes=|NumCPUs=|GRES=|NodeList='" || true
  fi

  echo
  echo "--- remote log tail ---"
  if [[ -n "$JOB_ID" ]]; then
    "$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "ls -1t logs/*${JOB_ID}*.out logs/*${JOB_ID}*.err 2>/dev/null | head -n 2 | xargs -r tail -n 40" || true
  else
    "$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "ls -1t logs/*.out logs/*.err 2>/dev/null | head -n 2 | xargs -r tail -n 40" || true
  fi

  sleep 3
done
