#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-zw1}"
REMOTE_REPO="${2:-vibe-research}"
RUN_ID="${3:-}"
if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 <host> <remote_repo> <RUN_ID>" >&2
  exit 2
fi

ZX_SSH_SCRIPT="${ZX_SSH_SCRIPT:-$HOME/.codex/skills/paracloud-zx-ssh-workflow/scripts/ssh_zx.sh}"

while true; do
  "$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "python - <<'PY'
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
  sleep 2
done
