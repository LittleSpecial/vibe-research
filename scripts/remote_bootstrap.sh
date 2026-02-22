#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-zw1}"
REMOTE_REPO="${2:-vibe-research}"
ZX_SSH_SCRIPT="${ZX_SSH_SCRIPT:-$HOME/.codex/skills/paracloud-zx-ssh-workflow/scripts/ssh_zx.sh}"

"$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && python -m pip install -e ."

echo "Remote bootstrap done on $HOST:$REMOTE_REPO"
