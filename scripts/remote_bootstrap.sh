#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-zw1}"
REMOTE_REPO="${2:-vibe-research}"
ZX_SSH_SCRIPT="${ZX_SSH_SCRIPT:-$HOME/.codex/skills/paracloud-zx-ssh-workflow/scripts/ssh_zx.sh}"

"$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "bash scripts/bootstrap_on_cluster.sh"

echo "Remote bootstrap done on $HOST:$REMOTE_REPO"
