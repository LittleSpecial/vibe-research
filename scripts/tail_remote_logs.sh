#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-zw1}"
REMOTE_REPO="${2:-vibe-research}"
ZX_SSH_SCRIPT="${ZX_SSH_SCRIPT:-$HOME/.codex/skills/paracloud-zx-ssh-workflow/scripts/ssh_zx.sh}"
"$ZX_SSH_SCRIPT" "$HOST" --repo "$REMOTE_REPO" "ls -1t logs/*.out logs/*.err 2>/dev/null | head -n 6 | xargs -r tail -n 120"
