#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-zw1}"
REMOTE_REPO="${2:-vibe-research}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'runs/*/checkpoints' \
  --exclude 'runs/*/logs' \
  --exclude 'runs/*/artifacts' \
  "$ROOT/" "$HOST:~/zx/$REMOTE_REPO/"

echo "Synced to $HOST:~/zx/$REMOTE_REPO"
