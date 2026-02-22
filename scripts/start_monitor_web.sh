#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run ./scripts/bootstrap_local.sh first." >&2
  exit 2
fi
PORT="${1:-8787}"
HOST="${2:-127.0.0.1}"
REMOTE_HOST="${3:-zw1}"
REMOTE_REPO="${4:-vibe-research}"

mkdir -p runs/.orchestrator_logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="runs/.orchestrator_logs/monitor_web_${TS}.log"

nohup .venv/bin/vibe-monitor --host "$HOST" --port "$PORT" --remote-host "$REMOTE_HOST" --remote-repo "$REMOTE_REPO" >"$LOG_PATH" 2>&1 </dev/null &
PID=$!
echo "$PID" > runs/LATEST_MONITOR_PID

echo "Started vibe-monitor pid=$PID"
echo "URL: http://$HOST:$PORT"
echo "Log: $ROOT/$LOG_PATH"
