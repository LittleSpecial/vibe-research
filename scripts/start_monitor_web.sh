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
ln -sfn "monitor_web_${TS}.log" runs/.orchestrator_logs/latest_monitor.log

nohup .venv/bin/vibe-monitor --host "$HOST" --port "$PORT" --remote-host "$REMOTE_HOST" --remote-repo "$REMOTE_REPO" >"$LOG_PATH" 2>&1 </dev/null &
PID=$!
echo "$PID" > runs/LATEST_MONITOR_PID
sleep 1
if ! ps -p "$PID" >/dev/null 2>&1; then
  echo "vibe-monitor failed to start (pid=$PID is not alive)" >&2
  echo "---- log tail ----" >&2
  tail -n 80 "$LOG_PATH" >&2 || true
  exit 1
fi

if command -v curl >/dev/null 2>&1; then
  READY=0
  for _ in {1..20}; do
    if curl -fsS "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 0.5
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "vibe-monitor started but /healthz is not reachable yet: http://$HOST:$PORT/healthz" >&2
  fi
fi

echo "Started vibe-monitor pid=$PID"
echo "URL: http://$HOST:$PORT"
echo "Log: $ROOT/$LOG_PATH"
