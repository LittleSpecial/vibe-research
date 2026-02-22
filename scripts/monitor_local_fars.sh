#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run ./scripts/bootstrap_local.sh first." >&2
  exit 2
fi
source .venv/bin/activate

RUN_ID="${1:-}"
if [[ -z "$RUN_ID" ]]; then
  if [[ -f runs/LATEST_RUN ]]; then
    RUN_ID="$(cat runs/LATEST_RUN)"
  else
    echo "Usage: $0 <RUN_ID>" >&2
    exit 2
  fi
fi

RUN_DIR="runs/$RUN_ID"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run dir not found: $RUN_DIR" >&2
  exit 2
fi

while true; do
  if [[ -t 1 ]]; then
    clear
  fi
  echo "==== Vibe Research Local Monitor ===="
  echo "Time: $(date '+%F %T')"
  echo "Run:  $RUN_ID"
  echo

  if [[ -f runs/LATEST_PID ]]; then
    PID="$(cat runs/LATEST_PID || true)"
    if [[ -n "$PID" ]] && ps -p "$PID" >/dev/null 2>&1; then
      echo "Orchestrator PID: $PID (alive)"
    else
      echo "Orchestrator PID: ${PID:-unknown} (not running)"
    fi
  fi
  echo

  vibe-research watch-run --run-dir "$RUN_DIR" || true

  echo
  echo "--- progress tail ---"
  if [[ -f "$RUN_DIR/progress.log" ]]; then
    tail -n 12 "$RUN_DIR/progress.log"
  else
    echo "progress.log not found"
  fi

  echo
  echo "--- feedback files ---"
  ls -1 "$RUN_DIR/feedback" 2>/dev/null || echo "feedback dir not ready"

  sleep 2
done
