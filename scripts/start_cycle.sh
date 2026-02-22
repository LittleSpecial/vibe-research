#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run ./scripts/bootstrap_local.sh first." >&2
  exit 2
fi
source .venv/bin/activate

CONFIG="${1:-configs/local.toml}"
TOPIC="${2:-}"
MODE="${3:-interactive}"  # interactive|noninteractive
AGENT_COUNT="${4:-0}"      # 0 => use config default
TIMEOUT="${5:-0}"          # feedback timeout; 0 => wait forever

mkdir -p runs/.orchestrator_logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="runs/.orchestrator_logs/run_cycle_${TS}.log"
PREV_RUN_ID=""
if [[ -f runs/LATEST_RUN ]]; then
  PREV_RUN_ID="$(cat runs/LATEST_RUN || true)"
fi

CMD=(vibe-research run-cycle --config "$CONFIG" --agent-count "$AGENT_COUNT" --feedback-timeout "$TIMEOUT")
if [[ -n "$TOPIC" ]]; then
  CMD+=(--topic "$TOPIC")
fi
if [[ "$MODE" == "interactive" ]]; then
  CMD+=(--interactive)
fi

"${CMD[@]}" >"$LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > runs/LATEST_PID

echo "Started run-cycle pid=$PID"
echo "Log: $ROOT/$LOG_PATH"

echo "Waiting for run id..."
for _ in {1..20}; do
  if [[ -f runs/LATEST_RUN ]]; then
    RUN_ID="$(cat runs/LATEST_RUN)"
    if [[ -n "$RUN_ID" ]] && [[ "$RUN_ID" != "$PREV_RUN_ID" ]]; then
      echo "RUN_ID=$RUN_ID"
      break
    fi
  fi
  sleep 1
done

echo "Use monitor:" 
echo "  $ROOT/scripts/monitor_local_fars.sh ${RUN_ID:-<RUN_ID>}"
