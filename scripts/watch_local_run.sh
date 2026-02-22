#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
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

vibe-research watch-run --run-dir "runs/$RUN_ID" --follow --until-done
