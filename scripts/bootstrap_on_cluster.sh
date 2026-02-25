#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ ! -f remote/cluster_env.sh ]]; then
  echo "[ERR] missing remote/cluster_env.sh" >&2
  exit 2
fi

# shellcheck disable=SC1091
source remote/cluster_env.sh
setup_cluster_env

choose_python() {
  local cand=""
  for cand in \
    python3 \
    /home/bingxing2/apps/miniforge3/24.1.2/bin/python3 \
    /home/bingxing2/apps/miniforge3/bin/python3 \
    "${HOME}/miniforge3/bin/python3" \
    "${HOME}/miniconda3/bin/python3" \
    python; do
    if ! command -v "${cand}" >/dev/null 2>&1; then
      continue
    fi
    if "${cand}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

if ! PY_BIN="$(choose_python)"; then
  echo "[ERR] python not found after environment setup" >&2
  echo "[ERR] need Python >= 3.10 for this project" >&2
  exit 2
fi

if [[ ! -f .venv/bin/activate ]]; then
  "${PY_BIN}" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
if [[ -f requirements.txt ]]; then
  python -m pip install -r requirements.txt
fi
# Install editable package only when requirements.txt does not already include it.
if [[ ! -f requirements.txt ]] || ! grep -Eq '^[[:space:]]*-e[[:space:]]+\.[[:space:]]*$' requirements.txt; then
  python -m pip install -e .
fi

echo "[ok] cluster bootstrap complete at ${ROOT}"
