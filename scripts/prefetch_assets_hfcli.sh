#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install -U huggingface_hub datasets >/dev/null

HF_HOME_DIR="${HF_HOME_DIR:-.cache/huggingface}"
PRIMARY_ENDPOINT="${PRIMARY_ENDPOINT:-https://hf-mirror.com}"
FALLBACK_ENDPOINT="${FALLBACK_ENDPOINT:-https://huggingface.co}"

MODELS_CSV="${MODELS_CSV-Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct}"
DATASETS_CSV="${DATASETS_CSV-EleutherAI/hendrycks_math,openai/gsm8k,tasksource/PRM800K,mbpp,openai_humaneval}"

SUMMARY_DIR="runs/.prefetch"
SUMMARY_TXT="${SUMMARY_DIR}/hfcli_summary.txt"
mkdir -p "${HF_HOME_DIR}" "${SUMMARY_DIR}" logs
: > "${SUMMARY_TXT}"

split_csv() {
  local text="$1"
  echo "${text}" | tr ',' '\n' | sed '/^[[:space:]]*$/d'
}

try_download() {
  local repo_type="$1"
  local repo_id="$2"

  local ep=""
  for ep in "${PRIMARY_ENDPOINT}" "${FALLBACK_ENDPOINT}"; do
    [[ -z "${ep}" ]] && continue
    export HF_ENDPOINT="${ep}"
    echo "[info] ${repo_type}:${repo_id} via ${ep}"
    if hf download "${repo_id}" --repo-type "${repo_type}" --cache-dir "${HF_HOME_DIR}" >/dev/null; then
      printf "%s\t%s\tok\t%s\n" "${repo_type}" "${repo_id}" "${ep}" >> "${SUMMARY_TXT}"
      return 0
    fi
    echo "[warn] ${repo_type}:${repo_id} failed via ${ep}"
  done

  printf "%s\t%s\tfailed\t%s\n" "${repo_type}" "${repo_id}" "${FALLBACK_ENDPOINT}" >> "${SUMMARY_TXT}"
  return 1
}

while IFS= read -r model; do
  try_download model "${model}" || true
done < <(split_csv "${MODELS_CSV}")

while IFS= read -r ds; do
  try_download dataset "${ds}" || true
done < <(split_csv "${DATASETS_CSV}")

echo "[info] materialize project datasets under data/"
python scripts/download_datasets.py \
  --math_train_size 8000 \
  --math_difficulty medium,hard \
  --eval_sets MATH500,GSM8K-hard \
  --include_prm800k_heldout \
  --seed 42 \
  --output_root data

python - <<'PY'
from pathlib import Path
import json
import time

summary_txt = Path("runs/.prefetch/hfcli_summary.txt")
rows = []
ok = 0
fail = 0
for line in summary_txt.read_text(encoding="utf-8").splitlines():
    parts = line.split("\t")
    if len(parts) != 4:
        continue
    repo_type, repo_id, status, endpoint = parts
    rows.append(
        {
            "repo_type": repo_type,
            "repo_id": repo_id,
            "status": status,
            "endpoint": endpoint,
        }
    )
    if status == "ok":
        ok += 1
    else:
        fail += 1

out = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ok_count": ok,
    "fail_count": fail,
    "items": rows,
}
Path("runs/.prefetch/hfcli_summary.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("[ok] wrote runs/.prefetch/hfcli_summary.json")
print(f"[ok] done: ok={ok} fail={fail}")
PY
