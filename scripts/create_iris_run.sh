#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"          # smoke | formal
TOPIC_SLUG="${2:-iris-partial-id-evaluator-shift}"

if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: $0 [smoke|formal] [topic-slug]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${STAMP}_${TOPIC_SLUG}-${MODE}"
RUN_DIR="$ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$RUN_DIR/final" "$RUN_DIR/eval"

SMOKE_DEFAULT=1
if [[ "$MODE" == "formal" ]]; then
  SMOKE_DEFAULT=0
fi

cat > "$RUN_DIR/experiment.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="\$(cd "\${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="\${SCRIPT_DIR}"
RUN_ID="\$(basename "\${RUN_DIR}")"

SMOKE_MODE="\${SMOKE_MODE:-${SMOKE_DEFAULT}}"
SEED="\${SEED:-42}"

# backend controls rollout generation realism.
# smoke default -> synthetic
# formal default -> local_hf
ROLLOUT_BACKEND="\${ROLLOUT_BACKEND:-auto}"   # auto | synthetic | local_hf
ROLLOUT_MODEL="\${ROLLOUT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PY_BIN="\${PY_BIN:-python3}"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ "\${SMOKE_MODE}" == "1" ]]; then
  MATH_TRAIN_SIZE="\${MATH_TRAIN_SIZE:-320}"
  N_ROLLOUTS="\${N_ROLLOUTS:-2}"
  MAX_NEW_TOKENS="\${MAX_NEW_TOKENS:-96}"
  UPDATES_MAIN="\${UPDATES_MAIN:-70}"
  UPDATES_RAW="\${UPDATES_RAW:-55}"
  UPDATES_GPESS="\${UPDATES_GPESS:-55}"
  PARTIAL_MODE="\${PARTIAL_MODE:-hard}"
  JUDGE_PERSONAS="\${JUDGE_PERSONAS:-3}"
  [[ "\${ROLLOUT_BACKEND}" == "auto" ]] && ROLLOUT_BACKEND="synthetic"
else
  # keep formal defaults practical for 12h wall-clock under local_hf single-process rollout.
  MATH_TRAIN_SIZE="\${MATH_TRAIN_SIZE:-1800}"
  N_ROLLOUTS="\${N_ROLLOUTS:-2}"
  MAX_NEW_TOKENS="\${MAX_NEW_TOKENS:-128}"
  UPDATES_MAIN="\${UPDATES_MAIN:-320}"
  UPDATES_RAW="\${UPDATES_RAW:-220}"
  UPDATES_GPESS="\${UPDATES_GPESS:-220}"
  PARTIAL_MODE="\${PARTIAL_MODE:-soft}"
  JUDGE_PERSONAS="\${JUDGE_PERSONAS:-4}"
  [[ "\${ROLLOUT_BACKEND}" == "auto" ]] && ROLLOUT_BACKEND="local_hf"
fi

if [[ "\${SMOKE_MODE}" == "1" ]]; then
  LOCAL_HF_REQUIRE_CUDA="\${LOCAL_HF_REQUIRE_CUDA:-0}"
else
  LOCAL_HF_REQUIRE_CUDA="\${LOCAL_HF_REQUIRE_CUDA:-1}"
fi
LOCAL_HF_FALLBACK_TO_SYNTHETIC="\${LOCAL_HF_FALLBACK_TO_SYNTHETIC:-0}"

pick_python_with_ml_runtime() {
  local cand
  # keep requested PY_BIN if it already has required modules.
  if "\${PY_BIN}" - <<'PY' >/dev/null 2>&1
import torch, transformers
PY
  then
    return 0
  fi

  for cand in \
    /home/bingxing2/apps/miniforge3/24.1.2/bin/python3 \
    /home/bingxing2/apps/miniforge3/bin/python3 \
    "\${HOME}/miniforge3/bin/python3" \
    "\${HOME}/miniconda3/bin/python3"; do
    if [[ -x "\${cand}" ]] && "\${cand}" - <<'PY' >/dev/null 2>&1
import torch, transformers
PY
    then
      PY_BIN="\${cand}"
      return 0
    fi
  done
  return 1
}

local_hf_cuda_available() {
  "\${PY_BIN}" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

START_TS="\$(date -Iseconds)"
START_EPOCH="\$(date +%s)"

write_results_json() {
  local rc="\$1"
  local end_ts end_epoch elapsed
  end_ts="\$(date -Iseconds)"
  end_epoch="\$(date +%s)"
  elapsed=\$(( end_epoch - START_EPOCH ))

  EXIT_CODE="\${rc}" RUN_DIR="\${RUN_DIR}" RUN_ID="\${RUN_ID}" START_TS="\${START_TS}" END_TS="\${end_ts}" ELAPSED_SEC="\${elapsed}" ROLLOUT_BACKEND="\${ROLLOUT_BACKEND}" ROLLOUT_MODEL="\${ROLLOUT_MODEL}" PY_BIN_USED="\${PY_BIN}" \
  "\${PY_BIN}" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
run_id = os.environ["RUN_ID"]
rc = int(os.environ["EXIT_CODE"])

metrics_path = run_dir / "eval" / "summary.json"
metrics = {}
if metrics_path.exists():
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}

obj = {
    "run_id": run_id,
    "status": "success" if rc == 0 else "failed",
    "start_time": os.environ.get("START_TS"),
    "end_time": os.environ.get("END_TS"),
    "elapsed_sec": int(os.environ.get("ELAPSED_SEC", "0")),
    "runtime": {
        "rollout_backend": os.environ.get("ROLLOUT_BACKEND"),
        "rollout_model": os.environ.get("ROLLOUT_MODEL"),
        "python": os.environ.get("PY_BIN_USED"),
    },
    "artifacts": {
        "rollouts": str(run_dir / "rollouts.jsonl"),
        "step_prefs": str(run_dir / "step_prefs.jsonl"),
        "iris_pairs": str(run_dir / "iris_pairs.jsonl"),
        "eval_report": str(run_dir / "eval" / "report.json"),
        "final_report": str(run_dir / "final" / "report.json"),
    },
    "eval_summary": metrics,
}
(run_dir / "results.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
PY
}

trap 'rc=\$?; write_results_json "\$rc"' EXIT

cd "\${PROJECT_DIR}"
mkdir -p "\${RUN_DIR}/logs" "\${RUN_DIR}/eval" "\${RUN_DIR}/final"

if [[ "\${ROLLOUT_BACKEND}" == "local_hf" ]]; then
  if ! pick_python_with_ml_runtime; then
    echo "[ERR] local_hf backend requested but no python with torch+transformers found." >&2
    exit 2
  fi
  if [[ "\${LOCAL_HF_REQUIRE_CUDA}" == "1" ]] && ! local_hf_cuda_available; then
    if [[ "\${LOCAL_HF_FALLBACK_TO_SYNTHETIC}" == "1" ]]; then
      echo "[warn] local_hf requested but CUDA unavailable; fallback to synthetic backend." >&2
      ROLLOUT_BACKEND="synthetic"
    else
      echo "[ERR] local_hf backend requires CUDA, but torch.cuda.is_available()=False." >&2
      echo "[ERR] set LOCAL_HF_FALLBACK_TO_SYNTHETIC=1 to auto-fallback." >&2
      exit 3
    fi
  fi
fi

echo "[info] RUN_ID=\${RUN_ID} smoke=\${SMOKE_MODE}"
echo "[info] backend=\${ROLLOUT_BACKEND} model=\${ROLLOUT_MODEL} python=\${PY_BIN}"

echo "[info] preparing datasets"
"\${PY_BIN}" scripts/download_datasets.py \
  --math_train_size "\${MATH_TRAIN_SIZE}" \
  --math_difficulty medium,hard \
  --eval_sets MATH500,GSM8K-hard \
  --include_prm800k_heldout

"\${PY_BIN}" scripts/build_structured_noise_features.py \
  --input data/math/train.jsonl \
  --output data/math/train.features.parquet \
  --features step_index,repetition,length,difficulty,source

"\${PY_BIN}" scripts/make_judge_personas.py \
  --output configs/judges.yaml \
  --anchor verifier \
  --llm_personas "\${JUDGE_PERSONAS}"

echo "[info] generating rollouts"
if [[ "\${ROLLOUT_BACKEND}" == "local_hf" ]]; then
  "\${PY_BIN}" scripts/generate_rollouts.py \
    --input data/math/train.jsonl \
    --n_rollouts "\${N_ROLLOUTS}" \
    --max_new_tokens "\${MAX_NEW_TOKENS}" \
    --backend local_hf \
    --local_model "\${ROLLOUT_MODEL}" \
    --device auto \
    --dtype auto \
    --output "\${RUN_DIR}/rollouts.jsonl"
else
  "\${PY_BIN}" scripts/generate_rollouts.py \
    --input data/math/train.jsonl \
    --n_rollouts "\${N_ROLLOUTS}" \
    --max_new_tokens "\${MAX_NEW_TOKENS}" \
    --backend synthetic \
    --output "\${RUN_DIR}/rollouts.jsonl"
fi

echo "[info] judging rollout step pairs"
"\${PY_BIN}" scripts/judge_step_pairs.py \
  --rollouts "\${RUN_DIR}/rollouts.jsonl" \
  --judges configs/judges.yaml \
  --features data/math/train.features.parquet \
  --output "\${RUN_DIR}/step_prefs.jsonl"

echo "[info] building partial-ID lower-bound pairs"
"\${PY_BIN}" scripts/build_iris_partial_id_pairs.py \
  --input "\${RUN_DIR}/step_prefs.jsonl" \
  --output "\${RUN_DIR}/iris_pairs.jsonl" \
  --summary "\${RUN_DIR}/iris_pairs.summary.json" \
  --mode "\${PARTIAL_MODE}"

CKPT_ROOT="checkpoints/\${RUN_ID}"
CKPT_MAIN="\${CKPT_ROOT}/iris_partial_id"
CKPT_RAW="\${CKPT_ROOT}/raw_dpo"
CKPT_GPESS="\${CKPT_ROOT}/generic_pess"
mkdir -p "\${CKPT_MAIN}" "\${CKPT_RAW}" "\${CKPT_GPESS}"

echo "[info] training main + baselines"
"\${PY_BIN}" train/train_iris_shift_dpo.py \
  --pairs "\${RUN_DIR}/iris_pairs.jsonl" \
  --target q_l \
  --updates "\${UPDATES_MAIN}" \
  --output_dir "\${CKPT_MAIN}" \
  --seed "\${SEED}"

"\${PY_BIN}" train/train_iris_shift_dpo.py \
  --pairs "\${RUN_DIR}/iris_pairs.jsonl" \
  --target raw \
  --updates "\${UPDATES_RAW}" \
  --output_dir "\${CKPT_RAW}" \
  --seed "\${SEED}" > "\${RUN_DIR}/logs/train_raw.log" 2>&1 &
PID_RAW=\$!

"\${PY_BIN}" train/train_iris_shift_dpo.py \
  --pairs "\${RUN_DIR}/iris_pairs.jsonl" \
  --target q_generic \
  --updates "\${UPDATES_GPESS}" \
  --output_dir "\${CKPT_GPESS}" \
  --seed "\${SEED}" > "\${RUN_DIR}/logs/train_gpess.log" 2>&1 &
PID_GPESS=\$!

wait "\${PID_RAW}"
wait "\${PID_GPESS}"

echo "[info] running shift evaluation"
"\${PY_BIN}" eval/run_iris_shift_eval.py \
  --models "iris=\${CKPT_MAIN},raw=\${CKPT_RAW},generic_pess=\${CKPT_GPESS}" \
  --reasoning_sets MATH500,GSM8K-hard \
  --output_dir "\${RUN_DIR}/eval" \
  --seed "\${SEED}"

cp "\${RUN_DIR}/eval/report.json" "\${RUN_DIR}/final/report.json"
cp "\${RUN_DIR}/eval/summary.json" "\${RUN_DIR}/final/summary.json"

echo "[ok] IRIS experiment completed: \${RUN_DIR}"
EOF

chmod +x "$RUN_DIR/experiment.sh"

cat > "$RUN_DIR/status.json" <<EOF
{
  "run_id": "$RUN_ID",
  "topic": "IRIS partial-ID under evaluator shift",
  "state": "ready",
  "stage": "implementation",
  "step": 1,
  "total_steps": 1,
  "message": "experiment materialized and ready for remote submit",
  "updated_at": "$(date -Iseconds)"
}
EOF

echo "$RUN_ID" > "$ROOT/runs/LATEST_RUN"
echo "$RUN_ID"
