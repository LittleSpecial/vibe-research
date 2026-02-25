#!/bin/bash
#SBATCH --job-name=vibe_research
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/vibe_research_%j.out
#SBATCH --error=logs/vibe_research_%j.err
#SBATCH --export=ALL

set -euo pipefail

RUN_DIR_REL="${1:-}"
if [[ -z "$RUN_DIR_REL" ]]; then
  echo "Usage: sbatch remote/slurm_run_experiment.sh runs/<RUN_ID>" >&2
  exit 2
fi

CLUSTER_ENV_SH=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/remote/cluster_env.sh" ]]; then
  CLUSTER_ENV_SH="${SLURM_SUBMIT_DIR}/remote/cluster_env.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "${SCRIPT_DIR}/cluster_env.sh" ]]; then
    CLUSTER_ENV_SH="${SCRIPT_DIR}/cluster_env.sh"
  fi
fi

if [[ -z "${CLUSTER_ENV_SH}" ]]; then
  echo "[ERR] Cannot locate remote/cluster_env.sh" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CLUSTER_ENV_SH}"
setup_cluster_env

mkdir -p logs
if [[ ! -f .venv/bin/activate ]]; then
  echo "[ERR] missing .venv. Run scripts/remote_bootstrap.sh before submitting jobs." >&2
  exit 2
fi
# shellcheck disable=SC1091
source .venv/bin/activate

EXP_SH="$RUN_DIR_REL/experiment.sh"
if [[ ! -f "$EXP_SH" ]]; then
  echo "[ERR] missing experiment script: $EXP_SH" >&2
  exit 2
fi

bash "$EXP_SH"
