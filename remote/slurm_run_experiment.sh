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

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile || true
  source /etc/profile.d/modules.sh || true
fi

module purge || true
module load cuda/12.2 || true
module load gcc/11 || true
module load python/3.10 || true

arch="$(uname -m)"
if [[ "$arch" != "aarch64" ]]; then
  echo "Unexpected architecture: $arch (expected aarch64)" >&2
  exit 2
fi

mkdir -p logs
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python -m vibe_research.cli run-experiment --run-dir "$RUN_DIR_REL"
