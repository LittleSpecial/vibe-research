#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

ensure_module_cmd() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi
  source /etc/profile >/dev/null 2>&1 || true
  source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
  command -v module >/dev/null 2>&1
}

load_first_available_module() {
  local group="$1"
  shift

  local mod=""
  for mod in "$@"; do
    if module load "${mod}" >/dev/null 2>&1; then
      echo "[env] loaded ${group} module: ${mod}"
      return 0
    fi
  done

  echo "[env] failed to load any ${group} module from: $*" >&2
  return 1
}

setup_cluster_env() {
  if ! ensure_module_cmd; then
    echo "[env] module command unavailable after sourcing /etc/profile*" >&2
    return 2
  fi

  module purge

  load_first_available_module "cuda" \
    compilers/cuda/12.2 \
    compilers/cuda/12.4 \
    compilers/cuda/12.8 \
    compilers/cuda/12.3 \
    compilers/cuda/12.1 \
    cuda/11.7.0 \
    cuda/11.6.0 \
    cuda/11.2.0 \
    cuda/11.0.3

  # Torch/transformers on this cluster may depend on explicit cudnn module.
  if ! load_first_available_module "cudnn" \
    cudnn/8.9.5.29_cuda12.x \
    cudnn/8.9.4.25_cuda12.x \
    cudnn/8.8.1.3_cuda12.x \
    cudnn/8.6.0.163_cuda11.x \
    cudnn/8.4.0.27_cuda11.x; then
    echo "[env] cudnn module not found; torch CUDA runtime may be unavailable" >&2
  fi

  load_first_available_module "gcc" \
    compilers/gcc/11.3.0 \
    compilers/gcc/12.2.0 \
    compilers/gcc/10.3.1 \
    compilers/gcc/9.3.0

  # Optional: many clusters already provide python via miniforge path.
  if ! load_first_available_module "python" python/3.10 python/3.11 python/3.9; then
    echo "[env] python module not found; keep current python from PATH" >&2
  fi

  # Keep a Python >=3.10 runtime for local venv/bootstrap steps.
  local py_ok="0"
  if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"; then
      py_ok="1"
    fi
  fi
  if [[ "${py_ok}" != "1" ]]; then
    local cand_bin=""
    for cand_bin in \
      /home/bingxing2/apps/miniforge3/24.1.2/bin \
      /home/bingxing2/apps/miniforge3/bin \
      "${HOME}/miniforge3/bin" \
      "${HOME}/miniconda3/bin"; do
      if [[ -x "${cand_bin}/python3" ]]; then
        export PATH="${cand_bin}:${PATH}"
        break
      fi
    done
  fi

  local arch
  arch="$(uname -m)"
  if [[ "${arch}" != "aarch64" ]]; then
    echo "[env] unexpected arch=${arch}, expected aarch64" >&2
    return 2
  fi

  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path cuda_root
    nvcc_path="$(command -v nvcc)"
    cuda_root="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
    export CUDA_HOME="${cuda_root}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    if [[ -d "${CUDA_HOME}/lib64" ]]; then
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    fi
  fi

  echo "[env] python=$(command -v python3 || command -v python || true)"
  if command -v python3 >/dev/null 2>&1; then
    python3 --version || true
  elif command -v python >/dev/null 2>&1; then
    python --version || true
  fi
  echo "[env] nvcc=$(command -v nvcc || echo not-found)"
  echo "[env] CUDA_HOME=${CUDA_HOME:-unset}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  setup_cluster_env
fi
