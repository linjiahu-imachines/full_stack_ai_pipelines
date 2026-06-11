#!/usr/bin/env bash
# Source from bash:  source /home/linhu/projects/sglang_exploration/activate_sglang_env.sh
# Sets PATH to sglang_venv and extends LD_LIBRARY_PATH; optionally CUDA_HOME when a real toolkit is present.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_EXPLORATION_ROOT="$ROOT"
VENV="$ROOT/sglang_venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing $VENV — run $ROOT/setup_sglang_env.sh first." >&2
  return 2 2>/dev/null || exit 2
fi

export PATH="$VENV/bin:$PATH"
LIBS=$(find "$VENV/lib/python3.12/site-packages/nvidia" -type d -name lib 2>/dev/null | paste -sd: -)
if [[ -z "${LIBS:-}" ]]; then
  LIBS=$(find "$VENV/lib" -path "*/site-packages/nvidia/*/lib" -type d 2>/dev/null | paste -sd: -)
fi
if [[ -n "$LIBS" ]]; then
  export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Prefer a real toolkit CUDA_HOME when available.
if [[ -z "${CUDA_HOME:-}" && -z "${CUDA_PATH:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    _nvcc=$(command -v nvcc)
    export CUDA_HOME="$(cd "$(dirname "$_nvcc")/.." && pwd)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  else
    for _d in /usr/local/cuda-*; do
      if [[ -x "${_d}/bin/nvcc" ]]; then
        export CUDA_HOME="$_d"
        break
      fi
    done
  fi
fi

# deep_gemm asserts if CUDA_HOME is unset. On Jetson images without nvcc/toolkit, /usr keeps
# deep_gemm importable; RoPE JIT is already redirected to torch-native fallback in this project.
if [[ -z "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME=/usr
fi
