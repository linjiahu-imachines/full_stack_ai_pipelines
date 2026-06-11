#!/bin/bash
# Quick entrypoint for vLLM CPU test.

set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_CPU_KVCACHE_SPACE=4

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

CPU_PYTHON="${VLLM_CPU_PYTHON:-}"
if [ -z "${CPU_PYTHON}" ]; then
  if [ -x "${PROJECT_ROOT}/vllm_test/venv/bin/python3" ]; then
    CPU_PYTHON="${PROJECT_ROOT}/vllm_test/venv/bin/python3"
  elif [ -x "${PROJECT_ROOT}/vllm_cpu_venv/bin/python3" ]; then
    CPU_PYTHON="${PROJECT_ROOT}/vllm_cpu_venv/bin/python3"
  else
    echo "No CPU Python interpreter found."
    echo "Set VLLM_CPU_PYTHON or create one of:"
    echo "  ${PROJECT_ROOT}/vllm_test/venv/bin/python3"
    echo "  ${PROJECT_ROOT}/vllm_cpu_venv/bin/python3"
    exit 1
  fi
fi

exec "${CPU_PYTHON}" -u "${SCRIPT_DIR}/test2_vllm_cpu.py" 2>&1
