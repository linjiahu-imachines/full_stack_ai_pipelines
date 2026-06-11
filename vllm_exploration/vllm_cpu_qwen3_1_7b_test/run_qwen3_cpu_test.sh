#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

PYTHON_BIN="${CPU_TEST_PYTHON:-${VLLM_CPU_PYTHON:-}}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x "${PROJECT_ROOT}/vllm_cpu_venv/bin/python3" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/vllm_cpu_venv/bin/python3"
  elif [ -x "${PROJECT_ROOT}/vllm_test/venv/bin/python3" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/vllm_test/venv/bin/python3"
  else
    echo "No CPU test interpreter found."
    echo "Set CPU_TEST_PYTHON or create one of:"
    echo "  ${PROJECT_ROOT}/vllm_cpu_venv/bin/python3"
    echo "  ${PROJECT_ROOT}/vllm_test/venv/bin/python3"
    exit 1
  fi
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Qwen3-1.7B CPU sweep: batch sizes 1, 500, 1000, 2000 (without + with vLLM)"

exec env -u VLLM_CPU_PYTHON "${PYTHON_BIN}" "${SCRIPT_DIR}/test_qwen3_cpu_compare.py" \
  --batch-sizes "1,500,1000,2000" \
  --max-new-tokens 30 \
  --vllm-max-num-seqs 2048
