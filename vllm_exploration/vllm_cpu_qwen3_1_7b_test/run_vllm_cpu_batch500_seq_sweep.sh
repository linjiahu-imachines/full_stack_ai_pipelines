#!/bin/bash
# vLLM CPU only: Qwen3-1.7B, batch 500, max_num_seqs sweep (default 64,128,256).
# Example: --max-num-seqs-list 32,64
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${CPU_TEST_PYTHON:-${VLLM_CPU_PYTHON:-}}"
if [ -z "${PYTHON_BIN}" ] && [ -x "${PROJECT_ROOT}/vllm_cpu_venv/bin/python3" ]; then
  PYTHON_BIN="${PROJECT_ROOT}/vllm_cpu_venv/bin/python3"
fi
if [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; then
  echo "Set CPU_TEST_PYTHON or ensure ${PROJECT_ROOT}/vllm_cpu_venv exists."
  exit 1
fi
exec env -u VLLM_CPU_PYTHON "${PYTHON_BIN}" "${SCRIPT_DIR}/run_vllm_cpu_batch500_seq_sweep.py" "$@"
