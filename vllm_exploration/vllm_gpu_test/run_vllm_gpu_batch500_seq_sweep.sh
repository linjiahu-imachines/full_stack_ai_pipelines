#!/bin/bash
# vLLM on GPU: Qwen3-1.7B, batch 500, max_num_seqs sweep (default 64,128,256).
# Example low-concurrency rerun: --max-num-seqs-list 32,64
# Requires CUDA vLLM venv: vllm_gpu_test/venv (see install_jetson_gpu_deps.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${GPU_VLLM_PYTHON:-${GPU_TEST_PYTHON:-}}"
if [ -z "${PYTHON_BIN}" ] && [ -x "${SCRIPT_DIR}/venv/bin/python3" ]; then
  PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python3"
fi
if [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; then
  echo "Set GPU_VLLM_PYTHON or ensure ${SCRIPT_DIR}/venv exists (install_jetson_gpu_deps.sh)."
  exit 1
fi
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_vllm_gpu_batch500_seq_sweep.py" "$@"
