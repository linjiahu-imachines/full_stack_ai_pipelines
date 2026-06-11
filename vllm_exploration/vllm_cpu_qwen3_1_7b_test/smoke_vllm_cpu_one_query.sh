#!/bin/bash
# Single-query vLLM smoke on CPU. Requires a vLLM wheel whose version string
# contains "cpu" (see vllm.platforms); a standard CUDA-only install will not
# run inference on CPU even when VLLM_TARGET_DEVICE=cpu.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Prefer CPU_TEST_PYTHON — do not use VLLM_* for the interpreter path (vLLM logs
# unknown VLLM_* env vars). Legacy: VLLM_CPU_PYTHON.
PYTHON_BIN="${CPU_TEST_PYTHON:-${VLLM_CPU_PYTHON:-}}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x "${PROJECT_ROOT}/vllm_cpu_venv/bin/python3" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/vllm_cpu_venv/bin/python3"
  elif [ -x "${PROJECT_ROOT}/vllm_test/venv/bin/python3" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/vllm_test/venv/bin/python3"
  elif [ -x "${PROJECT_ROOT}/vllm_gpu_test/venv/bin/python3" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/vllm_gpu_test/venv/bin/python3"
  fi
fi

if [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; then
  echo "No Python interpreter found. Set CPU_TEST_PYTHON or create e.g."
  echo "  ${PROJECT_ROOT}/vllm_cpu_venv/bin/python3"
  exit 1
fi

echo "Using Python: ${PYTHON_BIN}"

unset VLLM_CPU_PYTHON 2>/dev/null || true

"${PYTHON_BIN}" <<'PY'
import importlib.metadata as m
v = m.version("vllm")
if "cpu" not in v.lower():
    print()
    print("CPU-only vLLM smoke cannot run: this interpreter has a non-CPU vLLM build.")
    print(f"  importlib.metadata.version('vllm') == {v!r}")
    print("Install a CPU-capable vLLM (wheel whose version contains 'cpu'), then re-run.")
    print("Docs: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/")
    raise SystemExit(2)
print(f"OK: CPU vLLM build detected ({v})")
PY

echo "Running one query (facebook/opt-125m, batch 1, max-new-tokens 8)..."
exec env -u VLLM_CPU_PYTHON "${PYTHON_BIN}" "${SCRIPT_DIR}/test_qwen3_vllm_cpu.py" \
  --model facebook/opt-125m \
  --batch-size 1 \
  --max-new-tokens 8 \
  --max-num-seqs 8
