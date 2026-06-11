#!/usr/bin/env bash

# Script to run vLLM GPU tests (x86 discrete GPU or Jetson with CUDA PyTorch).

echo "=================================="
echo "vLLM GPU Testing Suite"
echo "=================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "$SCRIPT_DIR/venv/bin/python3" ]]; then
  PYTHON="$SCRIPT_DIR/venv/bin/python3"
elif [[ -f "$SCRIPT_DIR/venv/bin/python" ]]; then
  PYTHON="$SCRIPT_DIR/venv/bin/python"
else
  echo "Error: Virtual environment not found at ${SCRIPT_DIR}/venv"
  echo "Create it, then on Jetson run:  bash install_jetson_gpu_deps.sh"
  echo "Or:  python3 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi

# Modes that do not require working CUDA (Transformers CPU vs GPU runs CPU always).
SKIP_VERIFY=0
case "${1:-compare}" in
  transformers-cpu-gpu) SKIP_VERIFY=1 ;;
esac

if [[ "$SKIP_VERIFY" -eq 0 ]]; then
  if ! "$PYTHON" "$SCRIPT_DIR/verify_gpu_env.py"; then
    echo ""
    echo "If torch shows +cu130 but CUDA is still false, fix GPU device access (groups / reboot):"
    echo "  see $SCRIPT_DIR/JETSON_GPU_ACCESS.md"
    echo "Reinstall stack:"
    echo "  bash $SCRIPT_DIR/install_jetson_gpu_deps.sh"
    exit 1
  fi
fi

echo ""

case "${1:-compare}" in
  transformers-cpu-gpu)
    echo "Transformers only: CPU vs GPU (no vLLM; GPU skipped if no CUDA)..."
    "$PYTHON" "$SCRIPT_DIR/test_transformers_cpu_vs_gpu.py"
    ;;
  with)
    echo "Running tests WITH vLLM on GPU..."
    "$PYTHON" test_with_vllm_gpu.py
    ;;
  without)
    echo "Running tests WITHOUT vLLM (transformers only) on GPU..."
    "$PYTHON" test_without_vllm_gpu.py
    ;;
  compare)
    echo "Running full GPU comparison tests..."
    "$PYTHON" test_gpu_comparison.py
    ;;
  verify)
    echo "Environment OK (see above)."
    ;;
  *)
    echo "Usage: ./run_gpu_tests.sh [with|without|compare|verify|transformers-cpu-gpu]"
    echo ""
    echo "  with                    - Run tests WITH vLLM on GPU"
    echo "  without                 - Run tests WITHOUT vLLM (transformers only) on GPU"
    echo "  compare                 - Full vLLM vs Transformers comparison (default)"
    echo "  verify                  - Only check torch CUDA + vLLM import"
    echo "  transformers-cpu-gpu    - Transformers only: benchmark CPU then GPU (no vLLM)"
    exit 1
    ;;
esac
