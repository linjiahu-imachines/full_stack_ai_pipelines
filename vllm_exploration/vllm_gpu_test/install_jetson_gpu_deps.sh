#!/usr/bin/env bash
# Install CUDA-enabled PyTorch for Jetson, then vLLM + test deps into vllm_gpu_test/venv.
# Run from this directory after creating the venv (see README).
#
# Env:
#   VLLM_TRY_JETSON_PYPI_FIRST=1  — try pypi.jetson-ai-lab.dev before download.pytorch.org (aarch64 only).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$SCRIPT_DIR/venv/bin/pip" ]]; then
  echo "No venv found. Create one first, e.g.:"
  echo "  python3 -m venv venv   # needs: sudo apt install python3.12-venv"
  echo "  or:  pip install --break-system-packages virtualenv && virtualenv venv"
  exit 1
fi

PIP="$SCRIPT_DIR/venv/bin/pip"
"$PIP" install -U pip wheel
# vLLM 0.19.x pins setuptools <81
"$PIP" install 'setuptools>=77.0.3,<81.0.0'

echo "Removing prior torch stack (CPU or CUDA) so pip can install a clean CUDA build…"
"$PIP" uninstall -y torch torchvision torchaudio 2>/dev/null || true

INSTALLED=0

try_jetson_index() {
  local name="$1" url="$2"
  echo ""
  echo "Trying PyTorch index: $name ($url)"
  if "$PIP" install --no-cache-dir torch torchvision torchaudio --index-url "$url"; then
    INSTALLED=1
  fi
}

try_official_cu130() {
  echo ""
  echo "Trying PyTorch official: cu130 (download.pytorch.org), pinned for vLLM 0.19.x"
  if "$PIP" install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url "https://download.pytorch.org/whl/cu130" \
    --extra-index-url "https://pypi.org/simple"; then
    INSTALLED=1
  fi
}

ARCH="$(uname -m)"

if [[ "$ARCH" == "aarch64" && "${VLLM_TRY_JETSON_PYPI_FIRST:-0}" != "1" ]]; then
  # Default on Jetson: official PyTorch has reliable DNS/CDN; jetson-ai-lab often missing in /etc/resolv.conf.
  try_official_cu130
  if [[ "$INSTALLED" -eq 0 ]]; then
    try_jetson_index "jetson-ai-lab jp7 cu130" "https://pypi.jetson-ai-lab.dev/jp7/cu130/"
  fi
  if [[ "$INSTALLED" -eq 0 ]]; then
    try_jetson_index "jetson-ai-lab jp6 cu128" "https://pypi.jetson-ai-lab.dev/jp6/cu128/"
  fi
  if [[ "$INSTALLED" -eq 0 ]]; then
    try_jetson_index "jetson-ai-lab jp6 cu126" "https://pypi.jetson-ai-lab.dev/jp6/cu126/"
  fi
else
  # Optional: Jetson PyPI first, or non-aarch64 fallback order
  try_jetson_index "jetson-ai-lab jp7 cu130" "https://pypi.jetson-ai-lab.dev/jp7/cu130/"
  if [[ "$INSTALLED" -eq 0 ]]; then
    try_jetson_index "jetson-ai-lab jp6 cu128" "https://pypi.jetson-ai-lab.dev/jp6/cu128/"
  fi
  if [[ "$INSTALLED" -eq 0 ]]; then
    try_jetson_index "jetson-ai-lab jp6 cu126" "https://pypi.jetson-ai-lab.dev/jp6/cu126/"
  fi
  if [[ "$INSTALLED" -eq 0 ]]; then
    try_official_cu130
  fi
fi

if [[ "$INSTALLED" -eq 0 ]]; then
  echo ""
  echo "Could not install CUDA PyTorch."
  echo "Install PyTorch manually per NVIDIA docs, then run:"
  echo "  $PIP install -r requirements.txt"
  echo "Docs: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html"
  exit 1
fi

echo ""
echo "Installing vLLM + test requirements (includes nvidia-cuda-runtime-cu12 for vLLM + cu130 torch)…"
"$PIP" install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Verifying environment…"
if ! "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/verify_gpu_env.py"; then
  echo ""
  echo "PyTorch is installed but CUDA is not visible to this process."
  echo "See: $SCRIPT_DIR/JETSON_GPU_ACCESS.md"
  exit 1
fi

echo ""
echo "Done. Run:  ./run_gpu_tests.sh compare"
