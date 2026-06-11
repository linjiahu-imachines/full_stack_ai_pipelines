#!/usr/bin/env bash
# One-shot: create sglang_venv, install PyTorch cu130 + SGLang 0.5.9 + Jetson/Thor workarounds.
# Run from repo root:  ./setup_sglang_env.sh
#
# Requires: ``virtualenv`` (Debian images without python3-venv use ~/.local/bin/virtualenv).
# Optional: sudo apt install cuda-toolkit (only if you replace the torch_memory_saver stub with the real wheel).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/sglang_venv"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  if command -v virtualenv >/dev/null 2>&1; then
    echo "Creating venv with virtualenv at $VENV …"
    virtualenv -p python3 "$VENV"
  else
    echo "Need virtualenv (python3 -m venv failed on this image). Install with:"
    echo "  python3 -m pip install --user virtualenv"
    exit 1
  fi
fi

"$PIP" install -U pip wheel 'setuptools>=77.0.3,<81.0.0'

echo "Installing PyTorch 2.9.1+cu130 (matches SGLang 0.5.9 pin) …"
"$PIP" uninstall -y torch torchvision torchaudio 2>/dev/null || true
"$PIP" install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url "https://download.pytorch.org/whl/cu130" \
  --extra-index-url "https://pypi.org/simple"

echo "Installing CUDA 12 user-space libs (sgl_kernel / Triton look for .so.12 at runtime) …"
"$PIP" install \
  'nvidia-cuda-runtime-cu12>=12.4,<13' \
  'nvidia-cuda-nvrtc-cu12' \
  'nvidia-cublas-cu12' \
  'nvidia-cuda-nvcc-cu12'

echo "Installing local torch_memory_saver stub (real package needs system CUDA headers to build) …"
"$PIP" install "$ROOT/packaging/stub_torch_memory_saver"

echo "Installing SGLang and dependencies …"
"$PIP" install 'sglang==0.5.9'

echo "Installing venv .pth RoPE bootstrap (SGLang workers ignore venv sitecustomize when /usr/lib/sitecustomize exists) …"
PY_SITE="$("$PY" -c "import sysconfig; print(sysconfig.get_path('purelib'))")"
# A line starting with "import" is executed by site (see site.addpackage); runpy loads our hook.
printf '%s\n' "import runpy; runpy.run_path(r'''$ROOT/sglang_test/pth_rope_bootstrap.py''')" >"$PY_SITE/_sglang_exploration_rope.pth"
rm -f "$PY_SITE/sitecustomize.py"

echo ""
echo "Verifying (with Jetson library path) …"
# shellcheck source=/dev/null
source "$ROOT/activate_sglang_env.sh"
"$PY" -c "import torch; import sglang as sgl; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('sglang', getattr(sgl,'__version__','?'))"

echo ""
echo "Done."
echo "  source $ROOT/activate_sglang_env.sh"
echo "  python $ROOT/sglang_test/run_offline_batch_experiment.py --device cuda --model Qwen/Qwen3-1.7B --batch-size 1 --max-new-tokens 16"
