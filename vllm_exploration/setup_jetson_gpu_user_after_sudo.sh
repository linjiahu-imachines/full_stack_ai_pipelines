#!/usr/bin/env bash
# Run once after you receive sudo: GPU groups + Python venv packages.
# You must LOG OUT completely (or reboot) afterward so new groups apply.

set -euo pipefail

echo "=== 1. Add your user to video + render (GPU device access) ==="
if sudo usermod -aG video,render "$USER"; then
  echo "OK: $USER added to video,render"
else
  echo "FAILED: sudo cannot run usermod (limited or deny-all sudoers)."
  echo "  Ask admin to run as root:  usermod -aG video,render $USER"
  echo "  Then reboot or log out completely."
  echo ""
fi

echo "=== 2. Python venv + headers (vLLM/Triton JIT compiles extensions; needs Python.h) ==="
if sudo apt-get update -qq && sudo apt-get install -y python3.12-venv python3.12-dev python3-pip build-essential; then
  echo "OK: python3.12-venv, python3.12-dev, python3-pip, build-essential"
else
  echo "FAILED: sudo apt-get not allowed or network error. Ask admin to install:"
  echo "  python3.12-venv python3.12-dev python3-pip build-essential"
fi

echo ""
echo "============================================================"
echo "NEXT STEPS (required — groups do not apply to old sessions):"
echo "  1) Log out of the desktop completely, or:  sudo reboot"
echo "  2) Log back in, then run:  groups"
echo "     You should see: video (and render)"
echo "  3) GPU + vLLM stack:"
echo "     cd $(dirname "$0")/vllm_gpu_test"
echo "     bash install_jetson_gpu_deps.sh"
echo "     ./run_gpu_tests.sh verify"
echo "     ./run_gpu_tests.sh transformers-cpu-gpu   # CPU + GPU Transformers"
echo "     ./run_gpu_tests.sh compare                # vLLM vs Transformers (GPU)"
echo "============================================================"
