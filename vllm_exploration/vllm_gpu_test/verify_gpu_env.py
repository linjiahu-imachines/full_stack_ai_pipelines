#!/usr/bin/env python3
"""Exit 0 if CUDA PyTorch + vLLM import work; else exit 1 with a short message."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError as e:
        print("FAIL: torch not installed:", e, file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        hint = (
            "build is CPU-only (+cpu) — install a CUDA wheel (see install_jetson_gpu_deps.sh)."
            if "+cpu" in torch.__version__
            else (
                "PyTorch is CUDA-enabled but the driver/GPU is not usable from this process "
                "(permissions, wrong user, container without GPU, or NvRm errors). "
                "Try: nvidia-smi; run tests on the Jetson desktop session or fix GPU access."
            )
        )
        print(
            f"FAIL: torch.cuda.is_available() is False — {hint}\n"
            f"  torch.__version__ = {torch.__version__}",
            file=sys.stderr,
        )
        return 1

    try:
        from gpu_env import apply_jetson_vllm_runtime_patches

        apply_jetson_vllm_runtime_patches()
        import vllm
        # Shallow `import vllm` can succeed; `LLM` loads vllm._C and needs libcudart.so.12
        # (install nvidia-cuda-runtime-cu12 alongside PyTorch cu130).
        from vllm import LLM  # noqa: F401
    except ImportError as e:
        print("FAIL: vllm / LLM import:", e, file=sys.stderr)
        print(
            "  If the error mentions libcudart.so.12: pip install 'nvidia-cuda-runtime-cu12>=12.4,<13'",
            file=sys.stderr,
        )
        return 1

    print(f"OK — torch {torch.__version__}, vllm {vllm.__version__}, GPU {torch.cuda.get_device_name(0)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
