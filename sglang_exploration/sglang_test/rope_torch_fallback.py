"""
Force SGLang ``RotaryEmbedding`` to use ``forward_native`` (pure PyTorch) when
``nvcc`` is missing, so TVM-FFI JIT RoPE is never invoked (including in worker processes).

Used by ``sitecustomize.py`` (venv) and ``run_offline_batch_experiment.py``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def nvcc_available() -> bool:
    if shutil.which("nvcc"):
        return True
    for base in (Path("/usr/local/cuda"), *sorted(Path("/usr/local").glob("cuda-*"))):
        if (base / "bin" / "nvcc").is_file():
            return True
    for env_key in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_key)
        if root and (Path(root) / "bin" / "nvcc").is_file():
            return True
    return False


def apply() -> None:
    flag = os.environ.get("SGLANG_FORCE_TORCH_ROPE_CUDA", "auto").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return
    if flag in ("1", "true", "yes", "on"):
        use_torch = True
    elif flag == "auto":
        use_torch = not nvcc_available()
    else:
        return
    if not use_torch:
        return

    try:
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    except Exception:
        return

    if getattr(RotaryEmbedding.__init__, "_sglang_torch_rope_patched", False):
        return

    _orig_init = RotaryEmbedding.__init__

    def _wrapped_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._forward_method = self.forward_native

    _wrapped_init._sglang_torch_rope_patched = True  # type: ignore[attr-defined]
    RotaryEmbedding.__init__ = _wrapped_init  # type: ignore[method-assign]

    print(
        "SGLang: PyTorch-native RoPE (no nvcc / JIT RoPE skipped). "
        "Install CUDA toolkit + nvcc for fused RoPE; "
        "SGLANG_FORCE_TORCH_ROPE_CUDA=0 disables.",
        file=sys.stderr,
    )
