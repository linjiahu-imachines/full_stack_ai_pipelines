"""
Shared GPU test helpers: CUDA checks, vLLM kwargs, and memory cleanup (Jetson-friendly).
"""
from __future__ import annotations

import gc
import os
import sys

DEFAULT_MODEL = os.environ.get("GPU_TEST_MODEL", "facebook/opt-125m")


def apply_jetson_vllm_runtime_patches() -> None:
    """Call once per process **before** ``from vllm import LLM`` on Jetson Thor (CC 11.0).

    Patches vLLM V1 to avoid Triton ``ptxas-blackwell`` failures on sm_110a:

    * ``BlockTable.compute_slot_mapping`` (see ``vllm_jetson_patch.py``).
    * ``apply_top_k_top_p`` → PyTorch when batch dim ≥ 8 (sampler ``profile_run``).
    """
    try:
        from vllm_jetson_patch import (
            apply_vllm_v1_block_table_torch_fallback,
            apply_vllm_v1_topk_topp_torch_fallback,
        )
    except ImportError:
        return
    apply_vllm_v1_block_table_torch_fallback()
    apply_vllm_v1_topk_topp_torch_fallback()

# Sentinel: do not pass compilation_config to vllm.LLM (use vLLM's default CompilationConfig).
_OMIT_COMPILATION_CONFIG = object()
_GPU_MEM_UTIL_LOGGED = False


def _vllm_compilation_config():
    """Resolve vLLM ``compilation_config`` (torch.compile / Inductor).

    Jetson Thor reports CC **11.0** (sm_110a). Triton's ``ptxas-blackwell`` may fail with
    ``Value 'sm_110a' is not defined for option 'gpu-name'`` during ``torch.compile``.
    For that case we default to mode **0** (``CompilationMode.NONE``, fully eager).

    Environment:

    * **Unset or ``auto``:** use mode ``0`` on device capability (11, 0); else omit (vLLM default).
    * **``default``:** always omit (try vLLM default compilation even on 11.0).
    * **``0`` / ``none`` / ``NONE``:** force eager (no torch.compile).
    * **``1``–``3``:** integer ``CompilationMode`` (see vLLM docs).

    Returns:
        ``int`` mode, or ``_OMIT_COMPILATION_CONFIG`` to omit the kwarg.
    """
    raw = os.environ.get("VLLM_COMPILATION_MODE", "").strip()
    key = raw.lower()
    if key == "default":
        return _OMIT_COMPILATION_CONFIG
    if key in ("", "auto"):
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (11, 0):
            return 0
        return _OMIT_COMPILATION_CONFIG
    if key in ("0", "none"):
        return 0
    if raw.isdigit():
        return int(raw)
    try:
        from vllm.config.compilation import CompilationMode

        return int(CompilationMode[raw.upper()])
    except Exception:
        print(
            f"WARNING: invalid VLLM_COMPILATION_MODE={raw!r}; "
            "using vLLM default compilation settings.",
            file=sys.stderr,
        )
        return _OMIT_COMPILATION_CONFIG


def require_cuda(exit_code: int = 1) -> None:
    import torch

    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available. This suite needs a CUDA-enabled PyTorch build.\n"
            "  On Jetson AGX Thor, install PyTorch for your JetPack, then reinstall vLLM if needed.\n"
            "  Run:  bash install_jetson_gpu_deps.sh\n"
            f"  torch.__version__ = {torch.__version__}",
            file=sys.stderr,
        )
        sys.exit(exit_code)
    print(
        f"CUDA OK — torch {torch.__version__}, "
        f"{torch.cuda.device_count()} device(s), "
        f"device0={torch.cuda.get_device_name(0)!r}"
    )


def cuda_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.synchronize()
    except Exception:
        pass


def _default_vllm_gpu_mem_util() -> float:
    """vLLM V1 ``request_memory`` requires ``free_memory >= total * gpu_memory_utilization``.

    On large unified-memory devices (Jetson Thor), default **0.85** implies ~85% of
    total memory must be *free* at engine startup — unrealistic after other GPU work
    or with desktop memory use. Use a lower default on CC **11.0** unless
    ``VLLM_GPU_MEM_UTIL`` is set explicitly.
    """
    raw = os.environ.get("VLLM_GPU_MEM_UTIL", "").strip()
    if raw:
        return float(raw)
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (11, 0):
        return 0.10
    return 0.85


def vllm_llm_kwargs(
    *,
    tensor_parallel_size: int = 1,
    max_num_seqs: int = 1,
) -> dict:
    """Build kwargs for vllm.LLM() with env overrides (Jetson / unified memory)."""
    import torch

    mem = _default_vllm_gpu_mem_util()
    global _GPU_MEM_UTIL_LOGGED
    if (
        not _GPU_MEM_UTIL_LOGGED
        and os.environ.get("VLLM_GPU_MEM_UTIL", "").strip() == ""
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(0) == (11, 0)
    ):
        _GPU_MEM_UTIL_LOGGED = True
        print(
            f"vLLM: gpu_memory_utilization={mem} (CC 11.0 default; vLLM needs "
            f"free_memory ≥ total×util. Raise with VLLM_GPU_MEM_UTIL if plenty of free memory.)"
        )
    kv: dict = {
        "model": DEFAULT_MODEL,
        "tensor_parallel_size": tensor_parallel_size,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": mem,
    }
    if os.environ.get("VLLM_ENFORCE_EAGER", "").lower() in ("1", "true", "yes"):
        kv["enforce_eager"] = True
    dtype = os.environ.get("VLLM_DTYPE", "").strip()
    if dtype:
        kv["dtype"] = dtype
    comp = _vllm_compilation_config()
    if comp is not _OMIT_COMPILATION_CONFIG:
        kv["compilation_config"] = comp
        if (
            comp == 0
            and os.environ.get("VLLM_COMPILATION_MODE", "").strip().lower()
            in ("", "auto")
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) == (11, 0)
        ):
            print(
                "vLLM: compilation_config=0 (eager, no torch.compile) for CUDA capability "
                "11.0 — avoids Triton/ptxas-blackwell issues on sm_110a. "
                "Override: VLLM_COMPILATION_MODE=default"
            )
    return kv


def transformers_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    dt = os.environ.get("TRANSFORMERS_DTYPE", "float16").lower()
    if dt in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16
