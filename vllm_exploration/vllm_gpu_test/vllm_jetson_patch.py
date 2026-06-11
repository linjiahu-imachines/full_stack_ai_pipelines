"""
Jetson Thor (CUDA capability 11.0, sm_110a): Triton's bundled ``ptxas-blackwell`` may
reject ``--gpu-name=sm_110a``. vLLM V1 still JIT-compiles Triton kernels in
``block_table.compute_slot_mapping`` even when ``torch.compile`` is disabled.

This module monkey-patches:

* ``BlockTable.compute_slot_mapping`` — Triton slot mapping (see below).
* ``apply_top_k_top_p`` in ``topk_topp_sampler`` — Triton top-k/top-p for batch
  size ≥ 8 (hits ``profile_run`` / sampler; same ``ptxas-blackwell`` issue).

Call the ``apply_vllm_v1_*`` functions once per process **before**
``from vllm import LLM``, or use :func:`apply_all_jetson_thor_vllm_patches`.
"""
from __future__ import annotations

import os

_PATCHED = False
_TOPK_TOPP_PATCHED = False


def apply_vllm_v1_block_table_torch_fallback() -> bool:
    """
    Replace Triton slot-mapping kernel with PyTorch math (slower, works without ptxas).

    * Default: enable on device capability **(11, 0)** unless
      ``VLLM_JETSON_BLOCK_TABLE_TRITON=1`` (force stock Triton).
    * ``VLLM_JETSON_BLOCK_TABLE_TRITON=0`` / ``torch``: apply on any CUDA device
      (for debugging).

    Returns True if the patch was applied.
    """
    global _PATCHED
    if _PATCHED:
        return True

    raw = os.environ.get("VLLM_JETSON_BLOCK_TABLE_TRITON", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    cap = torch.cuda.get_device_capability(0)
    if cap != (11, 0):
        if raw not in ("0", "false", "torch", "pytorch"):
            return False

    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker import block_table as bt_mod

    def _compute_slot_mapping_torch(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Same logic as ``_compute_slot_mapping_kernel`` in block_table.py (Triton)."""
        num_tokens = positions.shape[0]
        max_num_tokens = self.max_num_batched_tokens
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        cp_interleave = self.cp_kv_cache_interleave_size
        block_size = self.block_size
        virtual_block_size = block_size * total_cp_world_size
        table = self.block_table.gpu
        sm = self.slot_mapping.gpu

        qsl = query_start_loc[: num_reqs + 1]
        pos = positions.long()

        for req_idx in range(num_reqs):
            start_idx = int(qsl[req_idx].item())
            end_idx = int(qsl[req_idx + 1].item())
            if start_idx >= end_idx:
                continue
            p = pos[start_idx:end_idx]
            block_indices = p // virtual_block_size
            block_numbers = table[req_idx, block_indices].long()
            virtual_block_offsets = p - block_indices * virtual_block_size
            is_local = ((virtual_block_offsets // cp_interleave) % total_cp_world_size) == total_cp_rank
            local_block_offsets = (
                (virtual_block_offsets // (total_cp_world_size * cp_interleave)) * cp_interleave
                + (virtual_block_offsets % cp_interleave)
            )
            slot_ids = block_numbers * block_size + local_block_offsets
            slot_ids = torch.where(
                is_local,
                slot_ids,
                torch.full_like(slot_ids, PAD_SLOT_ID, dtype=torch.int64),
            )
            sm[start_idx:end_idx] = slot_ids

        if num_tokens < max_num_tokens:
            sm[num_tokens:max_num_tokens].fill_(PAD_SLOT_ID)

    bt_mod.BlockTable.compute_slot_mapping = _compute_slot_mapping_torch  # type: ignore[assignment]
    _PATCHED = True
    print(
        "vLLM: BlockTable.compute_slot_mapping → PyTorch (Jetson Thor / sm_110a "
        "Triton/ptxas workaround). Set VLLM_JETSON_BLOCK_TABLE_TRITON=1 to use Triton."
    )
    return True


def apply_vllm_v1_topk_topp_torch_fallback() -> bool:
    """
    Replace ``apply_top_k_top_p`` Triton path with PyTorch (``apply_top_k_top_p_pytorch``).

    vLLM calls Triton when ``HAS_TRITON`` and ``logits.shape[0] >= 8``; engine
    ``profile_run`` exercises this and ``ptxas-blackwell`` fails on ``sm_110a``.

    * Default: enable on device capability **(11, 0)** unless
      ``VLLM_JETSON_TOPK_TOPP_TRITON=1`` (force stock Triton).
    * ``VLLM_JETSON_TOPK_TOPP_TRITON=0`` / ``torch``: apply on any CUDA device (debug).

    Returns True if the patch was applied.
    """
    global _TOPK_TOPP_PATCHED
    if _TOPK_TOPP_PATCHED:
        return True

    raw = os.environ.get("VLLM_JETSON_TOPK_TOPP_TRITON", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    cap = torch.cuda.get_device_capability(0)
    if cap != (11, 0):
        if raw not in ("0", "false", "torch", "pytorch"):
            return False

    from vllm.v1.sample.ops import topk_topp_sampler as tsm
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    def apply_top_k_top_p_no_triton(logits, k, p):
        if p is None and k is None:
            return logits
        return apply_top_k_top_p_pytorch(logits, k, p)

    tsm.apply_top_k_top_p = apply_top_k_top_p_no_triton  # type: ignore[assignment]
    _TOPK_TOPP_PATCHED = True
    print(
        "vLLM: apply_top_k_top_p → PyTorch (Jetson Thor / sm_110a; sampler Triton/ptxas workaround). "
        "Set VLLM_JETSON_TOPK_TOPP_TRITON=1 to use Triton."
    )
    return True


def apply_all_jetson_thor_vllm_patches() -> None:
    """Apply block-table and top-k/top-p PyTorch fallbacks when appropriate."""
    apply_vllm_v1_block_table_torch_fallback()
    apply_vllm_v1_topk_topp_torch_fallback()
