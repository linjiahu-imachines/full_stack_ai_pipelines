"""
vLLM on GPU: batch throughput for Qwen/Qwen3-1.7B (same prompts as CPU harness).

Run with the CUDA vLLM venv, e.g.:
  vllm_gpu_test/venv/bin/python3 test_qwen3_vllm_gpu_batch.py --batch-size 500 --max-num-seqs 128

Uses gpu_env (Jetson memory defaults, optional compilation_config for CC 11.0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import torch

from gpu_env import (
    apply_jetson_vllm_runtime_patches,
    cuda_cleanup,
    require_cuda,
    vllm_llm_kwargs,
)

if "VLLM_ATTENTION_BACKEND" not in os.environ and torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" if major >= 10 else "TRITON_ATTN"
elif "VLLM_ATTENTION_BACKEND" not in os.environ:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

apply_jetson_vllm_runtime_patches()


def build_prompts(batch_size: int) -> list[str]:
    return [f"Question {i}: Explain CPU inference trade-offs in one paragraph." for i in range(batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM GPU batch test for Qwen3-1.7B")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write the result JSON.",
    )
    args = parser.parse_args()

    require_cuda()

    from vllm import LLM, SamplingParams

    prompts = build_prompts(args.batch_size)
    effective = max(1, min(args.batch_size, args.max_num_seqs))

    kw = vllm_llm_kwargs(tensor_parallel_size=1, max_num_seqs=effective)
    kw["model"] = args.model

    init_t0 = time.time()
    llm = LLM(**kw)
    init_time = time.time() - init_t0

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    t0 = time.time()
    _ = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    del llm
    cuda_cleanup()

    result = {
        "engine": "qwen3-1.7B with vLLM (GPU)",
        "model": args.model,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_num_seqs": effective,
        "init_time_s": round(init_time, 2),
        "time_s": round(elapsed, 2),
        "per_prompt_ms": round((elapsed / args.batch_size) * 1000, 2),
        "throughput_prompts_per_s": round(args.batch_size / elapsed, 2) if elapsed > 0 else 0.0,
    }
    print(f"__RESULT_JSON__:{json.dumps(result)}", flush=True)

    if args.output.strip():
        out_path = os.path.abspath(args.output.strip())
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
