"""
Test LLM inference WITH vLLM on GPU
"""
import time
import torch

from gpu_env import (
    apply_jetson_vllm_runtime_patches,
    cuda_cleanup,
    require_cuda,
    vllm_llm_kwargs,
)

apply_jetson_vllm_runtime_patches()
from vllm import LLM, SamplingParams


def test_vllm_gpu_single():
    """Test single GPU inference with vLLM (one process, batched prompts).

    ``max_num_seqs`` must be >= number of prompts in one ``generate()`` call; otherwise
    vLLM serializes them (e.g. ``max_num_seqs=1`` with 3 prompts ≈ 3× latency).
    """
    print("\n=== Testing WITH vLLM on Single GPU ===")

    llm = None
    try:
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "Python is a programming language that",
        ]
        # Must allow all prompts to be scheduled together (not one-at-a-time).
        llm = LLM(
            **vllm_llm_kwargs(
                tensor_parallel_size=1,
                max_num_seqs=max(len(prompts), 4),
            )
        )

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50,
        )

        print(f"\nRunning inference on {len(prompts)} prompts (Single GPU)...")
        start_time = time.time()

        outputs = llm.generate(prompts, sampling_params)

        elapsed_time = time.time() - start_time

        print(f"\nCompleted in {elapsed_time:.2f} seconds")
        print(f"Average time per prompt: {elapsed_time/len(prompts):.2f} seconds\n")

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 80)

        return elapsed_time
    finally:
        if llm is not None:
            del llm
        cuda_cleanup()


def test_vllm_gpu_batch():
    """Test batch inference with vLLM on single GPU"""
    print("\n=== Testing BATCH inference with vLLM on Single GPU ===")

    llm = None
    try:
        llm = LLM(**vllm_llm_kwargs(tensor_parallel_size=1, max_num_seqs=5))

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=30,
        )

        prompts = [f"Question {i}: What is" for i in range(5)]

        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed_time = time.time() - start_time

        print(f"Batch of {len(prompts)} prompts completed in {elapsed_time:.2f} seconds")
        print(f"Average time per prompt: {elapsed_time/len(prompts):.2f} seconds")

        return elapsed_time
    finally:
        if llm is not None:
            del llm
        cuda_cleanup()


def test_vllm_multi_gpu():
    """Test multi-GPU inference with vLLM using tensor parallelism"""
    print("\n=== Testing WITH vLLM on Multi-GPU (Tensor Parallel) ===")

    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    if n_gpus < 2:
        print("Skipping multi-GPU test (need at least 2 GPUs)")
        return None

    tp = min(4, n_gpus)
    llm = None
    try:
        llm = LLM(**vllm_llm_kwargs(tensor_parallel_size=tp, max_num_seqs=5))

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50,
        )

        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "Python is a programming language that",
        ]

        print(f"\nRunning inference on {len(prompts)} prompts (Multi-GPU, TP={tp})...")
        start_time = time.time()

        outputs = llm.generate(prompts, sampling_params)

        elapsed_time = time.time() - start_time

        print(f"\nCompleted in {elapsed_time:.2f} seconds")
        print(f"Average time per prompt: {elapsed_time/len(prompts):.2f} seconds")

        return elapsed_time
    finally:
        if llm is not None:
            del llm
        cuda_cleanup()


if __name__ == "__main__":
    print("=" * 80)
    print("vLLM GPU Performance Tests")
    print("=" * 80)
    require_cuda()  # also sets VLLM_ATTENTION_BACKEND when appropriate
    print(f"\nGPU Information:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    try:
        time1 = test_vllm_gpu_single()
        time2 = test_vllm_gpu_batch()
        time3 = test_vllm_multi_gpu()

        print("\n" + "=" * 80)
        print("SUMMARY - vLLM GPU Tests")
        print("=" * 80)
        print(f"Single GPU inference:     {time1:.2f}s")
        print(f"Single GPU batch:         {time2:.2f}s")
        if time3:
            print(f"Multi-GPU inference:      {time3:.2f}s")
            if time1:
                speedup = time1 / time3
                print(f"Multi-GPU speedup:        {speedup:.2f}x")

    except Exception as e:
        print(f"Error during vLLM GPU testing: {e}")
        import traceback

        traceback.print_exc()
