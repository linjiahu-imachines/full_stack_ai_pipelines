"""
Transformers only (no vLLM): compare inference on CPU vs GPU with the same prompts/settings.
Skips GPU section if CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpu_env import DEFAULT_MODEL
from gpu_env import cuda_cleanup, transformers_dtype


def _single_prompts():
    return [
        "Hello, my name is",
        "The capital of France is",
        "Python is a programming language that",
    ]


def _batch_prompts(n: int):
    return [f"Question {i}: What is" for i in range(n)]


def run_transformers_bench(
    *,
    model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    max_new_tokens_single: int,
    max_new_tokens_batch: int,
    batch_size: int,
    label: str,
) -> dict:
    print(f"\n{'=' * 80}\n{label}\ndevice={device} dtype={torch_dtype}\n{'=' * 80}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_kw = dict(
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompts_s = _single_prompts()
    t0 = time.time()
    for prompt in prompts_s:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens_single, **gen_kw)
    if device.type == "cuda":
        torch.cuda.synchronize()
    single_elapsed = time.time() - t0

    prompts_b = _batch_prompts(batch_size)
    t1 = time.time()
    inputs = tokenizer(
        prompts_b, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens_batch, **gen_kw)
    if device.type == "cuda":
        torch.cuda.synchronize()
    batch_elapsed = time.time() - t1

    del model
    gc.collect()
    cuda_cleanup()

    n_s, n_b = len(prompts_s), len(prompts_b)
    out = {
        "device": str(device),
        "dtype": str(torch_dtype),
        "single_time_s": round(single_elapsed, 4),
        "single_ms_per_prompt": round((single_elapsed / n_s) * 1000, 2),
        "batch_time_s": round(batch_elapsed, 4),
        "batch_ms_per_prompt": round((batch_elapsed / n_b) * 1000, 2),
    }
    print(f"Single ({n_s} prompts, max_new_tokens={max_new_tokens_single}): {out['single_time_s']}s "
          f"({out['single_ms_per_prompt']} ms/prompt)")
    print(f"Batch ({n_b} prompts, max_new_tokens={max_new_tokens_batch}): {out['batch_time_s']}s "
          f"({out['batch_ms_per_prompt']} ms/prompt)")
    return out


def main():
    parser = argparse.ArgumentParser(description="Transformers CPU vs GPU (no vLLM)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Override HF model id")
    parser.add_argument("--max-new-tokens-single", type=int, default=50)
    parser.add_argument("--max-new-tokens-batch", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--skip-gpu", action="store_true", help="Only run CPU")
    parser.add_argument("--skip-cpu", action="store_true", help="Only run GPU (requires CUDA)")
    args = parser.parse_args()

    model_name = args.model

    print("Transformers-only CPU vs GPU benchmark (no vLLM)")
    print(f"Model: {model_name}")
    print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()}")

    results: dict = {"model": model_name}

    if not args.skip_cpu:
        cpu_dtype = torch.float32
        results["cpu"] = run_transformers_bench(
            model_name=model_name,
            device=torch.device("cpu"),
            torch_dtype=cpu_dtype,
            max_new_tokens_single=args.max_new_tokens_single,
            max_new_tokens_batch=args.max_new_tokens_batch,
            batch_size=args.batch_size,
            label="CPU (Transformers)",
        )

    if not args.skip_gpu:
        if not torch.cuda.is_available():
            print("\n[GPU] CUDA not available — skipping GPU section.")
            results["gpu"] = None
        else:
            results["gpu"] = run_transformers_bench(
                model_name=model_name,
                device=torch.device("cuda:0"),
                torch_dtype=transformers_dtype(),
                max_new_tokens_single=args.max_new_tokens_single,
                max_new_tokens_batch=args.max_new_tokens_batch,
                batch_size=args.batch_size,
                label="GPU (Transformers)",
            )

    if results.get("cpu") and results.get("gpu"):
        c, g = results["cpu"], results["gpu"]
        print("\n" + "=" * 80)
        print("Ratio (GPU time / CPU time); <1 means GPU faster")
        print("=" * 80)
        print(
            f"Single: {g['single_time_s'] / c['single_time_s']:.3f}x | "
            f"Batch: {g['batch_time_s'] / c['batch_time_s']:.3f}x"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
