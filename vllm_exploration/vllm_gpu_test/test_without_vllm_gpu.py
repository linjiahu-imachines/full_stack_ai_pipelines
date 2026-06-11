"""
Test LLM inference WITHOUT vLLM (using transformers directly) on GPU
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gpu_env import cuda_cleanup, transformers_dtype
from gpu_env import DEFAULT_MODEL as MODEL_NAME


def test_transformers_gpu_single():
    """Test basic inference with transformers (without vLLM) on single GPU"""
    print("\n=== Testing WITHOUT vLLM (using transformers) on Single GPU ===")
    
    # Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=transformers_dtype(),
    )
    
    # Move to GPU
    device = "cuda:0"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    
    # Test prompts (same as vLLM test)
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Python is a programming language that",
    ]
    
    print(f"\nRunning inference on {len(prompts)} prompts...")
    start_time = time.time()
    
    results = []
    
    # Generate outputs one by one
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append((prompt, generated_text))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print results
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Average time per prompt: {elapsed_time/len(prompts):.2f} seconds\n")
    
    for prompt, generated_text in results:
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 80)

    del model
    del tokenizer
    cuda_cleanup()
    return elapsed_time


def test_transformers_gpu_batch():
    """Test batch inference with transformers on single GPU"""
    print("\n=== Testing BATCH inference with transformers on Single GPU ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=transformers_dtype(),
    )

    device = "cuda:0"
    model = model.to(device)
    model.eval()
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Larger batch for testing
    prompts = [f"Question {i}: What is" for i in range(5)]
    
    start_time = time.time()
    
    # Batch encoding
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    elapsed_time = time.time() - start_time
    
    print(f"Batch of {len(prompts)} prompts completed in {elapsed_time:.2f} seconds")
    print(f"Average time per prompt: {elapsed_time/len(prompts):.2f} seconds")

    del model
    del tokenizer
    cuda_cleanup()
    return elapsed_time


def test_transformers_multi_gpu():
    """Test multi-GPU inference using DataParallel"""
    print("\n=== Testing Multi-GPU with transformers (DataParallel) ===")
    
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    
    if n_gpus < 2:
        print("Skipping multi-GPU test (need at least 2 GPUs)")
        return None
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=transformers_dtype(),
    )

    # Use DataParallel to distribute across GPUs
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    
    print(f"Model distributed across {n_gpus} GPUs")
    
    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is", 
        "Python is a programming language that",
    ]
    
    print(f"\nRunning inference on {len(prompts)} prompts...")
    start_time = time.time()
    
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").cuda()
        
        with torch.no_grad():
            outputs = model.module.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append((prompt, generated_text))
    
    elapsed_time = time.time() - start_time
    
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Average time per prompt: {elapsed_time/len(prompts):.2f} seconds")

    del model
    del tokenizer
    cuda_cleanup()
    return elapsed_time


if __name__ == "__main__":
    from gpu_env import require_cuda

    print("=" * 80)
    print("Transformers GPU Performance Tests")
    print("=" * 80)
    require_cuda()
    print(f"\nGPU Information:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    try:
        time1 = test_transformers_gpu_single()
        time2 = test_transformers_gpu_batch()
        time3 = test_transformers_multi_gpu()
        
        print("\n" + "=" * 80)
        print("SUMMARY - Transformers GPU Tests")
        print("=" * 80)
        print(f"Single GPU inference:     {time1:.2f}s")
        print(f"Single GPU batch:         {time2:.2f}s")
        if time3:
            print(f"Multi-GPU inference:      {time3:.2f}s")
            if time1:
                speedup = time1 / time3
                print(f"Multi-GPU speedup:        {speedup:.2f}x")
        
    except Exception as e:
        print(f"Error during transformers GPU testing: {e}")
        import traceback
        traceback.print_exc()
