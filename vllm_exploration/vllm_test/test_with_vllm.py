"""
Test LLM inference WITH vLLM on CPU-only setup
"""
import time
from vllm import LLM, SamplingParams


def test_vllm_inference():
    """Test basic inference with vLLM on CPU"""
    print("\n=== Testing WITH vLLM ===")
    
    # Initialize vLLM with CPU-only settings
    # Using a small model for CPU testing
    # Note: CPU build of vLLM automatically uses CPU
    llm = LLM(
        model="facebook/opt-125m",  # Small model suitable for CPU
        max_num_seqs=1,
        enforce_eager=True,  # Disable CUDA graphs for CPU
    )
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50
    )
    
    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Python is a programming language that",
    ]
    
    print(f"\nRunning inference on {len(prompts)} prompts...")
    start_time = time.time()
    
    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print results
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Average time per prompt: {elapsed_time/len(prompts):.2f} seconds\n")
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 80)
    
    return elapsed_time


def test_vllm_batch_inference():
    """Test batch inference with vLLM"""
    print("\n=== Testing BATCH inference with vLLM ===")
    
    llm = LLM(
        model="facebook/opt-125m",
        max_num_seqs=5,
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=30
    )
    
    # Larger batch for testing
    prompts = [f"Question {i}: What is" for i in range(5)]
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_time = time.time() - start_time
    
    print(f"Batch of {len(prompts)} prompts completed in {elapsed_time:.2f} seconds")
    print(f"Average time per prompt: {elapsed_time/len(prompts):.2f} seconds")
    
    return elapsed_time


if __name__ == "__main__":
    print("Starting vLLM CPU-only tests...")
    
    try:
        time1 = test_vllm_inference()
        time2 = test_vllm_batch_inference()
        
        print("\n" + "=" * 80)
        print("SUMMARY - vLLM Tests")
        print("=" * 80)
        print(f"Single inference test: {time1:.2f}s")
        print(f"Batch inference test: {time2:.2f}s")
        
    except Exception as e:
        print(f"Error during vLLM testing: {e}")
        import traceback
        traceback.print_exc()
