"""
Test LLM inference WITHOUT vLLM (using transformers directly) on CPU-only setup
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_transformers_inference():
    """Test basic inference with transformers (without vLLM) on CPU"""
    print("\n=== Testing WITHOUT vLLM (using transformers) ===")
    
    # Load model and tokenizer
    model_name = "facebook/opt-125m"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure CPU usage
    device = "cpu"
    model = model.to(device)
    model.eval()
    
    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Python is a programming language that",
    ]
    
    print(f"\nRunning inference on {len(prompts)} prompts...")
    start_time = time.time()
    
    results = []
    
    # Generate outputs one by one (no batching for fair comparison)
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
    
    return elapsed_time


def test_transformers_batch_inference():
    """Test batch inference with transformers"""
    print("\n=== Testing BATCH inference with transformers ===")
    
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = "cpu"
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
    
    return elapsed_time


if __name__ == "__main__":
    print("Starting transformers CPU-only tests (without vLLM)...")
    
    try:
        time1 = test_transformers_inference()
        time2 = test_transformers_batch_inference()
        
        print("\n" + "=" * 80)
        print("SUMMARY - Transformers Tests (without vLLM)")
        print("=" * 80)
        print(f"Single inference test: {time1:.2f}s")
        print(f"Batch inference test: {time2:.2f}s")
        
    except Exception as e:
        print(f"Error during transformers testing: {e}")
        import traceback
        traceback.print_exc()
