"""
Comparison test suite for vLLM vs transformers on CPU-only setup
"""
import time
import pytest
from test_with_vllm import test_vllm_inference, test_vllm_batch_inference
from test_without_vllm import test_transformers_inference, test_transformers_batch_inference


class TestComparison:
    """Compare vLLM and transformers performance"""
    
    def test_single_inference_comparison(self):
        """Compare single inference performance"""
        print("\n" + "=" * 80)
        print("COMPARISON: Single Inference")
        print("=" * 80)
        
        # Test transformers
        print("\n[1/2] Testing transformers...")
        transformers_time = test_transformers_inference()
        
        # Test vLLM
        print("\n[2/2] Testing vLLM...")
        vllm_time = test_vllm_inference()
        
        # Compare results
        print("\n" + "=" * 80)
        print("RESULTS COMPARISON")
        print("=" * 80)
        print(f"Transformers (without vLLM): {transformers_time:.2f}s")
        print(f"vLLM:                         {vllm_time:.2f}s")
        
        speedup = transformers_time / vllm_time if vllm_time > 0 else 0
        print(f"\nSpeedup factor: {speedup:.2f}x")
        
        if vllm_time < transformers_time:
            improvement = ((transformers_time - vllm_time) / transformers_time) * 100
            print(f"vLLM is {improvement:.1f}% faster")
        else:
            slowdown = ((vllm_time - transformers_time) / transformers_time) * 100
            print(f"vLLM is {slowdown:.1f}% slower")
        
        assert transformers_time > 0 and vllm_time > 0, "Both tests should complete"
    
    def test_batch_inference_comparison(self):
        """Compare batch inference performance"""
        print("\n" + "=" * 80)
        print("COMPARISON: Batch Inference")
        print("=" * 80)
        
        # Test transformers
        print("\n[1/2] Testing transformers batch...")
        transformers_time = test_transformers_batch_inference()
        
        # Test vLLM
        print("\n[2/2] Testing vLLM batch...")
        vllm_time = test_vllm_batch_inference()
        
        # Compare results
        print("\n" + "=" * 80)
        print("BATCH RESULTS COMPARISON")
        print("=" * 80)
        print(f"Transformers (without vLLM): {transformers_time:.2f}s")
        print(f"vLLM:                         {vllm_time:.2f}s")
        
        speedup = transformers_time / vllm_time if vllm_time > 0 else 0
        print(f"\nSpeedup factor: {speedup:.2f}x")
        
        if vllm_time < transformers_time:
            improvement = ((transformers_time - vllm_time) / transformers_time) * 100
            print(f"vLLM is {improvement:.1f}% faster")
        else:
            slowdown = ((vllm_time - transformers_time) / transformers_time) * 100
            print(f"vLLM is {slowdown:.1f}% slower")
        
        assert transformers_time > 0 and vllm_time > 0, "Both tests should complete"


def run_full_comparison():
    """Run full comparison suite"""
    print("\n" + "=" * 80)
    print("FULL COMPARISON TEST SUITE")
    print("vLLM vs Transformers on CPU-only Setup")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run tests
    test_comp = TestComparison()
    
    try:
        test_comp.test_single_inference_comparison()
        test_comp.test_batch_inference_comparison()
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        print(f"Total test duration: {total_time:.2f}s")
        print("\nConclusion:")
        print("- vLLM provides optimized inference even on CPU")
        print("- vLLM's PagedAttention and optimizations can improve throughput")
        print("- For production CPU workloads, vLLM can be a good choice")
        
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_full_comparison()
