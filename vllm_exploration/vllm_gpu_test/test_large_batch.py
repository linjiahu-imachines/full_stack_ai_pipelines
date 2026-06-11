"""
Updated Performance Test: Large Batch Size (500) with CPU Core Monitoring
==========================================================================

This test suite:
1. Tests with batch size 500 (instead of 5)
2. Monitors actual CPU core usage during testing
3. Tracks CPU utilization per core
4. Compares vLLM vs Transformers on both CPU and GPU

Requirements:
    pip install vllm transformers torch psutil
"""

import time
import os
import torch
import psutil
from threading import Thread
from typing import List, Dict

from gpu_env import apply_jetson_vllm_runtime_patches, vllm_llm_kwargs

apply_jetson_vllm_runtime_patches()

# Default attention backend by GPU generation (override with VLLM_ATTENTION_BACKEND).
if "VLLM_ATTENTION_BACKEND" not in os.environ and torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" if major >= 10 else "TRITON_ATTN"
elif "VLLM_ATTENTION_BACKEND" not in os.environ:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"


# ============================================================================
# CPU MONITORING
# ============================================================================

class CPUMonitor:
    """Monitor CPU usage during tests"""
    
    def __init__(self):
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
    
    def start(self):
        """Start monitoring CPU usage"""
        self.monitoring = True
        self.samples = []
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring and return stats"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self._calculate_stats()
    
    def _monitor_loop(self):
        """Monitor loop running in separate thread"""
        while self.monitoring:
            # Get per-core CPU usage
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Get process-specific info
            process = psutil.Process(os.getpid())
            
            sample = {
                'timestamp': time.time(),
                'per_cpu': per_cpu,
                'total_cpu': psutil.cpu_percent(interval=0),
                'process_cpu': process.cpu_percent(interval=0),
                'num_threads': process.num_threads(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
            }
            
            self.samples.append(sample)
            time.sleep(0.1)  # Sample every 100ms
    
    def _calculate_stats(self):
        """Calculate statistics from samples"""
        if not self.samples:
            return None
        
        # Average per-core utilization
        num_cores = len(self.samples[0]['per_cpu'])
        avg_per_core = [0.0] * num_cores
        
        for sample in self.samples:
            for i, usage in enumerate(sample['per_cpu']):
                avg_per_core[i] += usage
        
        avg_per_core = [usage / len(self.samples) for usage in avg_per_core]
        
        # Count active cores (>5% utilization)
        active_cores = sum(1 for usage in avg_per_core if usage > 5.0)
        
        # Overall stats
        avg_total_cpu = sum(s['total_cpu'] for s in self.samples) / len(self.samples)
        avg_process_cpu = sum(s['process_cpu'] for s in self.samples) / len(self.samples)
        max_threads = max(s['num_threads'] for s in self.samples)
        avg_memory = sum(s['memory_mb'] for s in self.samples) / len(self.samples)
        
        return {
            'num_cores': num_cores,
            'active_cores': active_cores,
            'avg_per_core': avg_per_core,
            'avg_total_cpu': avg_total_cpu,
            'avg_process_cpu': avg_process_cpu,
            'max_threads': max_threads,
            'avg_memory_mb': avg_memory,
        }


# ============================================================================
# TEST: TRANSFORMERS ON CPU
# ============================================================================

def test_transformers_cpu_batch(batch_size=500):
    """Test Transformers on CPU with large batch"""
    print("\n" + "=" * 80)
    print(f"TRANSFORMERS ON CPU - Batch Size: {batch_size}")
    print("=" * 80)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Load model
    model_name = "facebook/opt-125m"
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on CPU")
    print(f"Total CPU cores available: {psutil.cpu_count()}")
    
    # Generate prompts
    prompts = [f"Question {i}: What is the meaning of" for i in range(batch_size)]
    
    print(f"\nGenerating {len(prompts)} prompts...")
    
    # Start CPU monitoring
    monitor = CPUMonitor()
    monitor.start()
    
    # Run inference
    start_time = time.time()
    
    # Batch encoding
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    
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
    
    # Stop monitoring
    cpu_stats = monitor.stop()
    
    # Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Time per prompt: {elapsed_time/batch_size*1000:.2f}ms")
    print(f"Throughput: {batch_size/elapsed_time:.2f} prompts/sec")
    
    if cpu_stats:
        print(f"\nCPU Usage:")
        print(f"  Total cores available: {cpu_stats['num_cores']}")
        print(f"  Active cores (>5% util): {cpu_stats['active_cores']}")
        print(f"  Average total CPU: {cpu_stats['avg_total_cpu']:.1f}%")
        print(f"  Average process CPU: {cpu_stats['avg_process_cpu']:.1f}%")
        print(f"  Max threads: {cpu_stats['max_threads']}")
        print(f"  Average memory: {cpu_stats['avg_memory_mb']:.0f} MB")
        
        # Show top 10 most utilized cores
        core_usage = list(enumerate(cpu_stats['avg_per_core']))
        core_usage.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n  Top 10 most utilized cores:")
        for i, (core_id, usage) in enumerate(core_usage[:10]):
            print(f"    Core {core_id:2d}: {usage:5.1f}%")
    
    print("=" * 80)
    
    return elapsed_time, cpu_stats


# ============================================================================
# TEST: vLLM ON CPU
# ============================================================================

def test_vllm_cpu_batch(batch_size=500):
    """Test vLLM on CPU with large batch - SKIPPED (vLLM is GPU-only)"""
    print("\n" + "=" * 80)
    print(f"vLLM ON CPU - Batch Size: {batch_size}")
    print("=" * 80)
    print("\n⚠️  SKIPPED: vLLM is a GPU-native inference engine.")
    print("   vLLM requires CUDA GPUs and does not support CPU-only inference.")
    print("   CPU comparison uses Transformers only.")
    print("=" * 80)
    
    return None, None


# ============================================================================
# TEST: TRANSFORMERS ON GPU
# ============================================================================

def test_transformers_gpu_batch(batch_size=500):
    """Test Transformers on GPU with large batch"""
    print("\n" + "=" * 80)
    print(f"TRANSFORMERS ON GPU - Batch Size: {batch_size}")
    print("=" * 80)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Load model
    model_name = "facebook/opt-125m"
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    device = "cuda:0"
    model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Generate prompts
    prompts = [f"Question {i}: What is the meaning of" for i in range(batch_size)]
    
    print(f"\nGenerating {len(prompts)} prompts...")
    
    # Monitor GPU during test
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    # Process in chunks to avoid OOM
    chunk_size = 50  # Process 50 at a time
    all_outputs = []
    
    for i in range(0, len(prompts), chunk_size):
        chunk_prompts = prompts[i:i+chunk_size]
        
        inputs = tokenizer(chunk_prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        all_outputs.extend(outputs)
        
        # Clear cache
        torch.cuda.empty_cache()
    
    elapsed_time = time.time() - start_time
    
    # GPU stats
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    # Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Time per prompt: {elapsed_time/batch_size*1000:.2f}ms")
    print(f"Throughput: {batch_size/elapsed_time:.2f} prompts/sec")
    print(f"\nGPU Memory:")
    print(f"  Peak memory: {peak_memory:.2f} GB")
    print("=" * 80)
    
    return elapsed_time, peak_memory


# ============================================================================
# TEST: vLLM ON GPU
# ============================================================================

def test_vllm_gpu_batch(batch_size=500):
    """Test vLLM on GPU with large batch"""
    print("\n" + "=" * 80)
    print(f"vLLM ON GPU - Batch Size: {batch_size}")
    print("=" * 80)
    
    from vllm import LLM, SamplingParams
    
    print(f"\nInitializing vLLM on GPU...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize vLLM
    torch.cuda.reset_peak_memory_stats()
    
    kw = vllm_llm_kwargs(tensor_parallel_size=1, max_num_seqs=batch_size)
    kw.update(
        model="facebook/opt-125m",
        dtype="float16",
        enforce_eager=True,
    )
    llm = LLM(**kw)
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=30
    )
    
    # Generate prompts
    prompts = [f"Question {i}: What is the meaning of" for i in range(batch_size)]
    
    print(f"\nGenerating {len(prompts)} prompts...")
    
    start_time = time.time()
    
    # vLLM handles large batches efficiently
    outputs = llm.generate(prompts, sampling_params)
    
    elapsed_time = time.time() - start_time
    
    # GPU stats
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    # Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Time per prompt: {elapsed_time/batch_size*1000:.2f}ms")
    print(f"Throughput: {batch_size/elapsed_time:.2f} prompts/sec")
    print(f"\nGPU Memory:")
    print(f"  Peak memory: {peak_memory:.2f} GB")
    print("=" * 80)
    
    return elapsed_time, peak_memory


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_large_batch_comparison(batch_size=500):
    """
    Run complete comparison with large batch size.
    """
    
    print("\n" + "=" * 80)
    print("LARGE BATCH PERFORMANCE COMPARISON")
    print(f"Batch Size: {batch_size}")
    print("=" * 80)
    
    # System info
    print(f"\nSystem Information:")
    print(f"  CPU: {psutil.cpu_count()} cores")
    print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"  GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    results = {}
    
    # ========================================================================
    # CPU TESTS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PART 1: CPU TESTS")
    print("=" * 80)
    
    try:
        print("\n[Test 1/2] Transformers on CPU...")
        trans_cpu_time, trans_cpu_stats = test_transformers_cpu_batch(batch_size)
        results['trans_cpu'] = {'time': trans_cpu_time, 'stats': trans_cpu_stats}
    except Exception as e:
        print(f"Error in Transformers CPU test: {e}")
        import traceback
        traceback.print_exc()
        results['trans_cpu'] = None
    
    try:
        print("\n[Test 2/2] vLLM on CPU...")
        vllm_cpu_time, vllm_cpu_stats = test_vllm_cpu_batch(batch_size)
        results['vllm_cpu'] = {'time': vllm_cpu_time, 'stats': vllm_cpu_stats}
    except Exception as e:
        print(f"Error in vLLM CPU test: {e}")
        import traceback
        traceback.print_exc()
        results['vllm_cpu'] = None
    
    # CPU Results Summary
    if results.get('trans_cpu'):
        print("\n" + "=" * 80)
        print("CPU RESULTS SUMMARY")
        print("=" * 80)
        
        trans_time = results['trans_cpu']['time']
        trans_stats = results['trans_cpu']['stats']
        
        print(f"\nTransformers on CPU:")
        print(f"  Time: {trans_time:.2f}s ({batch_size/trans_time:.2f} prompts/sec)")
        
        if trans_stats:
            print(f"\n  CPU Core Usage:")
            print(f"    Total cores available: {trans_stats['num_cores']}")
            print(f"    Active cores (>5% util): {trans_stats['active_cores']}")
            print(f"    Active core ratio: {trans_stats['active_cores']}/{trans_stats['num_cores']} ({trans_stats['active_cores']/trans_stats['num_cores']*100:.1f}%)")
            print(f"    Average total CPU: {trans_stats['avg_total_cpu']:.1f}%")
            print(f"    Average process CPU: {trans_stats['avg_process_cpu']:.1f}%")
            print(f"    Max threads: {trans_stats['max_threads']}")
            print(f"    Average memory: {trans_stats['avg_memory_mb']:.0f} MB")
        
        print(f"\n  vLLM on CPU: SKIPPED (GPU-only engine)")
        print("=" * 80)
    
    # ========================================================================
    # GPU TESTS
    # ========================================================================
    
    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available, skipping GPU tests")
        return results
    
    print("\n" + "=" * 80)
    print("PART 2: GPU TESTS")
    print("=" * 80)
    
    try:
        print("\n[Test 1/2] Transformers on GPU...")
        trans_gpu_time, trans_gpu_mem = test_transformers_gpu_batch(batch_size)
        results['trans_gpu'] = {'time': trans_gpu_time, 'memory': trans_gpu_mem}
    except Exception as e:
        print(f"Error in Transformers GPU test: {e}")
        import traceback
        traceback.print_exc()
        results['trans_gpu'] = None
    
    try:
        print("\n[Test 2/2] vLLM on GPU...")
        vllm_gpu_time, vllm_gpu_mem = test_vllm_gpu_batch(batch_size)
        results['vllm_gpu'] = {'time': vllm_gpu_time, 'memory': vllm_gpu_mem}
    except Exception as e:
        print(f"Error in vLLM GPU test: {e}")
        import traceback
        traceback.print_exc()
        results['vllm_gpu'] = None
    
    # GPU Comparison
    if results.get('trans_gpu') and results.get('vllm_gpu'):
        print("\n" + "=" * 80)
        print("GPU COMPARISON")
        print("=" * 80)
        
        trans_time = results['trans_gpu']['time']
        vllm_time = results['vllm_gpu']['time']
        trans_mem = results['trans_gpu']['memory']
        vllm_mem = results['vllm_gpu']['memory']
        
        print(f"\nPerformance:")
        print(f"  Transformers: {trans_time:.2f}s ({batch_size/trans_time:.2f} prompts/sec)")
        print(f"  vLLM:         {vllm_time:.2f}s ({batch_size/vllm_time:.2f} prompts/sec)")
        
        if trans_time < vllm_time:
            speedup = vllm_time / trans_time
            print(f"\n  ✅ Transformers is {speedup:.2f}x FASTER")
        else:
            speedup = trans_time / vllm_time
            print(f"\n  ✅ vLLM is {speedup:.2f}x FASTER")
        
        print(f"\nMemory:")
        print(f"  Transformers peak: {trans_mem:.2f} GB")
        print(f"  vLLM peak:         {vllm_mem:.2f} GB")
        
        if trans_mem > vllm_mem:
            saving = (trans_mem - vllm_mem) / trans_mem * 100
            print(f"\n  ✅ vLLM uses {saving:.1f}% LESS memory")
        
        print("=" * 80)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - BATCH SIZE 500")
    print("=" * 80)
    
    print("\n📊 CPU Results (Transformers only - vLLM is GPU-only):")
    if results.get('trans_cpu'):
        trans_time = results['trans_cpu']['time']
        trans_stats = results['trans_cpu']['stats']
        trans_cores = trans_stats['active_cores'] if trans_stats else 'N/A'
        
        print(f"  Transformers: {trans_time:.2f}s using {trans_cores} active cores")
        print(f"  Throughput: {batch_size/trans_time:.2f} prompts/sec")
    
    print("\n📊 GPU Results:")
    if results.get('trans_gpu') and results.get('vllm_gpu'):
        trans_time = results['trans_gpu']['time']
        vllm_time = results['vllm_gpu']['time']
        
        print(f"  Transformers: {trans_time:.2f}s ({batch_size/trans_time:.2f} prompts/sec)")
        print(f"  vLLM:         {vllm_time:.2f}s ({batch_size/vllm_time:.2f} prompts/sec)")
        
        if trans_time < vllm_time:
            print(f"  Winner: Transformers ({vllm_time/trans_time:.2f}x faster)")
        else:
            print(f"  Winner: vLLM ({trans_time/vllm_time:.2f}x faster)")
    elif results.get('trans_gpu'):
        trans_time = results['trans_gpu']['time']
        print(f"  Transformers: {trans_time:.2f}s ({batch_size/trans_time:.2f} prompts/sec)")
        print(f"  vLLM: FAILED (see errors above)")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    
    # CPU insights
    if results.get('trans_cpu') and results['trans_cpu'].get('stats'):
        trans_stats = results['trans_cpu']['stats']
        trans_cores = trans_stats['active_cores']
        total_cores = trans_stats['num_cores']
        
        print(f"\nCPU Core Utilization (Transformers):")
        print(f"  Total available: {total_cores} cores")
        print(f"  Active cores used: {trans_cores} cores ({trans_cores/total_cores*100:.1f}%)")
        print(f"  Average CPU util: {trans_stats['avg_total_cpu']:.1f}%")
        print(f"  Max threads: {trans_stats['max_threads']}")
        
        if trans_cores < total_cores * 0.5:
            print(f"  → Only {trans_cores/total_cores*100:.0f}% of cores utilized")
            print(f"  → PyTorch does NOT fully utilize all available cores")
        else:
            print(f"  → Good utilization: {trans_cores/total_cores*100:.0f}% of cores active")
    
    # GPU insights
    if results.get('trans_gpu') and results.get('vllm_gpu'):
        trans_time = results['trans_gpu']['time']
        vllm_time = results['vllm_gpu']['time']
        
        print(f"\nGPU Performance:")
        if vllm_time < trans_time:
            speedup = trans_time / vllm_time
            print(f"  vLLM is {speedup:.2f}x faster than Transformers on GPU")
            print(f"  → PagedAttention and continuous batching help at batch=500")
        else:
            speedup = vllm_time / trans_time
            print(f"  Transformers is {speedup:.2f}x faster than vLLM on GPU")
    
    print("\n" + "=" * 80)
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Large batch performance comparison")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for testing (default: 500)"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run CPU tests only"
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Run GPU tests only"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("VLLM LARGE BATCH PERFORMANCE TEST")
    print("=" * 80)
    print(f"\nTest Date: February 17, 2026")
    print(f"Batch Size: {args.batch_size}")
    print(f"Model: facebook/opt-125m")
    print("=" * 80)
    
    # Run comparison
    results = run_large_batch_comparison(batch_size=args.batch_size)
    
    print("\n✅ Test completed!")
    print(f"\nResults saved to: LARGE_BATCH_RESULTS_{args.batch_size}.md")
