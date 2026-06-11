#!/usr/bin/env python3
"""
Benchmark vLLM (CPU) vs llama.cpp (native x86) on the same hardware.

This script compares vLLM CPU mode with llama.cpp native x86 build
to establish a performance baseline. Results inform whether vLLM
is worth compiling for RISC-V.

Usage:
    # Start vLLM server first (Terminal 1):
    vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cpu --port 8001
    
    # Run benchmark (Terminal 2):
    python scripts/benchmark_vllm_vs_llamacpp.py
"""

import subprocess
import requests
import time
import json
from pathlib import Path
import sys
import argparse

# Configuration
VLLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
VLLM_URL = "http://localhost:8001/v1/completions"
LLAMACPP_BIN = Path("/home/linhu/repo/iminn-tools/dev_env/llama.cpp/linux-build-x86-native/bin/llama-cli")
LLAMACPP_MODEL = Path("/home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/stories15M-q4_0.gguf")

# Alternative: Use x86 build if native not available
if not LLAMACPP_BIN.exists():
    LLAMACPP_BIN = Path("/home/linhu/repo/iminn-tools/dev_env/llama.cpp/linux-build-x86/bin/llama-cli")

# Test prompts
TEST_PROMPTS = [
    "Hello, how are you?",
    "Write a short story about a robot.",
    "Explain what RISC-V is in one sentence.",
]

# Token lengths to test
TOKEN_LENGTHS = [32, 64, 128]


def benchmark_vllm(prompt: str, max_tokens: int) -> dict:
    """Benchmark vLLM inference."""
    try:
        start = time.time()
        response = requests.post(
            VLLM_URL,
            json={
                "model": VLLM_MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "seed": 42
            },
            timeout=300
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            tokens_generated = data['usage']['completion_tokens']
            throughput = tokens_generated / elapsed if elapsed > 0 else 0
            
            return {
                "success": True,
                "time": elapsed,
                "tokens": tokens_generated,
                "throughput": throughput,
                "latency_per_token": elapsed / tokens_generated if tokens_generated > 0 else 0,
                "text": data['choices'][0]['text'][:100]
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def benchmark_llamacpp(prompt: str, max_tokens: int) -> dict:
    """Benchmark llama.cpp inference."""
    try:
        cmd = [
            str(LLAMACPP_BIN),
            "-m", str(LLAMACPP_MODEL),
            "-p", prompt,
            "-n", str(max_tokens),
            "--seed", "42",
            "-ngl", "0",
            "--no-warmup",
            "-t", "1"  # Single thread for fair comparison
        ]
        
        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            # Parse output for metrics
            output = result.stdout
            
            # Extract tokens/sec from output
            import re
            throughput_match = re.search(r'(\d+\.\d+)\s+tokens per second', output)
            tokens_match = re.search(r'eval time.*?/\s+(\d+)\s+runs', output)
            
            tokens_generated = int(tokens_match.group(1)) if tokens_match else max_tokens
            throughput = float(throughput_match.group(1)) if throughput_match else (tokens_generated / elapsed)
            
            return {
                "success": True,
                "time": elapsed,
                "tokens": tokens_generated,
                "throughput": throughput,
                "latency_per_token": 1000.0 / throughput if throughput > 0 else 0,
                "text": output[:100]
            }
        else:
            return {
                "success": False,
                "error": f"Exit code {result.returncode}: {result.stderr[:200]}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM vs llama.cpp")
    parser.add_argument("--vllm-url", default=VLLM_URL, help="vLLM server URL")
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKEN_LENGTHS, help="Token lengths to test")
    args = parser.parse_args()
    
    print("="*70)
    print("vLLM vs llama.cpp Benchmark (Native x86, No QEMU)")
    print("="*70)
    print()
    
    # Check if vLLM server is running
    try:
        health_url = args.vllm_url.replace('/v1/completions', '/health')
        response = requests.get(health_url, timeout=2)
        print("✅ vLLM server is running")
    except:
        print("❌ vLLM server not running!")
        print(f"\n   Start it with:")
        print(f"   vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cpu --port 8001")
        print()
        sys.exit(1)
    
    # Check if llama.cpp binary exists
    if not LLAMACPP_BIN.exists():
        print(f"❌ llama.cpp binary not found: {LLAMACPP_BIN}")
        print("\n   Build it first:")
        print("   cd dev_env/llama.cpp")
        print("   mkdir -p linux-build-x86-native && cd linux-build-x86-native")
        print("   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON ../")
        print("   cmake --build . -j$(nproc)")
        print()
        sys.exit(1)
    print(f"✅ llama.cpp binary found: {LLAMACPP_BIN}")
    
    if not LLAMACPP_MODEL.exists():
        print(f"❌ Model not found: {LLAMACPP_MODEL}")
        sys.exit(1)
    print(f"✅ Model found: {LLAMACPP_MODEL}")
    print()
    
    results = []
    
    for max_tokens in args.tokens:
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\nTest: Prompt {i+1}, {max_tokens} tokens")
            print(f"Prompt: '{prompt[:50]}...'")
            
            # Test vLLM
            print(f"  Testing vLLM CPU... ", end="", flush=True)
            vllm_result = benchmark_vllm(prompt, max_tokens)
            if vllm_result["success"]:
                print(f"✅ {vllm_result['throughput']:.2f} tok/s ({vllm_result['time']:.2f}s)")
            else:
                print(f"❌ {vllm_result.get('error', 'Unknown error')[:50]}")
            
            # Test llama.cpp
            print(f"  Testing llama.cpp... ", end="", flush=True)
            llamacpp_result = benchmark_llamacpp(prompt, max_tokens)
            if llamacpp_result["success"]:
                print(f"✅ {llamacpp_result['throughput']:.2f} tok/s ({llamacpp_result['time']:.2f}s)")
            else:
                print(f"❌ {llamacpp_result.get('error', 'Unknown error')[:50]}")
            
            # Store results
            results.append({
                "prompt_id": i+1,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "vllm": vllm_result,
                "llamacpp": llamacpp_result
            })
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    vllm_successes = [r for r in results if r["vllm"]["success"]]
    llamacpp_successes = [r for r in results if r["llamacpp"]["success"]]
    
    if vllm_successes:
        vllm_avg_throughput = sum(r["vllm"]["throughput"] for r in vllm_successes) / len(vllm_successes)
        vllm_avg_latency = sum(r["vllm"]["latency_per_token"] for r in vllm_successes) / len(vllm_successes)
        print(f"\nvLLM CPU:")
        print(f"  Average throughput: {vllm_avg_throughput:.2f} tokens/sec")
        print(f"  Average latency: {vllm_avg_latency:.0f} ms/token")
        print(f"  Success rate: {len(vllm_successes)}/{len(results)}")
    else:
        print("\nvLLM CPU: No successful tests")
    
    if llamacpp_successes:
        llamacpp_avg_throughput = sum(r["llamacpp"]["throughput"] for r in llamacpp_successes) / len(llamacpp_successes)
        llamacpp_avg_latency = sum(r["llamacpp"]["latency_per_token"] for r in llamacpp_successes) / len(llamacpp_successes)
        print(f"\nllama.cpp (native x86):")
        print(f"  Average throughput: {llamacpp_avg_throughput:.2f} tokens/sec")
        print(f"  Average latency: {llamacpp_avg_latency:.0f} ms/token")
        print(f"  Success rate: {len(llamacpp_successes)}/{len(results)}")
    else:
        print("\nllama.cpp: No successful tests")
    
    if vllm_successes and llamacpp_successes:
        ratio = llamacpp_avg_throughput / vllm_avg_throughput
        vllm_percent = (1/ratio)*100
        
        print(f"\n📊 Performance Comparison:")
        print(f"  llama.cpp is {ratio:.1f}x faster than vLLM CPU")
        print(f"  vLLM CPU is {vllm_percent:.1f}% of llama.cpp performance")
        
        # Decision guidance
        print("\n" + "="*70)
        print("Decision Guidance for RISC-V Compilation")
        print("="*70)
        if vllm_percent >= 80:
            print("✅ vLLM CPU is competitive with llama.cpp (>80%)")
            print("   → Definitely worth exploring RISC-V compilation")
        elif vllm_percent >= 50:
            print("⚠️  vLLM CPU is 50-80% of llama.cpp performance")
            print("   → Marginal case, consider project timeline and goals")
        elif vllm_percent >= 20:
            print("⚠️  vLLM CPU is 20-50% of llama.cpp performance")
            print("   → Probably not worth RISC-V compilation effort")
        else:
            print("❌ vLLM CPU is <20% of llama.cpp performance")
            print("   → NOT worth RISC-V compilation effort")
            print("   → Focus on agentic frameworks (Phase 6) instead")
        
        print(f"\nEstimated effort to compile vLLM for RISC-V: 3-4 weeks")
        print(f"Alternative: Complete Phase 6-8 in same timeframe")
        
        # Save detailed results
        output_file = Path("vllm_tests/benchmark_results.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "vllm_avg_throughput": vllm_avg_throughput if vllm_successes else 0,
                    "llamacpp_avg_throughput": llamacpp_avg_throughput if llamacpp_successes else 0,
                    "ratio": ratio if vllm_successes and llamacpp_successes else 0,
                    "vllm_percent": vllm_percent if vllm_successes and llamacpp_successes else 0
                },
                "detailed_results": results
            }, f, indent=2)
        print(f"\n📄 Detailed results saved to: {output_file}")
        
        return 0 if (vllm_successes and llamacpp_successes) else 1
    else:
        print("\n❌ Insufficient data for comparison")
        return 1


if __name__ == "__main__":
    sys.exit(main())
