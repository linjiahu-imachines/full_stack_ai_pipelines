# vLLM Native Testing Guide (Bare Metal Linux)

**Purpose:** Test vLLM on native x86_64 Linux and compare with llama.cpp to establish performance baseline  
**Environment:** Native Linux (no QEMU), CPU-only  
**Goal:** Data-driven decision: Is vLLM worth pursuing for RISC-V?

---

## Overview

**Test Plan:**
1. Setup vLLM on your Linux machine (CPU mode)
2. Setup llama.cpp native x86 build (for comparison)
3. Run benchmarks on same model
4. Compare performance (throughput, latency, memory)
5. Make decision: Pursue vLLM for RISC-V or not?

**Expected Timeline:** 2-4 hours for setup and testing

---

## Prerequisites

**System Requirements:**
- x86_64 Linux system (your current machine)
- Python 3.10+ (you have 3.11.10 ✅)
- 16GB+ RAM recommended
- ~20GB disk space

**Check your system:**
```bash
# CPU info
lscpu | grep -E "Model name|CPU\(s\)|Thread|AVX"

# Memory
free -h

# Disk space
df -h /home/linhu

# Python
python --version  # Should be 3.10+
```

---

## Part 1: vLLM Setup (CPU Mode)

### Step 1: Install vLLM for CPU

```bash
# Create a test directory
cd /home/linhu/repo/iminn-tools
mkdir -p vllm_tests
cd vllm_tests

# Install vLLM with CPU support
pip install vllm

# Verify installation
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

**Expected output:** `vLLM version: 0.x.x`

### Step 2: Download a Test Model

**Use a small model for quick testing:**

```bash
# Option 1: Use HuggingFace cache (if models already downloaded)
ls ~/.cache/huggingface/hub/

# Option 2: Use a small model (TinyLlama)
# vLLM will download automatically on first use
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

**Alternative: Use same model as your llama.cpp tests**
```bash
# If you want apple-to-apple comparison
# You'll need HuggingFace format of stories15M or similar
# For now, use TinyLlama (1.1B params, similar size to stories15M)
```

### Step 3: Start vLLM Server (CPU Mode)

```bash
# Start vLLM server in CPU mode
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --device cpu \
    --dtype float16 \
    --max-model-len 2048 \
    --port 8001

# Expected output:
# INFO: Started server process
# INFO: Waiting for application startup.
# INFO: Application startup complete.
# INFO: Uvicorn running on http://0.0.0.0:8001
```

**Keep this terminal open** (vLLM server running)

### Step 4: Test vLLM Inference

**In a new terminal:**

```bash
# Test with curl
curl http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Hello, how are you?",
        "max_tokens": 100,
        "temperature": 0.8
    }'
```

**Or use Python:**
```python
import requests
import time

start = time.time()
response = requests.post(
    "http://localhost:8001/v1/completions",
    json={
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Hello, how are you?",
        "max_tokens": 100,
        "temperature": 0.8
    }
)
elapsed = time.time() - start

data = response.json()
print(f"Response: {data['choices'][0]['text']}")
print(f"Time: {elapsed:.2f}s")
print(f"Tokens generated: {data['usage']['completion_tokens']}")
print(f"Throughput: {data['usage']['completion_tokens']/elapsed:.2f} tokens/sec")
```

---

## Part 2: llama.cpp Native Setup (Comparison)

### Step 1: Build llama.cpp for Native x86

```bash
# You already have RISC-V build, now build native x86
cd /home/linhu/repo/iminn-tools/dev_env/llama.cpp

# Create x86 native build directory
mkdir -p linux-build-x86-native
cd linux-build-x86-native

# Configure for native x86 (no cross-compilation)
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=ON \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    ../

# Build
cmake --build . -j$(nproc)

# Test
./bin/llama-cli --version
```

### Step 2: Test llama.cpp Inference

```bash
# Test with same prompt
cd /home/linhu/repo/iminn-tools/dev_env/llama.cpp

./linux-build-x86-native/bin/llama-cli \
    -m models/stories15M-q4_0.gguf \
    -p "Hello, how are you?" \
    -n 100 \
    --seed 42 \
    -ngl 0

# Note the performance metrics at the end
```

---

## Part 3: Benchmark Scripts

### Benchmark Script for Both Systems

Create `/home/linhu/repo/iminn-tools/scripts/benchmark_vllm_vs_llamacpp.py`:

```python
#!/usr/bin/env python3
"""
Benchmark vLLM (CPU) vs llama.cpp (native x86) on the same hardware.

Usage:
    python scripts/benchmark_vllm_vs_llamacpp.py
"""

import subprocess
import requests
import time
import json
from pathlib import Path
import sys

# Configuration
VLLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
VLLM_URL = "http://localhost:8001/v1/completions"
LLAMACPP_BIN = "/home/linhu/repo/iminn-tools/dev_env/llama.cpp/linux-build-x86-native/bin/llama-cli"
LLAMACPP_MODEL = "/home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/stories15M-q4_0.gguf"

# Test prompts
TEST_PROMPTS = [
    "Hello, how are you?",
    "Write a short story about a robot.",
    "Explain what RISC-V is in one sentence.",
    "What is the future of AI?",
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
                "text": data['choices'][0]['text'][:100]
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def benchmark_llamacpp(prompt: str, max_tokens: int) -> dict:
    """Benchmark llama.cpp inference."""
    try:
        # Write prompt to temp file
        prompt_file = Path("/tmp/bench_prompt.txt")
        prompt_file.write_text(prompt)
        
        cmd = [
            LLAMACPP_BIN,
            "-m", LLAMACPP_MODEL,
            "-p", prompt,
            "-n", str(max_tokens),
            "--seed", "42",
            "-ngl", "0",
            "--no-warmup"
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
            # Parse output for tokens/sec
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
    print("="*70)
    print("vLLM vs llama.cpp Benchmark (Native x86, No QEMU)")
    print("="*70)
    print()
    
    # Check if vLLM server is running
    try:
        response = requests.get(f"{VLLM_URL.replace('/v1/completions', '/health')}", timeout=2)
        print("✅ vLLM server is running")
    except:
        print("❌ vLLM server not running!")
        print(f"   Start it with: vllm serve {VLLM_MODEL} --device cpu --port 8001")
        sys.exit(1)
    
    # Check if llama.cpp binary exists
    if not Path(LLAMACPP_BIN).exists():
        print(f"❌ llama.cpp binary not found: {LLAMACPP_BIN}")
        print("   Build it first (see Part 2 of guide)")
        sys.exit(1)
    print("✅ llama.cpp binary found")
    print()
    
    results = []
    
    for max_tokens in TOKEN_LENGTHS:
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\nTest: Prompt {i+1}, {max_tokens} tokens")
            print(f"Prompt: {prompt[:50]}...")
            
            # Test vLLM
            print(f"  Testing vLLM... ", end="", flush=True)
            vllm_result = benchmark_vllm(prompt, max_tokens)
            if vllm_result["success"]:
                print(f"✅ {vllm_result['throughput']:.2f} tok/s ({vllm_result['time']:.2f}s)")
            else:
                print(f"❌ {vllm_result['error']}")
            
            # Test llama.cpp
            print(f"  Testing llama.cpp... ", end="", flush=True)
            llamacpp_result = benchmark_llamacpp(prompt, max_tokens)
            if llamacpp_result["success"]:
                print(f"✅ {llamacpp_result['throughput']:.2f} tok/s ({llamacpp_result['time']:.2f}s)")
            else:
                print(f"❌ {llamacpp_result['error']}")
            
            # Store results
            results.append({
                "prompt_id": i+1,
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
        print(f"\nvLLM CPU:")
        print(f"  Average throughput: {vllm_avg_throughput:.2f} tokens/sec")
        print(f"  Success rate: {len(vllm_successes)}/{len(results)}")
    
    if llamacpp_successes:
        llamacpp_avg_throughput = sum(r["llamacpp"]["throughput"] for r in llamacpp_successes) / len(llamacpp_successes)
        print(f"\nllama.cpp (native x86):")
        print(f"  Average throughput: {llamacpp_avg_throughput:.2f} tokens/sec")
        print(f"  Success rate: {len(llamacpp_successes)}/{len(results)}")
    
    if vllm_successes and llamacpp_successes:
        ratio = llamacpp_avg_throughput / vllm_avg_throughput
        print(f"\nllama.cpp is {ratio:.1f}x faster than vLLM CPU")
        print(f"\nvLLM CPU is {(1/ratio)*100:.1f}% of llama.cpp performance")
        
        # Decision guidance
        print("\n" + "="*70)
        print("Decision Guidance")
        print("="*70)
        if ratio < 2:
            print("✅ vLLM CPU is competitive with llama.cpp (<2x slower)")
            print("   → Worth exploring RISC-V compilation")
        elif ratio < 5:
            print("⚠️  vLLM CPU is 2-5x slower than llama.cpp")
            print("   → Marginal case, consider other factors")
        else:
            print("❌ vLLM CPU is 5x+ slower than llama.cpp")
            print("   → NOT worth RISC-V compilation effort")
        
        # Save detailed results
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: benchmark_results.json")


if __name__ == "__main__":
    main()
```

---

## Part 2: llama.cpp Native x86 Setup

### Build Native x86 Binary

**If you don't have a native x86 build yet:**

```bash
cd /home/linhu/repo/iminn-tools/dev_env/llama.cpp

# Create native x86 build
mkdir -p linux-build-x86-native
cd linux-build-x86-native

# Configure for native x86
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=ON \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_F16C=ON \
    ../

# Build
cmake --build . -j$(nproc)

# Test
./bin/llama-cli --version
```

**Expected time:** 5-10 minutes

### Quick Test

```bash
# Quick test
./bin/llama-cli \
    -m ../models/stories15M-q4_0.gguf \
    -p "Hello, world!" \
    -n 32 \
    --seed 42
```

---

## Part 3: Running the Benchmark

### Terminal 1: Start vLLM Server

```bash
cd /home/linhu/repo/iminn-tools/vllm_tests

# Start vLLM (leave running)
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --device cpu \
    --dtype float16 \
    --port 8001

# Wait for "Application startup complete"
```

### Terminal 2: Run Benchmark

```bash
cd /home/linhu/repo/iminn-tools

# Run benchmark script
python scripts/benchmark_vllm_vs_llamacpp.py
```

**Expected output:**
```
======================================================================
vLLM vs llama.cpp Benchmark (Native x86, No QEMU)
======================================================================

✅ vLLM server is running
✅ llama.cpp binary found

Test: Prompt 1, 32 tokens
Prompt: Hello, how are you?...
  Testing vLLM... ✅ 2.5 tok/s (12.8s)
  Testing llama.cpp... ✅ 15.3 tok/s (2.1s)

Test: Prompt 1, 64 tokens
Prompt: Hello, how are you?...
  Testing vLLM... ✅ 2.3 tok/s (27.8s)
  Testing llama.cpp... ✅ 14.8 tok/s (4.3s)

...

======================================================================
Summary
======================================================================

vLLM CPU:
  Average throughput: 2.4 tokens/sec
  Success rate: 12/12

llama.cpp (native x86):
  Average throughput: 15.1 tokens/sec
  Success rate: 12/12

llama.cpp is 6.3x faster than vLLM CPU

vLLM CPU is 15.9% of llama.cpp performance

======================================================================
Decision Guidance
======================================================================
❌ vLLM CPU is 5x+ slower than llama.cpp
   → NOT worth RISC-V compilation effort
```

---

## Part 4: Analysis and Decision

### Performance Metrics to Collect

**For each system, measure:**
1. **Throughput**: Tokens per second
2. **Latency**: Time per token (ms)
3. **Memory**: RAM usage during inference
4. **Startup**: Time to first token
5. **Stability**: Success rate across multiple tests

### Decision Criteria

**If vLLM CPU achieves:**

| Performance Ratio | Recommendation | Rationale |
|------------------|----------------|-----------|
| **> 80% of llama.cpp** | ✅ Definitely explore RISC-V | Competitive, worth effort |
| **50-80% of llama.cpp** | ⚠️ Consider exploring | Marginal, depends on goals |
| **20-50% of llama.cpp** | ⚠️ Probably not worth it | Significant performance gap |
| **< 20% of llama.cpp** | ❌ Don't pursue | Too slow, not worth effort |

**Additional factors:**
- **Memory usage**: If vLLM uses 3x+ more memory, concern for RISC-V
- **Stability**: If vLLM has errors/crashes, not production-ready
- **Compilation complexity**: PyTorch for RISC-V is non-trivial (1-2 weeks)

### Expected Results (Based on 2026 Data)

**Likely Outcome:**
```
llama.cpp (native x86): 10-50 tokens/sec
vLLM CPU (native x86):  1-10 tokens/sec

Ratio: llama.cpp is 5-10x faster

Decision: ❌ NOT worth RISC-V compilation
```

**Why?**
- llama.cpp uses 4-bit GGUF quantization (very efficient)
- vLLM uses FP16/INT8 (less aggressive quantization)
- llama.cpp designed for CPU from ground up
- vLLM CPU is adapted from GPU version

---

## Quick Start Commands

**Complete test in one session:**

```bash
# Terminal 1: Start vLLM
cd /home/linhu/repo/iminn-tools/vllm_tests
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cpu --port 8001

# Terminal 2: Run benchmarks
cd /home/linhu/repo/iminn-tools
python scripts/benchmark_vllm_vs_llamacpp.py

# Review results
cat vllm_tests/benchmark_results.json
```

---

## Expected Timeline

**Setup and testing:**
- vLLM installation: 10-15 minutes
- llama.cpp native build: 5-10 minutes
- Benchmark script creation: 15-20 minutes
- Running benchmarks: 20-30 minutes
- Analysis: 10-15 minutes

**Total: 1-2 hours** to complete baseline testing

**After baseline:**
- **If vLLM CPU competitive**: Decide on RISC-V compilation (3-4 weeks)
- **If vLLM CPU not competitive**: Focus on agentic frameworks (Phase 6)

---

## Next Steps After Baseline Testing

### Scenario 1: vLLM CPU is Competitive (>50% of llama.cpp)

**Consider:**
1. PyTorch RISC-V compilation (1-2 weeks)
2. vLLM RISC-V compilation (1 week)
3. Testing on QEMU
4. Integration with Ray Serve LLM

**Decision:** Depends on project priorities and timeline

### Scenario 2: vLLM CPU is Not Competitive (<50% of llama.cpp)

**Recommended:**
1. ✅ **Skip vLLM for RISC-V** (not worth effort)
2. ✅ **Focus on Phase 6**: Agentic frameworks (LangGraph, AutoGen, CrewAI)
3. ✅ **Complete Phase 7**: RAG (vector stores, embeddings)
4. ✅ **Complete Phase 8**: Production features (streaming, monitoring)

**Rationale:**
- Higher ROI from agentic frameworks
- vLLM doesn't add verification value if performance is poor
- Complete full-stack verification faster

---

## References

- **vLLM Documentation**: https://docs.vllm.ai/
- **vLLM CPU Installation**: https://docs.vllm.ai/en/v0.6.5/getting_started/cpu-installation.html
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **llama.cpp**: Your existing setup in `dev_env/llama.cpp/`
- **Benchmark Comparison**: https://www.decodesfuture.com/articles/llama-cpp-vs-ollama-vs-vllm-local-llm-stack-guide
