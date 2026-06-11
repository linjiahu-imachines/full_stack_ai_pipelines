# vLLM vs Transformers: CPU-Only Inference Comparison Report

## Executive Summary

**Report Date:** February 11, 2026  
**Test Environment:** Linux (60 CPU cores)  
**Model Tested:** facebook/opt-125m (125M parameters)  
**Test Duration:** ~90 seconds total

### Key Finding
**Direct HuggingFace Transformers is 8-11x faster than vLLM on CPU for this workload.**

---

## Test Configuration

### Hardware & Environment
| Component | Details |
|-----------|---------|
| CPU Cores | 60 physical cores |
| Memory | 62.88 GiB allocated for KV cache |
| OS | Linux (kernel 6.8.0-47-generic) |
| Python | 3.12.3 |

### Software Versions
| Library | vLLM Test | Transformers Test |
|---------|-----------|-------------------|
| Inference Engine | vLLM 0.15.1+cpu | Transformers 4.57.6 |
| PyTorch | 2.10.0+cpu | 2.10.0+cpu |
| Model | facebook/opt-125m | facebook/opt-125m |
| Data Type | float16 | float32 (default) |

---

## Performance Comparison

### Test 1: Single Inference (3 prompts, 50 tokens each)

| Metric | vLLM | Transformers | Winner | Speedup |
|--------|------|--------------|--------|---------|
| **Total Time** | 21.79s | 2.67s | ✅ Transformers | **8.16x faster** |
| **Time per Prompt** | 7.26s | 0.89s | ✅ Transformers | **8.16x faster** |
| **Input Speed** | 0.87 tok/s | N/A | - | - |
| **Output Speed** | 5.33 tok/s | N/A | - | - |

**Analysis:**
- Transformers completed all 3 prompts in the time vLLM took for just 1/3 of a prompt
- vLLM's overhead (initialization, scheduling, batching logic) dominates execution time
- For simple sequential inference, direct PyTorch is clearly superior

### Test 2: Batch Inference (5 prompts, 30 tokens each)

| Metric | vLLM | Transformers | Winner | Speedup |
|--------|------|--------------|--------|---------|
| **Total Time** | 9.70s | 0.87s | ✅ Transformers | **11.15x faster** |
| **Time per Prompt** | 1.94s | 0.17s | ✅ Transformers | **11.41x faster** |
| **Batch Efficiency** | 2.24x vs single | 5.24x vs single | ✅ Transformers | - |

**Analysis:**
- Both approaches benefit from batching, but transformers benefits more
- vLLM: 2.24x improvement with batching (7.26s → 1.94s per prompt)
- Transformers: 5.24x improvement with batching (0.89s → 0.17s per prompt)
- Transformers' native PyTorch batching is more efficient on CPU

---

## Detailed Output Comparison

### Prompt 1: "Hello, my name is"

**vLLM Output:**
> Shane and I live in the South Bay. I'm very happy to see you. I hope you are not in a hurry. Hi there, I'm Shane, and I live in the South Bay. I hope you are not in a hurry

**Transformers Output:**
> Hello, my name is Tashan. I'm 23 years old and have a 7 year old daughter who is a wonderful mother and sister. I'm a successful business owner with over 50+ years of experience. My business is a full-service catering company based in Los

**Observation:** Both produce coherent text, but content differs due to different random seeds and generation parameters.

### Prompt 2: "The capital of France is"

**vLLM Output:**
> also the capital of the Netherlands. For some reason, you're right

**Transformers Output:**
> The capital of France is known for its rich history, with a population of more than 50 million. The capital of the country is also known for its rich history...

**Observation:** vLLM produced a shorter, somewhat inaccurate response. Transformers produced more verbose output.

### Prompt 3: "Python is a programming language that"

**vLLM Output:**
> allows you to create a simple Python script with a few lines of code. You can use it to create an animated python script, along with creating a character on a screen. It can be used for a lot of different purposes. The question

**Transformers Output:**
> Python is a programming language that is used to create web applications. Python is a programming language that is used to create web applications. [repeated]

**Observation:** Both produce reasonable outputs. Transformers output shows repetition (common with small models).

---

## Performance Breakdown

### vLLM CPU Characteristics

**Strengths Observed:**
- ✅ Automatic thread binding across 60 CPU cores
- ✅ Large KV cache allocation (62.88 GiB)
- ✅ Batching does improve performance (2.24x)
- ✅ Progress monitoring and throughput metrics

**Weaknesses Observed:**
- ❌ High initialization overhead
- ❌ Scheduling/orchestration overhead dominates
- ❌ Per-request latency is much higher
- ❌ Batch efficiency lower than direct PyTorch

**Resource Usage:**
- 20 OpenMP threads per worker
- NUMA binding to optimize memory access
- Gloo backend for distributed coordination (overhead for single-node)

### Transformers Direct Approach

**Strengths Observed:**
- ✅ Minimal overhead
- ✅ Direct PyTorch execution
- ✅ Excellent batch efficiency (5.24x improvement)
- ✅ Very fast per-prompt latency
- ✅ Simple, straightforward code

**Weaknesses Observed:**
- ❌ No built-in serving infrastructure
- ❌ No automatic concurrency handling
- ❌ Manual batching required
- ❌ No KV cache optimization for long contexts

---

## Cost-Benefit Analysis

### Total Runtime Comparison

| Test | vLLM | Transformers | Time Saved |
|------|------|--------------|------------|
| Single (3 prompts) | 21.79s | 2.67s | 19.12s (87.7%) |
| Batch (5 prompts) | 9.70s | 0.87s | 8.83s (91.0%) |
| **Total** | **31.49s** | **3.54s** | **27.95s (88.8%)** |

**At Scale:**
- For 1,000 prompts (single): vLLM ~2h, Transformers ~15min (1h 45min saved)
- For 1,000 prompts (batch): vLLM ~32min, Transformers ~3min (29min saved)

---

## Use Case Recommendations

### ✅ Use Transformers (Direct PyTorch) When:

1. **CPU-only deployment** (primary finding)
2. **Small models** (<1B parameters)
3. **Low concurrency** (single user or batch processing)
4. **Minimal latency required**
5. **Simple inference patterns**
6. **Prototyping and development**

**Expected Performance:** 8-11x faster than vLLM on CPU

### ✅ Use vLLM When:

1. **GPU deployment** (vLLM's primary target)
2. **High concurrency** (many simultaneous users)
3. **Production API serving** (need OpenAI-compatible API)
4. **Large models** (>7B parameters)
5. **Long context windows** (PagedAttention benefits)
6. **Memory constrained** (efficient KV cache management)

**Note:** On GPU, vLLM typically provides 2-10x speedup over naive transformers usage.

---

## Technical Deep Dive

### Why is vLLM Slower on CPU?

1. **Initialization Overhead:**
   - Engine initialization: ~5-6s per model load
   - Thread binding and NUMA configuration
   - Distributed backend setup (even for single node)

2. **Scheduling Overhead:**
   - Request queuing and scheduling logic
   - Continuous batching infrastructure
   - Asynchronous execution coordination

3. **Optimization Mismatch:**
   - PagedAttention designed for GPU memory
   - CUDA-specific optimizations disabled on CPU
   - Kernel compilation overhead

4. **Architecture Complexity:**
   - Multiple worker processes
   - Inter-process communication
   - Request/response serialization

### Why is Transformers Faster on CPU?

1. **Direct Execution:**
   - No intermediate layers
   - Direct PyTorch operations
   - Minimal Python overhead

2. **CPU-Optimized PyTorch:**
   - Native PyTorch CPU kernels
   - MKL/oneDNN optimizations (when available)
   - Efficient memory access patterns

3. **Simple Batching:**
   - Native PyTorch batching
   - No scheduling overhead
   - Predictable execution path

---

## Recommendations for Different Scenarios

### Scenario 1: Research/Development (CPU)
**Recommended:** Transformers  
**Why:** Fastest iteration, simple code, minimal overhead

### Scenario 2: Single-User Application (CPU)
**Recommended:** Transformers  
**Why:** Best latency, simpler deployment

### Scenario 3: Batch Processing Pipeline (CPU)
**Recommended:** Transformers  
**Why:** 11x faster batch processing

### Scenario 4: Production API (CPU, Low Traffic)
**Recommended:** Transformers + FastAPI  
**Why:** Better performance, you can build API yourself

### Scenario 5: Production API (CPU, High Concurrency)
**Recommended:** Consider vLLM or Load Balancer + Multiple Transformers Workers  
**Why:** vLLM's request handling infrastructure might offset performance loss

### Scenario 6: Any GPU Scenario
**Recommended:** vLLM  
**Why:** vLLM excels on GPU with 2-10x speedup

---

## Conclusions

### Primary Conclusion
For CPU-only inference with small models (like opt-125m), **HuggingFace Transformers is 8-11x faster than vLLM** and should be the default choice for most use cases.

### Key Takeaways

1. ✅ **vLLM is not optimized for CPU inference** - Its design targets GPU acceleration
2. ✅ **Direct PyTorch/Transformers is highly efficient on CPU** - Minimal overhead, excellent performance
3. ✅ **Batching helps both approaches** - But transformers benefits more (5.24x vs 2.24x)
4. ✅ **vLLM's features come at a cost** - Scheduling, batching infrastructure adds overhead
5. ✅ **Choose the right tool for your deployment** - CPU vs GPU makes all the difference

### When Results Might Differ

This report's findings apply specifically to:
- CPU-only deployment
- Small models (~125M params)
- Single-node inference
- Low-to-medium concurrency

Results would differ with:
- GPU acceleration (vLLM would be faster)
- Larger models (>7B params)
- Very high concurrency (100+ concurrent users)
- Long context windows (>4K tokens)

---

## Appendix: Raw Test Outputs

### Full Output Files
- `outputs/vllm_output.txt` - Complete vLLM test output
- `outputs/transformers_output.txt` - Complete transformers test output

### Test Scripts
- `test_with_vllm.py` - vLLM test implementation
- `test_without_vllm.py` - Transformers test implementation
- `test_comparison.py` - Side-by-side comparison

### How to Reproduce
```bash
cd /home/linhu/projects/vllm_test
source venv/bin/activate

# Run individual tests
python test_with_vllm.py
python test_without_vllm.py

# Or run comparison
python test_comparison.py
```

---

**Report Generated:** February 11, 2026  
**Test Environment:** /home/linhu/projects/vllm_test/  
**Status:** ✅ All tests completed successfully
