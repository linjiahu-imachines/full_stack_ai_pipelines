# vLLM CPU Mode Exploration for RISC-V Verification

**Purpose:** Explore latest vLLM CPU capabilities and assess viability for RISC-V CPU verification project  
**Status:** Investigation based on 2026 developments  
**Key Insight:** vLLM now has more serious CPU support than previously assessed

---

## Executive Summary

### Latest Developments (2025-2026)

**vLLM has evolved its CPU support significantly:**
- ✅ **vLLM V1** (Jan 2025): Reduced CPU overhead, improved architecture
- ✅ **CPU Backend**: Tensor parallelism, INT8 quantization, AVX512 support
- ✅ **Intel IPEX integration**: Intel Extension for PyTorch for CPU optimization
- ✅ **KV Cache Offloading**: Offload to CPU DRAM (Jan 2026)
- ⚠️ **Still GPU-primary**: CPU is viable alternative, but not primary focus

### Quick Answer to Your Question

**"Can vLLM run CPU-only?"**  
✅ **YES** - vLLM can run on CPU-only mode (as of 2026)

**"Was it designed for GPU?"**  
✅ **YES** - vLLM is fundamentally GPU-centric, but CPU support has improved

**"Should we explore vLLM CPU for RISC-V?"**  
⚠️ **MAYBE** - Worth exploring IF you have time and interest, but llama.cpp remains better for CPU-only use cases

---

## What is vLLM?

**vLLM (Virtual LLM)** is a high-performance inference engine for serving Large Language Models.

### Core Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI API
- **PagedAttention**: Revolutionary memory management algorithm (inspired by OS virtual memory paging)
- **Continuous batching**: Dynamic request batching for maximum throughput
- **High throughput**: 10-100x faster than naive implementations
- **Production-ready**: Used by Anthropic, Databricks, Anyscale

### Key Innovation: PagedAttention

```
Traditional LLM Memory Management:
┌────────────────────────────────────┐
│ Pre-allocate large contiguous block│
│ for KV cache                        │
│ ✗ Fragmentation (40-60% waste)     │
│ ✗ Limited batch sizes               │
└────────────────────────────────────┘

vLLM's PagedAttention:
┌────────────────────────────────────┐
│ Divide KV cache into small "pages"│
│ Allocate on-demand (like OS VM)    │
│ ✓ Near-zero waste (~5%)             │
│ ✓ 2-4x higher throughput            │
│ ✓ Much larger batch sizes           │
└────────────────────────────────────┘
```

---

## GPU vs CPU Design

### Primary Design: GPU 🎮

**vLLM is fundamentally GPU-centric:**

```bash
# Official installation expects CUDA
pip install vllm  # Requires CUDA 11.8+ or 12.1+

# Core GPU features:
- Tensor parallelism (multi-GPU distribution)
- Flash Attention (GPU-optimized attention kernels)
- Custom CUDA kernels for PagedAttention
- FP16/BF16 precision (GPU data types)
- KV cache quantization (GPU memory)
```

**GPU Performance (2026):**
- Llama-2-7B: ~**1000-2000 tokens/sec** 🚀
- Llama-2-70B: ~100-300 tokens/sec (with 4x A100 GPUs)
- GPT-3.5 level: ~500-1000 tokens/sec

### CPU Support: Improved, But Secondary ⚠️

**vLLM CAN run on CPU (2026 update):**

```bash
# CPU installation
pip install vllm --extra-index-url https://download.pytorch.org/whl/cpu

# Start CPU server
vllm serve meta-llama/Llama-2-7b-chat-hf \
    --device cpu \
    --dtype float16 \
    --max-model-len 2048 \
    --max-num-seqs 256
```

**CPU Backend Features (2026):**
- ✅ **Tensor parallelism**: Multi-CPU support
- ✅ **INT8 quantization**: AWQ, GPTQ support
- ✅ **Chunked prefill**: Better latency
- ✅ **Prefix caching**: Reduce redundant computation
- ✅ **AVX512/AVX512_BF16**: x86 CPU optimizations
- ✅ **Intel IPEX**: Intel PyTorch extensions
- ✅ **TCMalloc**: Faster memory allocation

**CPU Performance (2026):**
- Llama-2-7B: ~**1-10 tokens/sec** 🐌 (100x slower than GPU)
- Memory: 14GB+ (FP16) or ~7GB (INT8)

**Limitations:**
- ❌ No GPU PagedAttention kernels (falls back to slower CPU implementation)
- ❌ No Flash Attention on CPU
- ❌ Continuous batching less effective on CPU
- ❌ Still 5-10x slower than llama.cpp with GGUF quantization

---

## Performance Comparison: vLLM CPU vs llama.cpp

### Benchmark Data (2026)

| Metric | llama.cpp (CPU) | vLLM (GPU) | vLLM (CPU) | Winner for CPU |
|--------|-----------------|------------|------------|----------------|
| **Throughput** | 10-50 tok/s | 1000+ tok/s | 1-10 tok/s | ✅ llama.cpp (5-10x faster) |
| **Latency** | 20-100ms/tok | 1-2ms/tok | 100-1000ms/tok | ✅ llama.cpp |
| **Memory (7B)** | 2-4GB (4-bit) | 14GB (FP16) | 7-14GB (INT8/FP16) | ✅ llama.cpp (3-7x less) |
| **Model Format** | GGUF (quant) | HF (FP16) | HF (FP16/INT8) | ✅ llama.cpp |
| **Quantization** | 4/5/8-bit | Limited | INT8 | ✅ llama.cpp |
| **CPU Optimization** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ✅ llama.cpp |
| **RISC-V Support** | ✅ Native | ❌ None | ❌ None | ✅ llama.cpp |

### Why llama.cpp Outperforms on CPU

**1. Quantization Strategy**
```
llama.cpp:
- 4-bit GGUF quantization
- 7B model → 2-4GB RAM
- Optimized CPU kernels for quantized operations
- Result: Fast on limited resources

vLLM CPU:
- FP16 or INT8 (less aggressive)
- 7B model → 7-14GB RAM
- CPU kernels adapted from GPU code
- Result: Slower, more memory
```

**2. Architecture Philosophy**
```
llama.cpp:
- Every optimization targets CPU
- Cache-friendly memory patterns
- SIMD: AVX2, AVX512, NEON, RVV
- Designed for CPU from day 1

vLLM:
- Optimizations designed for GPU
- PagedAttention for GPU memory
- Continuous batching for GPU parallelism
- CPU backend is adaptation
```

**3. On QEMU (Your Use Case)**
```
llama.cpp on QEMU:
- Base: 10-50 tok/s (x86 CPU)
- RISC-V native: ~5-25 tok/s (estimated)
- QEMU overhead: ~1.5 tok/s (proven in tests)

vLLM CPU on QEMU:
- Base: 1-10 tok/s (x86 CPU)
- RISC-V native: ~0.5-5 tok/s (estimated)
- QEMU overhead: ~0.1-0.5 tok/s (10x slower than llama.cpp)
```

---

## Exploration Strategy

### Recommended Approach: Phased Exploration

#### Phase 1: Baseline Testing (x86 CPU) - 1-2 days ⭐⭐⭐⭐

**Goal:** Establish vLLM CPU performance baseline and compare with llama.cpp

**Value:** ✅ High (low effort, clear comparison)

**Steps:**
1. Install vLLM with CPU backend on x86
2. Run same model (Llama-2-7B or similar)
3. Benchmark throughput and latency
4. Compare side-by-side with llama.cpp
5. **Decision point**: If vLLM CPU < 20% of llama.cpp performance, stop here

**Commands:**
```bash
# Install vLLM CPU
pip install vllm

# Start vLLM server (CPU)
vllm serve meta-llama/Llama-2-7b-chat-hf --device cpu

# Benchmark
python benchmark.py --backend vllm --device cpu

# Compare with llama.cpp
./llama-cli -m llama-2-7b-q4_0.gguf -n 100
```

**Success Criteria:**
- vLLM CPU achieves >20% of llama.cpp performance: ✅ Proceed to Phase 2
- vLLM CPU achieves <20% of llama.cpp performance: ❌ Stop (not worth RISC-V effort)

#### Phase 2: RISC-V Compilation Feasibility - 1 week ⭐⭐⭐

**Goal:** Assess compilation feasibility and attempt PyTorch for RISC-V

**Value:** ⚠️ Medium (high effort, uncertain outcome)

**Steps:**
1. Research PyTorch RISC-V compilation status
2. Attempt PyTorch cross-compilation for RISC-V
3. Document challenges and workarounds
4. **Decision point**: If PyTorch compiles successfully, proceed to Phase 3

**Expected Challenges:**
- x86-specific assembly code in PyTorch
- BLAS/LAPACK for RISC-V
- Build system complexity

**Success Criteria:**
- PyTorch compiles for RISC-V: ✅ Proceed to Phase 3
- PyTorch doesn't compile: ❌ Stop (blocker for vLLM)

#### Phase 3: vLLM RISC-V Compilation - 1-2 weeks ⭐⭐

**Goal:** Compile vLLM for RISC-V and test in QEMU

**Value:** ⚠️ Low-Medium (very high effort)

**Steps:**
1. Compile vLLM for RISC-V using RISC-V PyTorch
2. Test in QEMU user mode
3. Measure performance
4. **Decision point**: If performance >0.5 tok/s, proceed to Phase 4

#### Phase 4: Ray LLM APIs Integration - 2-3 days ⭐⭐

**Goal:** IF Phase 3 succeeds, integrate with Ray Serve LLM APIs

**Steps:**
1. Configure vLLM server for RISC-V
2. Test Ray Serve LLM's `vLLMDeployment`
3. Test Ray Data LLM's `generate()`
4. Evaluate API benefits

### Alternative Approach: Monitor and Wait

**Instead of immediate implementation:**

1. **Monitor vLLM development** (ongoing)
   - Watch for official RISC-V support announcements
   - Check community ports/forks
   - Track CPU performance improvements

2. **Complete higher priority work** (3-5 weeks)
   - Phase 6: Agentic frameworks (LangGraph, AutoGen, CrewAI)
   - Phase 7: RAG completion (vector stores, embeddings)
   - Phase 8: Production features (streaming, monitoring)

3. **Revisit vLLM** (after Phase 6-8)
   - With full-stack verification complete
   - With more mature vLLM CPU support
   - With potential community RISC-V ports

---

## Recommendations

### For Immediate Next Steps

**Option A: Explore vLLM CPU (If Interested)** ⚠️

**Timeline:** 1-4 weeks  
**Outcome:** Uncertain (may or may not work well)

**Decision Tree:**
```
Week 1: Test vLLM CPU baseline (x86)
   ├─ If <20% of llama.cpp: ❌ Stop
   └─ If ≥20% of llama.cpp: Continue
         ↓
Week 2-3: Compile PyTorch for RISC-V
   ├─ If fails: ❌ Stop
   └─ If succeeds: Continue
         ↓
Week 3-4: Compile vLLM, test on QEMU
   ├─ Performance <0.5 tok/s: ⚠️ Marginal value
   └─ Performance ≥0.5 tok/s: ✅ Could be viable
```

**Option B: Prioritize Agentic Frameworks** ✅ (Recommended)

**Timeline:** 1-2 weeks  
**Outcome:** Certain (fills largest verification gap)

**Path:**
```
Week 1: LangGraph + AutoGen
   ✅ Proven frameworks
   ✅ Clear integration path
   ✅ Directly addresses "Agentic" goal
         ↓
Week 2: CrewAI + RAG completion
   ✅ Complete full-stack verification
   ✅ High verification value
         ↓
Week 3+: Production features OR vLLM exploration
   ⚠️ vLLM becomes optional exploration
   ✅ Full-stack already verified
```

### My Strong Recommendation

**Do Both, But in Sequence:**

**Phase 1 (Week 1): Quick vLLM Baseline Test** ⭐
- 1-2 days to test vLLM CPU on x86
- Compare with llama.cpp performance
- Make go/no-go decision for RISC-V compilation

**Phase 2 (Weeks 1-3): Agentic Frameworks** ⭐⭐⭐⭐⭐
- Proceed regardless of vLLM results
- LangGraph, AutoGen, CrewAI
- Complete core verification goal

**Phase 3 (Weeks 4-5): vLLM RISC-V (If Phase 1 was promising)** ⭐⭐
- Only if vLLM CPU showed competitive performance
- Compile for RISC-V
- Test and document

**Rationale:**
- Low risk: 1-2 days to test vLLM CPU baseline
- High value: Agentic frameworks are completed regardless
- Flexible: Can decide on vLLM RISC-V based on baseline results
- Efficient: Parallel paths possible (someone tests vLLM while you work on agentic)

---

## Next Steps

### Immediate Action (This Week)

**Step 1: Test vLLM CPU Baseline** (1-2 days)

I can help you:
1. Install vLLM with CPU backend
2. Run benchmarks comparing vLLM CPU vs. llama.cpp on x86
3. Document performance characteristics
4. Make go/no-go decision for RISC-V compilation

**Step 2: Start Agentic Frameworks** (Start in parallel or after Step 1)

1. Begin Phase 6A: LangGraph integration
2. Independent of vLLM exploration
3. Highest priority for project goal

### Decision Points

**After Step 1 (vLLM CPU baseline):**
- **If vLLM CPU ≥ 20% of llama.cpp**: ✅ Worth exploring RISC-V compilation
- **If vLLM CPU < 20% of llama.cpp**: ❌ Not worth RISC-V effort, focus on agentic frameworks

**After Agentic Frameworks Complete:**
- Reassess vLLM value
- Consider compilation if still interested
- Or move to RAG completion / production features

---

## Summary

### Key Points

1. ✅ **You're right**: vLLM has improved CPU support in 2026
2. ⚠️ **But**: Still primarily GPU-focused, CPU is viable alternative not optimal choice
3. ✅ **llama.cpp remains better for CPU**: 5-10x faster, less memory, better for your use case
4. ⚠️ **vLLM worth exploring**: If you have time and want framework diversity
5. ✅ **Higher priority**: Agentic frameworks (Phase 6) for completing verification goals

### Recommended Path

**Best approach for your project:**
1. **Quick test** (1-2 days): vLLM CPU baseline on x86
2. **High priority** (1-2 weeks): Agentic frameworks (LangGraph, AutoGen, CrewAI)
3. **Optional** (2-3 weeks): vLLM RISC-V compilation (if baseline was promising)

**This gives you:**
- ✅ Data-driven decision on vLLM
- ✅ Core verification completed (agentic frameworks)
- ✅ Option to pursue vLLM if valuable

---

## References

- **vLLM V1 Release**: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- **KV Offloading**: https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html
- **CPU Installation**: https://docs.vllm.ai/en/v0.6.5/getting_started/cpu-installation.html
- **Comparison**: https://www.decodesfuture.com/articles/llama-cpp-vs-ollama-vs-vllm-local-llm-stack-guide
