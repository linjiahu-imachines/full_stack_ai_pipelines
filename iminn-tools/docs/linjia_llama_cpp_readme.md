# llama.cpp Architecture - Deep Dive

Based on analysis of the source code in `dev_env/llama.cpp/`, here's how llama.cpp works:

## **1. Core Architecture Overview**

llama.cpp is a C/C++ inference engine for Large Language Models with three main layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (llama-cli, llama-bench, test-backend-ops, etc.)           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      llama.cpp Layer                         │
│  (Model loading, KV cache, context, tokenization, etc.)     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      GGML Tensor Library                     │
│  (Computation graphs, tensor ops, backends, quantization)   │
└─────────────────────────────────────────────────────────────┘
```

## **2. GGML (The Foundation)**

**Location:** `ggml/src/ggml.c`, `ggml/include/ggml.h`

GGML is a tensor library implementing:

- **Computation Graphs**: Operations are defined as directed acyclic graphs (DAGs)
- **Lazy Evaluation**: Graphs are built first, then computed on demand
- **Automatic Differentiation**: Supports both forward and backward passes
- **Multi-dimensional Tensors**: Up to 4 dimensions with flexible strides

**Key Concepts:**

```c
// Define computation graph
struct ggml_context * ctx = ggml_init(params);
struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
struct ggml_tensor * result = ggml_add(ctx, a, b);  // Creates graph node

// Execute graph
ggml_graph_compute_with_ctx(ctx, graph, n_threads);
```

## **3. Backend System**

**Location:** `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-cpu/`

llama.cpp uses a pluggable backend architecture:

**Available Backends:**
- **CPU**: Generic C implementation + SIMD optimizations (AVX2, AVX512, NEON, etc.)
- **CPU-IMI**: Custom I-Machines RISC-V extensions (`ggml/src/ggml-cpu/imi/`)
- **CUDA**: NVIDIA GPU
- **Metal**: Apple Silicon
- **SYCL**: Intel GPUs
- **CANN**: Huawei Ascend
- **ROCm/HIP**: AMD GPUs
- **Vulkan**: Cross-platform GPU
- **REF**: Reference implementation for correctness testing

## **4. I-Machines Custom Backend (IMI)**

**Location:** `dev_env/llama.cpp/ggml/src/ggml-cpu/imi/`

This is the RISC-V custom implementation with I-Machines extensions:

**Key Files:**
- `imi.cpp` / `imi.h`: Main IMI backend registration and dispatch
- `opt-kernels.cpp`: Optimized RISC-V vector (RVV) kernels
- `generic-kernels.cpp`: Fallback scalar implementations
- `imi-common.h`: Custom data structures and RISC-V intrinsics

**Custom Features:**

1. **Custom Quantization Formats:**
   - `block_imi_q4_0x4`, `block_imi_q4_0x8`: Packed 4-bit quantization
   - `block_imi_q8_0x4`, `block_imi_q8_0x8`: Packed 8-bit quantization
   - `block_imi_mxfp4x4`, `block_imi_mxfp8x8`: MX floating point formats

2. **Optimized Kernels:**
   - `ggml_imi_gemv_*`: Matrix-vector multiplication (GEMV) for decode phase
   - `ggml_imi_gemm_*`: Matrix-matrix multiplication (GEMM) for prefill phase
   - Variants like `q4_0_q8_0_4x1`, `q8_0_q8_0_8x4` handle different quantization combos

3. **RISC-V Vector Extensions:**
   - Uses RVV intrinsics (`__riscv_*` functions)
   - Custom FP32→FP8 conversion with E4M3 format
   - VLEN=128 optimization (configurable vector length)

## **5. llama.cpp High-Level Components**

**Location:** `src/llama-*.cpp`

**Key Modules:**

| Module | File | Purpose |
|--------|------|---------|
| **Model** | `llama-model.cpp` | Load GGUF files, manage weights |
| **Context** | `llama-context.cpp` | Runtime state, batch processing |
| **KV Cache** | `llama-kv-cache.cpp` | Attention cache management |
| **Vocabulary** | `llama-vocab.cpp` | Tokenization (BPE, SentencePiece, etc.) |
| **Graph** | `llama-graph.cpp` | Build computation graphs for layers |
| **Sampling** | `llama-sampling.cpp` | Top-k, top-p, temperature, etc. |
| **Architecture** | `llama-arch.cpp` | Architecture-specific configs (Llama, GPT, Mamba, etc.) |

## **6. Quantization System**

**Location:** `ggml/src/ggml-quants.c`

Supports numerous quantization formats:

- **Float**: F32, F16, BF16
- **Integer**: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q6_K
- **K-Quantization**: Q2_K through Q6_K (higher quality)
- **IQ Quantization**: IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_NL (very low bit)
- **TQ**: TQ1_0, TQ2_0 (ternary quantization)
- **MXFP**: MXFP4, MXFP8 (microscaling formats for I-Machines)

**Quantization Strategy:**
- Weights (activations matrices): Quantized to save memory
- Activations (runtime): Usually F16/F32 for accuracy
- Compute: Mixed precision (e.g., Q4_0 weights × F32 activations → F32 output)

## **7. Inference Pipeline**

Here's how a forward pass works:

```
1. Load Model (GGUF file)
   └─> llama_model_loader::load()
   └─> Parse metadata, load tensors into backend memory

2. Create Context
   └─> llama_context_init()
   └─> Allocate KV cache, setup compute buffers

3. Tokenize Input
   └─> llama_tokenize()
   └─> Text → Token IDs using vocab

4. Process Tokens (Batch)
   └─> llama_decode()
   └─> Build computation graph for transformer layers

   For each layer:
   ├─> Attention (Q, K, V projections + scaled dot-product)
   ├─> KV Cache update
   ├─> MLP/FFN (feed-forward network)
   └─> Layer norm, residual connections

5. Sample Next Token
   └─> llama_sampler_sample()
   └─> Apply temperature, top-k, top-p, etc.

6. Repeat 3-5 (Autoregressive generation)
```

## **8. Key Optimizations**

1. **KV Cache**: Stores previous attention keys/values to avoid recomputation
2. **Flash Attention**: Memory-efficient attention implementation
3. **Batch Processing**: Process multiple sequences in parallel
4. **Continuous Batching**: Mix prefill and decode in same batch
5. **Quantization**: Reduce memory bandwidth and footprint
6. **SIMD**: Vectorized operations for CPU
7. **Custom Kernels**: Hand-optimized for specific hardware (IMI, AMX, etc.)

## **9. GGUF File Format**

**Location:** `ggml/src/gguf.cpp`

GGUF (GGML Universal Format) stores:
- Model architecture metadata
- Hyperparameters (n_layers, n_heads, etc.)
- Tokenizer vocabulary
- Tensor data (weights)

## **10. Test Infrastructure**

**test-backend-ops.cpp** is particularly important for development:

- Tests individual GGML operations
- Validates correctness across backends (CPU, IMI, CUDA, etc.)
- Performance benchmarking mode
- Region-of-Interest (ROI) markers for simulation

**Example Test:**
```bash
# Test Q8_0 matrix multiplication
test-backend-ops test -o MUL_MAT -b REF -p type_a=q8_0,type_b=f32,m=16,n=1,k=256

# Performance benchmark
test-backend-ops perf -o MUL_MAT -b CPU -p type_a=bf16,type_b=f32,m=4096,n=32,k=14336
```

## **11. Integration with iminn-tools**

The `iminnt` tool (`src/iminnt/llamacpp.py`) provides:

1. **Build Management**: Cross-compilation for RISC-V with IMI extensions
2. **Model Downloads**: Automatically fetches tiny models for testing
3. **Test Suite**: 50+ predefined tests covering all major operations
4. **Simulation Integration**: ROI markers enable Pilos performance modeling

**Workflow:**
```bash
# Build for RISC-V with IMI extensions
iminnt -t llama_imi build

# Run test with QEMU emulation
iminnt -t llama_imi run -d test_q4_0_stories

# Simulate with Pilos performance model
iminnt -t llama_imi sim -d test_q4_0_stories -o results/test1
```

## **12. Two-Phase Inference**

LLMs have two distinct phases:

1. **Prefill Phase** (First token):
   - Process entire prompt at once
   - Generate KV cache for all prompt tokens
   - Compute-bound (matrix-matrix ops)
   - Use GEMM kernels

2. **Decode Phase** (Subsequent tokens):
   - Generate one token at a time
   - Query against existing KV cache
   - Memory-bound (matrix-vector ops)
   - Use GEMV kernels

IMI optimizations target both phases with specialized `gemm_*` and `gemv_*` kernels.

---

## Summary

This architecture enables llama.cpp to run efficiently on diverse hardware while maintaining a clean, extensible codebase. The I-Machines integration adds RISC-V vector optimizations and custom quantization formats specifically tuned for their hardware extensions.

**Key Takeaways:**
- GGML provides the low-level tensor computation foundation
- Pluggable backend system supports multiple hardware targets
- IMI backend adds RISC-V optimizations with custom quantization
- Two-phase inference (prefill/decode) requires different kernel strategies
- Extensive test infrastructure validates correctness and measures performance
