# llama.cpp Execution Flow - Complete Code Trace

**Date:** December 9, 2025  
**Command:** `iminnt -t llama_imi run -d test_q4_0_stories`

This document traces the complete execution flow of llama.cpp from the command line entry point through model loading, context initialization, and inference execution.

---

## Table of Contents

1. [Command Execution Overview](#1-command-execution-overview)
2. [Entry Point: main()](#2-entry-point-main)
3. [Model Loading](#3-model-loading)
4. [Context Initialization](#4-context-initialization)
5. [Inference Loop](#5-inference-loop)
6. [Decode Execution](#6-decode-execution)
7. [Complete Call Stack](#7-complete-call-stack)

---

## 1. Command Execution Overview

### Actual Command Executed

```bash
qemu-riscv64 -E IMI_ROI_SIM="1" -cpu imicpu-v1 \
  /home/linhu/repo/iminn-tools/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
  --seed 42 -t 1 -ngl 0 -n 32 -no-cnv -st --no-warmup \
  --file /home/linhu/repo/iminn-tools/src/iminnt/resources/prompts/hello-world.txt
```

### Execution Environment

- **Binary**: RISC-V statically-linked `llama-cli`
- **Emulator**: QEMU user-mode with I-Machines CPU model (`imicpu-v1`)
- **Model**: `stories15M-q4_0.gguf` (4-bit quantized, 15M parameters)
- **Threads**: 1 (`-t 1`)
- **GPU Layers**: 0 (`-ngl 0` = CPU only)
- **Tokens to Generate**: 32 (`-n 32`)

---

## 2. Entry Point: main()

**File:** `tools/main/main.cpp:86`

### 2.1 Initialization

```cpp
int main(int argc, char ** argv) {
    common_params params;
    g_params = &params;
    
    // Parse command-line arguments
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }
    
    common_init();
    console::init(params.simple_io, params.use_color);
```

**Key Steps:**
1. Parse command-line arguments into `common_params`
2. Initialize common subsystems
3. Initialize console for output

### 2.2 Backend Initialization

```cpp
LOG_INF("%s: llama backend init\n", __func__);

llama_backend_init();      // Initialize GGML backends
llama_numa_init(params.numa);  // NUMA configuration
```

**Function:** `src/llama.cpp:70`

```cpp
void llama_backend_init(void) {
    ggml_time_init();
    
    // Initialize f16 tables
    struct ggml_init_params params = { 0, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    ggml_free(ctx);
}
```

### 2.3 Model and Context Loading

```cpp
// load the model and apply lora adapter, if any
LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
common_init_result llama_init = common_init_from_params(params);

model = llama_init.model.get();
ctx = llama_init.context.get();
```

**Function:** `common/common.cpp:952`

---

## 3. Model Loading

### 3.1 Entry Point

**Function:** `common/common.cpp:952`

```cpp
struct common_init_result common_init_from_params(common_params & params) {
    common_init_result iparams;
    auto mparams = common_model_params_to_llama(params);
    
    // Load model from GGUF file
    llama_model * model = llama_model_load_from_file(
        params.model.path.c_str(),  // "stories15M-q4_0.gguf"
        mparams);
```

### 3.2 Model Load Implementation

**Function:** `src/llama.cpp:155`

```cpp
static struct llama_model * llama_model_load_from_file_impl(
        const std::string & path_model,
        std::vector<std::string> & splits,
        struct llama_model_params params) {
    
    llama_model * model = new llama_model(params);
    
    // Load model from file
    int ret = llama_model_load(fname, splits, *model, params);
```

### 3.3 GGUF File Loading

**Function:** `src/llama.cpp:102`

```cpp
static int llama_model_load(
    const std::string & fname,
    std::vector<std::string> & splits,
    llama_model & model,
    llama_model_params & params) {
    
    // Create model loader
    llama_model_loader ml(
        fname,                    // "stories15M-q4_0.gguf"
        splits,
        params.use_mmap,          // Memory mapping
        params.check_tensors,
        params.kv_overrides,
        params.tensor_buft_overrides);
    
    // Load architecture
    model.load_arch(ml);
    
    // Load hyperparameters
    model.load_hparams(ml);
    
    // Load vocabulary
    model.load_vocab(ml);
    
    // Load tensors (weights)
    if (!model.load_tensors(ml)) {
        return -2;
    }
}
```

### 3.4 Model Loader Constructor

**File:** `src/llama-model-loader.cpp:471`

**Process:**
1. **Open GGUF file** - Read file header
2. **Parse metadata** - Extract key-value pairs (20 pairs for stories15M)
3. **Load tensor metadata** - Read tensor names, shapes, types (57 tensors)
4. **Memory mapping** - Map file regions to memory (if `use_mmap=true`)
5. **Build weight map** - Create mapping from tensor names to file offsets

**Key Data Structures:**
```cpp
struct llama_model_loader {
    gguf_context_ptr meta;              // GGUF metadata
    std::map<std::string, llama_tensor_weight> weights_map;  // Tensor weights
    llama_mmaps mappings;               // Memory mappings
    // ...
};
```

### 3.5 Tensor Loading

**Function:** `src/llama-model.cpp:load_tensors()`

**Process:**
1. **Create GGML tensors** - Allocate tensor metadata
2. **Allocate backend buffers** - Allocate memory in CPU_IMI backend
3. **Load tensor data:**
   - If memory-mapped: Create view into file
   - Otherwise: Read data from file and copy to buffer
4. **Repack weights** - Convert to I-Machines optimized format (if `GGML_CPU_IMI=ON`)

**Backend Buffer Allocation:**
```cpp
ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(
    get_buffer_type(tensor),  // CPU_IMI buffer type
    ggml_nbytes(tensor));
```

---

## 4. Context Initialization

### 4.1 Context Creation

**Function:** `common/common.cpp:967`

```cpp
auto cparams = common_context_params_to_llama(params);

llama_context * lctx = llama_init_from_model(model, cparams);
```

**Function:** `src/llama.cpp:llama_init_from_model()`

```cpp
llama_context * llama_init_from_model(
        struct llama_model * model,
        struct llama_context_params params) {
    
    return new llama_context(*model, params);
}
```

### 4.2 Context Constructor

**File:** `src/llama-context.cpp:19`

```cpp
llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    
    // Initialize context parameters
    cparams.n_ctx = params.n_ctx;
    cparams.n_batch = params.n_batch;
    cparams.n_threads = params.n_threads;
    cparams.n_threads_batch = params.n_threads_batch;
    
    // Initialize backends
    // GPU backends
    for (auto * dev : model.devices) {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        backends.emplace_back(backend);
    }
    
    // CPU backend (always added)
    backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    backends.emplace_back(backend_cpu);
    
    // Create backend scheduler
    sched.reset(ggml_backend_sched_new(
        backend_ptrs.data(),
        backend_buft.data(),
        backend_ptrs.size(),
        max_nodes,
        pipeline_parallel,
        cparams.op_offload));
    
    // Initialize memory (KV cache)
    memory.reset(model.create_memory(params_mem, cparams));
}
```

### 4.3 Backend Scheduler Setup

**Key Configuration:**
- **Backends**: CPU_IMI (primary), CPU (fallback)
- **Max Nodes**: Maximum graph nodes for worst-case
- **Pipeline Parallelism**: Disabled (single device)
- **Operation Offload**: Configured based on `op_offload` parameter

### 4.4 Thread Pool Setup

**File:** `tools/main/main.cpp:159-195`

```cpp
LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, 
        (int) params.cpuparams.n_threads);

auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
auto * reg = ggml_backend_dev_backend_reg(cpu_dev);

// Create thread pools
struct ggml_threadpool_params tpp_batch =
        ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
struct ggml_threadpool_params tpp =
        ggml_threadpool_params_from_cpu_params(params.cpuparams);

struct ggml_threadpool * threadpool_batch = NULL;
if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
    threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
    tpp.paused = true;
}

struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);

// Attach thread pools to context
llama_attach_threadpool(ctx, threadpool, threadpool_batch);
```

**For our command (`-t 1`):**
- `n_threads = 1` (decode thread pool)
- `n_threads_batch = 1` (prefill thread pool, if different)

---

## 5. Inference Loop

### 5.1 Main Loop Entry

**File:** `tools/main/main.cpp:573`

```cpp
while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
    // predict
    if (!embd.empty()) {
        // Process tokens in batches
        for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
            int n_eval = (int) embd.size() - i;
            if (n_eval > params.n_batch) {
                n_eval = params.n_batch;
            }
            
            // Decode batch
            if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                LOG_ERR("%s : failed to eval\n", __func__);
                return 1;
            }
            
            n_past += n_eval;
        }
    }
    
    // Sample next token
    const llama_token id = common_sampler_sample(smpl, ctx, -1);
    common_sampler_accept(smpl, id, /* accept_grammar= */ true);
    
    embd.push_back(id);
    --n_remain;
}
```

### 5.2 Batch Creation

**Function:** `include/llama.h:841`

```cpp
LLAMA_API struct llama_batch llama_batch_get_one(
        llama_token * tokens,
        int n_tokens,
        llama_pos pos,
        llama_seq_id seq_id,
        bool logits);
```

**Creates a batch with:**
- `n_tokens` tokens
- Single sequence (`seq_id = 0`)
- Positions starting from `n_past`
- `logits = true` for last token only

### 5.3 Decode Call

**Function:** `src/llama-context.cpp:2773`

```cpp
int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }
    return ret;
}
```

**Calls:** `llama_context::decode()`

---

## 6. Decode Execution

### 6.1 Context Decode Method

**File:** `src/llama-context.cpp:109`

```cpp
int llama_context::decode(const llama_batch & batch_inp) {
    // Initialize batch allocator
    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, 
                      cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, 
                      output_all)) {
        return -1;
    }
    
    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();
    
    // Initialize memory context
    llama_memory_context_ptr mctx;
    mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
    
    // Process unified batches
    do {
        const auto & ubatch = mctx->get_ubatch();
        
        // Process ubatch
        ggml_status status;
        const auto * res = process_ubatch(
            ubatch, 
            LLM_GRAPH_TYPE_DECODER, 
            mctx.get(), 
            status);
        
        if (!res) {
            return -2;
        }
        
    } while (mctx->next_ubatch());
    
    return 0;
}
```

### 6.2 Process Unified Batch

**File:** `src/llama-context.cpp:103`

```cpp
llm_graph_result * llama_context::process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret) {
    
    // Build computation graph
    auto * gf = graph_build(ubatch, gtype, mctx);
    
    // Schedule graph to backends
    ggml_backend_sched_reset(sched.get());
    if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
        ret = GGML_STATUS_ALLOC_FAILED;
        return nullptr;
    }
    
    // Execute graph
    ret = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (ret != GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    
    // Synchronize
    ggml_backend_sched_synchronize(sched.get());
    
    return gf_res_prev.get();
}
```

### 6.3 Graph Building

**File:** `src/llama-graph.cpp`

**Process:**
1. **Create input tensors:**
   - Token embeddings
   - Position embeddings
   - KV cache indices

2. **Build transformer layers:**
   - Attention (Q, K, V, O projections)
   - Feed-forward (gate, up, down)
   - Layer normalization
   - Residual connections

3. **Output layer:**
   - Output projection
   - Logits computation

### 6.4 Backend Scheduling

**Function:** `ggml/src/ggml-backend.cpp:ggml_backend_sched_graph_compute_async()`

**Process:**
1. **Analyze graph:**
   - Identify tensor dependencies
   - Determine operation requirements
   - Estimate execution costs

2. **Assign operations to backends:**
   - Prefer backend where tensors are stored
   - Check operation support
   - Consider memory constraints

3. **Allocate buffers:**
   - Allocate temporary buffers
   - Transfer tensors between backends if needed

4. **Execute operations:**
   - Dispatch to appropriate backend
   - Use thread pools for parallelization
   - Synchronize between operations

### 6.5 CPU_IMI Backend Execution

**For CPU_IMI backend:**
- Operations dispatched to `ggml_imi_*` kernels
- Uses RISC-V Vector (RVV) extensions
- VLEN=128 vector operations
- I-Machines custom instructions

**Key Kernels:**
- `ggml_imi_gemv_*` - Matrix-vector (decode phase)
- `ggml_imi_gemm_*` - Matrix-matrix (prefill phase)
- Quantized operations (Q4_0, Q8_0, etc.)

---

## 7. Complete Call Stack

### 7.1 Model Loading Call Stack

```
main() [tools/main/main.cpp:86]
  └─> common_init_from_params() [common/common.cpp:952]
      └─> llama_model_load_from_file() [src/llama.cpp:155]
          └─> llama_model_load() [src/llama.cpp:102]
              └─> llama_model_loader() [src/llama-model-loader.cpp:471]
                  ├─> gguf_init_from_file() [ggml/src/gguf.cpp]
                  ├─> model.load_arch() [src/llama-model.cpp]
                  ├─> model.load_hparams() [src/llama-model.cpp]
                  ├─> model.load_vocab() [src/llama-model.cpp]
                  └─> model.load_tensors() [src/llama-model.cpp]
                      └─> ggml_backend_buft_alloc_buffer() [ggml/src/ggml-backend.cpp]
```

### 7.2 Context Initialization Call Stack

```
main() [tools/main/main.cpp:140]
  └─> common_init_from_params() [common/common.cpp:967]
      └─> llama_init_from_model() [src/llama.cpp]
          └─> llama_context::llama_context() [src/llama-context.cpp:19]
              ├─> ggml_backend_init_by_type() [ggml/src/ggml-backend.cpp]
              ├─> ggml_backend_sched_new() [ggml/src/ggml-backend.cpp]
              └─> model.create_memory() [src/llama-memory.cpp]
```

### 7.3 Inference Call Stack

```
main() [tools/main/main.cpp:679]
  └─> llama_decode() [src/llama-context.cpp:2773]
      └─> llama_context::decode() [src/llama-context.cpp:109]
          ├─> llama_batch_allocr::init() [src/llama-batch.cpp:25]
          ├─> memory->init_batch() [src/llama-memory-hybrid.cpp]
          └─> llama_context::process_ubatch() [src/llama-context.cpp:103]
              ├─> graph_build() [src/llama-graph.cpp]
              ├─> ggml_backend_sched_alloc_graph() [ggml/src/ggml-backend.cpp]
              ├─> ggml_backend_sched_graph_compute_async() [ggml/src/ggml-backend.cpp]
              │   └─> ggml_graph_compute_thread() [ggml/src/ggml-threading.cpp]
              │       └─> ggml_compute_forward() [ggml/src/ggml.c]
              │           └─> ggml_imi_gemv_* / ggml_imi_gemm_* [ggml/src/ggml-cpu/ggml-cpu.c]
              └─> ggml_backend_sched_synchronize() [ggml/src/ggml-backend.cpp]
```

### 7.4 Sampling Call Stack

```
main() [tools/main/main.cpp:710]
  └─> common_sampler_sample() [common/sampling.cpp]
      └─> llama_get_logits_ith() [src/llama-context.cpp:63]
          └─> llama_context::get_logits_ith()
              └─> Access logits from graph output buffer
```

---

## 8. Key Data Flow

### 8.1 Token Flow

```
Command Line Arguments
    ↓
Tokenization (common_tokenize)
    ↓
embd_inp (vector<llama_token>)
    ↓
llama_batch_get_one()
    ↓
llama_batch structure
    ↓
llama_batch_allocr::init()
    ↓
llama_ubatch (unified batch)
    ↓
Graph Input Tensors
    ↓
Computation Graph
    ↓
Backend Execution (CPU_IMI)
    ↓
Output Logits
    ↓
Sampling (common_sampler_sample)
    ↓
Next Token
```

### 8.2 Model Weight Flow

```
GGUF File (stories15M-q4_0.gguf)
    ↓
llama_model_loader
    ↓
Memory-mapped file regions
    ↓
Backend Buffer Allocation (CPU_IMI)
    ↓
Tensor Views / Copies
    ↓
Weight Repacking (IMI format)
    ↓
Model Tensors in Backend Memory
    ↓
Graph Operations (GEMV/GEMM)
```

### 8.3 Computation Flow

```
Input Tokens
    ↓
Embedding Lookup
    ↓
Transformer Block 0
    ├─> Attention (Q, K, V, O)
    ├─> Feed-Forward
    └─> Layer Norm
    ↓
Transformer Block 1
    └─> ...
    ↓
Transformer Block N-1
    └─> ...
    ↓
Output Projection
    ↓
Logits
```

---

## 9. Memory Management

### 9.1 Model Memory

- **Location**: Backend buffers (CPU_IMI)
- **Type**: Memory-mapped or copied
- **Format**: Quantized (Q4_0) with I-Machines repacking

### 9.2 Context Memory (KV Cache)

- **Location**: `llama_memory_hybrid` or `llama_memory_recurrent`
- **Structure**: Per-sequence KV cache
- **Management**: Automatic allocation and eviction

### 9.3 Compute Memory

- **Location**: Temporary buffers in backends
- **Lifetime**: Per-graph execution
- **Management**: Backend scheduler allocates/deallocates

---

## 10. Performance Characteristics

### 10.1 For Our Command

- **Model Size**: 17.50 MiB (Q4_0, 6.01 bits per weight)
- **Context Size**: Default (typically 512 or 2048)
- **Batch Size**: 1 token at a time (decode phase)
- **Threads**: 1 (single-threaded)
- **Backend**: CPU_IMI only (no GPU)

### 10.2 Execution Phases

1. **Prefill** (if prompt provided):
   - Process entire prompt in parallel
   - Uses `n_threads_batch` threads
   - Matrix-matrix operations (GEMM)

2. **Decode** (autoregressive):
   - Process one token at a time
   - Uses `n_threads` threads
   - Matrix-vector operations (GEMV)

### 10.3 QEMU Overhead

- **Instruction Translation**: RISC-V → x86
- **System Call Emulation**: RISC-V syscalls → host syscalls
- **Memory Mapping**: Guest memory → host memory
- **CPU Model**: I-Machines extensions emulation

---

## 11. Key Files Reference

| Component | File | Purpose |
|-----------|------|---------|
| **Entry Point** | `tools/main/main.cpp` | Main CLI application |
| **Model Loading** | `src/llama.cpp:102` | Model load implementation |
| **GGUF Loading** | `src/llama-model-loader.cpp` | GGUF file parsing |
| **Context Init** | `src/llama-context.cpp:19` | Context constructor |
| **Decode** | `src/llama-context.cpp:109` | Decode implementation |
| **Graph Building** | `src/llama-graph.cpp` | Computation graph construction |
| **Backend Scheduler** | `ggml/src/ggml-backend.cpp` | Operation scheduling |
| **Batch Management** | `src/llama-batch.cpp` | Batch allocation and splitting |
| **Threading** | `ggml/src/ggml-threading.cpp` | Thread pool execution |

---

## 12. Debugging Tips

### 12.1 Enable Debug Logging

```bash
# Set environment variables
export LLAMA_BATCH_DEBUG=2
export LLAMA_GRAPH_INPUT_DEBUG=1
export LLAMA_GRAPH_REUSE_DISABLE=1
```

### 12.2 Key Log Messages

From execution logs (`llama_imi_1t`):
```
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_loader: loaded meta data with 20 key-value pairs and 57 tensors
llama_model_loader: - type  f32:   13 tensors
llama_model_loader: - type q4_0:   43 tensors
llama_model_loader: - type q8_0:    1 tensors
```

### 12.3 Performance Profiling

```cpp
// In llama-context.cpp
llama_perf_context_data perf = ctx->perf_get_data();
// Contains: t_load_ms, t_p_eval_ms, n_p_eval, t_eval_ms, n_eval
```

---

## Summary

The execution flow follows this path:

1. **QEMU** emulates RISC-V environment
2. **llama-cli** main() parses arguments and initializes
3. **Model Loading** reads GGUF file, loads weights to CPU_IMI backend
4. **Context Creation** sets up backends, scheduler, and KV cache
5. **Inference Loop** processes tokens one at a time
6. **Decode** builds graph, schedules to CPU_IMI, executes
7. **Sampling** selects next token from logits
8. **Repeat** until `n_remain == 0` or EOS token

All computation happens in the **CPU_IMI backend** using RISC-V Vector extensions and I-Machines optimized kernels, executed through QEMU emulation.

