# llama.cpp: From Model Loading to CPU Backend Inference

**Date:** January 2025  
**Purpose:** Complete trace of llama.cpp execution from GGUF model file to CPU backend kernel execution

**Focus:** CPU-only execution on RISC-V architecture (via QEMU emulation). This document focuses exclusively on the CPU backend path, as GPU backends are not available in this project.

This document provides a detailed, code-level walkthrough of how llama.cpp loads a GGUF model and executes inference on the **CPU backend only**, with specific focus on understanding the data flow and execution path for RISC-V CPU execution.

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 0: Entry Point and Command-Line Configuration](#phase-0-entry-point-and-command-line-configuration)
3. [Phase 1: Model Loading](#phase-1-model-loading)
4. [Phase 2: Context Initialization](#phase-2-context-initialization)
5. [Phase 3: Graph Construction](#phase-3-graph-construction)
6. [Phase 4: Backend Scheduling](#phase-4-backend-scheduling)
7. [Phase 5: CPU Backend Execution](#phase-5-cpu-backend-execution)
   - [Phase 5.5: Multi-Threading Workload Distribution (CPU-Only)](#phase-55-multi-threading-workload-distribution-cpu-only)
   - [Phase 5.5.8: Response: How is Work Distributed Across Cores?](#phase-558-response-how-is-work-distributed-across-cores)
8. [Phase 6: Kernel Dispatch](#phase-6-kernel-dispatch)
9. [Phase 7: Major Model Inference Computations and Data Structures](#phase-7-major-model-inference-computations-and-data-structures)
10. [Complete Data Flow Diagram](#complete-data-flow-diagram)
11. [Key Code Locations](#key-code-locations)

---

## Overview

### High-Level Flow (CPU-Only Execution)

```
GGUF File (stories15M-q4_0.gguf)
    ↓
[1] Model Loading
    ├─> Parse GGUF metadata
    ├─> Load tensor weights
    └─> Allocate in CPU backend buffers (RISC-V CPU)
    ↓
[2] Context Initialization
    ├─> Create llama_context
    ├─> Register CPU backend only (no GPU)
    ├─> Create backend scheduler (CPU-only)
    └─> Initialize KV cache in CPU memory
    ↓
[3] Inference Request
    ├─> Tokenize input text
    └─> Create llama_batch
    ↓
[4] Graph Construction
    ├─> Build computation graph
    ├─> Create tensor operations
    └─> Set input tensors
    ↓
[5] Backend Scheduling (CPU-Only)
    ├─> All operations assigned to CPU backend
    ├─> Allocate output tensors in CPU buffers
    └─> Plan execution order (single backend)
    ↓
[6] CPU Backend Execution
    ├─> Dispatch to thread pool (2 threads for -t 2)
    ├─> Execute operations sequentially
    └─> Synchronize threads
    ↓
[7] CPU Kernel Execution (RISC-V)
    ├─> Select CPU kernel (IMI-optimized or standard)
    ├─> Parallelize work across threads
    └─> Compute results on RISC-V CPU
    ↓
Output Logits → Sampling → Next Token
```

**Key Points:**
- **CPU-only execution**: No GPU backends, all operations run on CPU
- **RISC-V architecture**: Executed via QEMU emulation with I-Machines CPU extensions
- **Single backend**: Only CPU backend is registered and used
- **No cross-backend transfers**: All tensors stay in CPU memory

---

## Phase 0: Entry Point and Command-Line Configuration

### 0.1 Entry Point: llama-cli

**Yes, it starts from llama-cli!**

**File:** [`tools/main/main.cpp:86`](../dev_env/llama.cpp/tools/main/main.cpp#L86)

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
    
    // ... validation and initialization ...
    
    // Backend initialization
    llama_backend_init();
    llama_numa_init(params.numa);
    
    // Load model and create context
    common_init_result llama_init = common_init_from_params(params);
    
    model = llama_init.model.get();
    ctx = llama_init.context.get();  // <-- Context is fetched here!
    
    // ... rest of initialization ...
}
```

**Key Points:**
- Entry point is `main()` in `tools/main/main.cpp`
- Binary is `llama-cli` (built from `tools/main/main.cpp`)
- Command-line arguments are parsed into `common_params` structure
- Context is created via `common_init_from_params()` and stored in `ctx`

### 0.1.1 Information Registered During Initialization

**What gets registered/initialized:**

1. **Backend Registration** (`llama_backend_init()`)
   - **File:** [`src/llama.cpp:70`](../dev_env/llama.cpp/src/llama.cpp#L70)
   - Initializes GGML backend registry
   - Registers available backends (CPU, REF, GPU if available)
   - For CPU-only: Only CPU and REF backends registered

2. **NUMA Configuration** (`llama_numa_init()`)
   - **File:** [`src/llama.cpp:81`](../dev_env/llama.cpp/src/llama.cpp#L81)
   - Configures NUMA (Non-Uniform Memory Access) if enabled
   - For CPU-only: Typically disabled or not applicable

3. **CPU Core Detection** (for thread defaults)
   - **File:** [`common/arg.cpp:791`](../dev_env/llama.cpp/common/arg.cpp#L791)
   - Uses `std::thread::hardware_concurrency()` to detect CPU cores
   - **Only used if `-t` is not specified or set to 0 or negative**
   - If you specify `-t 2`, CPU core count is **not used** (your value takes precedence)

**CPU Core Detection Logic:**

```cpp
// In common/arg.cpp:786-794
add_opt(common_arg(
    {"-t", "--threads"}, "N",
    "number of CPU threads to use during generation (default: auto-detect)",
    [](common_params & params, int value) {
        params.cpuparams.n_threads = value;
        if (params.cpuparams.n_threads <= 0) {
            // Only if value <= 0, use hardware_concurrency()
            params.cpuparams.n_threads = std::thread::hardware_concurrency();
        }
    }
));
```

**For your command (`-t 2`):**
- CPU core count is **detected but NOT used** because you explicitly set `-t 2`
- `std::thread::hardware_concurrency()` may detect host CPU cores (e.g., 8, 16, etc.)
- But since you specified `-t 2`, llama.cpp uses **2 threads** regardless of core count
- CPU core information is **not passed to backend** - only thread count is used

**What Information is Actually Used:**
- **Thread count** (from `-t`): Used to create thread pools
- **Backend type** (CPU): Registered and used for all operations
- **Buffer types** (CPU_IMI or standard CPU): Determines which kernels to use
- **CPU core count**: Only used as default if `-t` not specified

### 0.2 Command-Line Options

**Yes, users can configure many options!**

Users can configure options through command-line arguments. Here are the key categories:

#### Model Options

```bash
-m, --model <path>              # Path to GGUF model file (required)
--mmproj <path>                 # Path to multimodal projector file
--lora <path>                   # LoRA adapter file(s)
--lora-scale <scale>            # LoRA scale factor
```

#### Context Options

**Detailed explanations of context-related options:**

##### 1. `-c, --ctx-size <n>`: Context Size

**Purpose:** Sets the maximum number of tokens that can be processed in a single sequence (context window).

**File:** [`common/arg.cpp:908`](../dev_env/llama.cpp/common/arg.cpp#L908)

**Default Behavior:**
- **Default:** 512 (if not specified)
- **If set to 0:** Uses the model's training context size (loaded from GGUF metadata)
- **If specified:** Uses the provided value

**What It Controls:**

1. **KV Cache Allocation:**
   ```cpp
   // KV cache size is determined by n_ctx
   KV cache per layer = [n_kv, n_ctx, head_dim]
   Total KV cache = n_layers * 2 * n_kv * n_ctx * head_dim * sizeof(f16)
   ```

2. **Memory Usage:**
   - **Example:** 6 layers, 2 KV heads, 48 head_dim, F16
   - `n_ctx = 512`: KV cache = 6 × 2 × 512 × 2 × 48 × 2 bytes = **~3.4 MiB**
   - `n_ctx = 4096`: KV cache = 6 × 2 × 4096 × 2 × 48 × 2 bytes = **~27 MiB**
   - **8× increase in context size = 8× increase in KV cache memory**

3. **Maximum Sequence Length:**
   - Model can process sequences up to `n_ctx` tokens
   - If sequence exceeds `n_ctx`, oldest tokens are evicted (or shifted if `--ctx-shift` enabled)

**Code Location:**
- **Parsing:** [`common/arg.cpp:908-913`](../dev_env/llama.cpp/common/arg.cpp#L908)
- **Usage:** [`src/llama-context.cpp:50`](../dev_env/llama.cpp/src/llama-context.cpp#L50)
  ```cpp
  cparams.n_ctx = params.n_ctx == 0 ? hparams.n_ctx_train : params.n_ctx;
  ```

**Example:**
```bash
# Use default (512)
llama-cli -m model.gguf ...

# Use model's training context size
llama-cli -m model.gguf -c 0 ...

# Use custom context size (4096)
llama-cli -m model.gguf -c 4096 ...
```

**Trade-offs:**
- **Larger `n_ctx`:** More context, but higher memory usage
- **Smaller `n_ctx`:** Less memory, but limited context window

##### 2. `--rope-freq-base <freq>`: RoPE Frequency Base

**Purpose:** Sets the base frequency for Rotary Position Embedding (RoPE). Controls how position information is encoded in Q and K vectors.

**File:** [`common/arg.cpp:1543`](../dev_env/llama.cpp/common/arg.cpp#L1543)

**Default Behavior:**
- **Default:** Loaded from model's GGUF metadata (typically 10000.0)
- **If not specified:** Uses `hparams.rope_freq_base_train` from model

**What It Does:**

**RoPE (Rotary Position Embedding)** rotates Q and K vectors based on position:
- **Formula:** `freq_i = freq_base / (10000^(2i/d))` where `i` is dimension index, `d` is head dimension
- **Higher `freq_base`:** Faster rotation (more position sensitivity)
- **Lower `freq_base`:** Slower rotation (less position sensitivity)

**Code Location:**
- **Parsing:** [`common/arg.cpp:1543-1548`](../dev_env/llama.cpp/common/arg.cpp#L1543)
- **Usage:** [`src/llama-context.cpp:51`](../dev_env/llama.cpp/src/llama-context.cpp#L51)
  ```cpp
  cparams.rope_freq_base = params.rope_freq_base == 0.0f ? 
      hparams.rope_freq_base_train : params.rope_freq_base;
  ```
- **Application:** [`src/llama-graph.cpp`](../dev_env/llama.cpp/src/llama-graph.cpp) (during graph building)
  ```cpp
  q = ggml_rope(ctx0, q, ubatch.pos, n_ctx, freq_base, freq_scale);
  k = ggml_rope(ctx0, k, ubatch.pos, n_ctx, freq_base, freq_scale);
  ```

**Example:**
```bash
# Use default (from model, typically 10000.0)
llama-cli -m model.gguf ...

# Use custom frequency base
llama-cli -m model.gguf --rope-freq-base 5000.0 ...
```

**When to Change:**
- **Context extension:** When using longer contexts than training (often combined with `--rope-freq-scale`)
- **Fine-tuning:** When fine-tuning models for different context lengths

##### 3. `--rope-freq-scale <scale>`: RoPE Frequency Scale

**Purpose:** Scales RoPE frequencies to extend context length beyond training. Allows models trained on shorter contexts to work with longer contexts.

**File:** [`common/arg.cpp:1550`](../dev_env/llama.cpp/common/arg.cpp#L1550)

**Default Behavior:**
- **Default:** 1.0 (no scaling, use training frequencies)
- **If not specified:** Uses `hparams.rope_freq_scale_train` from model (typically 1.0)

**What It Does:**

**Frequency Scaling Formula:**
- **Effective frequency:** `freq_effective = freq_base / freq_scale`
- **Context extension factor:** `1 / freq_scale`
- **Example:** `freq_scale = 0.5` → 2× context extension

**How It Works:**
```
Original RoPE (freq_scale = 1.0):
  Position 0: Rotation angle = 0°
  Position 100: Rotation angle = 100 × base_freq
  Position 200: Rotation angle = 200 × base_freq

Scaled RoPE (freq_scale = 0.5):
  Position 0: Rotation angle = 0°
  Position 100: Rotation angle = 100 × base_freq / 0.5 = 200 × base_freq
  Position 200: Rotation angle = 200 × base_freq / 0.5 = 400 × base_freq
  
Result: Same rotation angles at 2× positions → 2× context extension
```

**Code Location:**
- **Parsing:** [`common/arg.cpp:1550-1555`](../dev_env/llama.cpp/common/arg.cpp#L1550)
- **Usage:** [`src/llama-context.cpp:52`](../dev_env/llama.cpp/src/llama-context.cpp#L52)
  ```cpp
  cparams.rope_freq_scale = params.rope_freq_scale == 0.0f ? 
      hparams.rope_freq_scale_train : params.rope_freq_scale;
  ```

**Example:**
```bash
# No scaling (use training frequencies)
llama-cli -m model.gguf ...

# Extend context by 2× (freq_scale = 0.5)
llama-cli -m model.gguf --rope-freq-scale 0.5 -c 2048 ...

# Extend context by 4× (freq_scale = 0.25)
llama-cli -m model.gguf --rope-freq-scale 0.25 -c 4096 ...
```

**Trade-offs:**
- **Lower `freq_scale`:** Longer context, but may reduce position accuracy
- **Higher `freq_scale`:** Better position accuracy, but shorter effective context

**Common Use Case:**
- Model trained on 2048 tokens, want to use 4096 tokens:
  ```bash
  --rope-freq-scale 0.5 -c 4096  # 2× extension
  ```

##### 4. `--ctx-shift` / `--no-context-shift`: KV Cache Shifting

**Purpose:** Enables or disables KV cache shifting when context exceeds `n_ctx`. Allows processing sequences longer than the context window by shifting/evicting old tokens.

**File:** [`common/arg.cpp:980-991`](../dev_env/llama.cpp/common/arg.cpp#L980)

**Default Behavior:**
- **Default:** **Enabled** (`ctx_shift = true`) for infinite text generation
- **Can be disabled:** Using `--no-context-shift` flag

**What KV Cache Shifting Does:**

**Without Context Shifting (Disabled):**
```
Context window: n_ctx = 512 tokens

Token sequence: [0, 1, 2, ..., 510, 511, 512, 513, ...]
                              ↑
                         Context full!

When token 512 arrives:
├─> Context is full (512 tokens)
├─> Oldest tokens (0, 1, 2, ...) are EVICTED (removed)
└─> New tokens appended, but oldest are lost
```

**With Context Shifting (Enabled):**
```
Context window: n_ctx = 512 tokens

Token sequence: [0, 1, 2, ..., 510, 511, 512, 513, ...]
                              ↑
                         Context full!

When token 512 arrives:
├─> Context is full (512 tokens)
├─> KV cache is SHIFTED: Oldest tokens removed, new tokens appended
├─> Maintains sliding window of last n_ctx tokens
└─> Allows processing sequences longer than n_ctx
```

**How It Works:**

**KV Cache Shifting Process:**
1. **When context exceeds `n_ctx`:**
   - Oldest tokens are removed from KV cache
   - Remaining tokens are shifted to fill the gap
   - New tokens are appended to the end

2. **Position Remapping:**
   - Token positions are remapped to fit within `[0, n_ctx)`
   - Attention masks are updated accordingly

**Code Location:**
- **Parsing:** [`common/arg.cpp:980-991`](../dev_env/llama.cpp/common/arg.cpp#L980)
- **Usage:** Memory management modules (`src/llama-memory-*.cpp`)

**Example:**
```bash
# Context shifting enabled (default)
llama-cli -m model.gguf -c 512 -n -1 ...  # Infinite generation

# Context shifting disabled
llama-cli -m model.gguf -c 512 --no-context-shift -n -1 ...
```

**When to Use:**

**Enable Context Shifting (`--ctx-shift`):**
- **Infinite text generation** (`-n -1`)
- **Long conversations** that exceed context window
- **Streaming applications** with unbounded input

**Disable Context Shifting (`--no-context-shift`):**
- **Fixed-length sequences** (don't exceed `n_ctx`)
- **Reproducible results** (no position remapping)
- **Debugging** (easier to track positions)

**Memory Impact:**
- **No additional memory:** Shifting reuses existing KV cache slots
- **Memory stays constant:** Always `n_ctx` tokens, regardless of sequence length

**Summary Table:**

| Option | Purpose | Default | Impact |
|--------|---------|---------|--------|
| **`-c, --ctx-size`** | Maximum context length | 512 (or model default) | Controls KV cache size and memory usage |
| **`--rope-freq-base`** | RoPE base frequency | From model (typically 10000.0) | Controls position encoding sensitivity |
| **`--rope-freq-scale`** | RoPE frequency scaling | 1.0 (no scaling) | Extends context beyond training length |
| **`--ctx-shift`** | Enable KV cache shifting | Enabled | Allows sequences longer than `n_ctx` |

#### Threading Options

```bash
-t, --threads <n>               # Number of threads for decode (default: auto-detect)
--threads-batch <n>             # Number of threads for prefill (default: same as --threads)
--threads-pool <n>              # Thread pool size
```

**Important:** When you specify `-t 2`:
- **Decode phase** (autoregressive token generation): Uses **2 threads**
- **Prefill phase** (prompt processing): Also uses **2 threads** (defaults to same as `-t`)

If you want different thread counts:
```bash
-t 2 --threads-batch 4    # 2 threads for decode, 4 threads for prefill
```

**File:** [`common/arg.cpp:796-797`](../dev_env/llama.cpp/common/arg.cpp#L796)
The help text confirms: `--threads-batch` default is "same as --threads"

#### GPU/Backend Options (Not Used in This Project)

```bash
-ngl, --n-gpu-layers <n>        # Number of layers to offload to GPU (0 = CPU only)
--gpu-layers-draft <n>          # GPU layers for draft model
--main-gpu <id>                 # Main GPU device ID
--tensor-split <split>          # Tensor split across GPUs
```

**For CPU-only execution (this project):**
```bash
-ngl 0  # Always set to 0 (CPU only, no GPU available)
```

**Note:** In this project, GPU backends are not available. All execution happens on the RISC-V CPU backend via QEMU emulation.

#### Generation Options

```bash
-n, --n-predict <n>             # Number of tokens to generate (default: -1 = infinite)
-p, --prompt <text>             # Prompt text
-f, --file <path>               # Prompt file
--seed <n>                      # Random seed
--temp <temp>                   # Temperature (default: 0.8)
--top-k <k>                     # Top-k sampling (default: 40)
--top-p <p>                     # Top-p (nucleus) sampling (default: 0.95)
```

#### Sampling Options

```bash
--repeat-penalty <penalty>      # Repeat penalty (default: 1.0)
--repeat-last-n <n>             # Last N tokens to penalize (default: 64)
--frequency-penalty <penalty>   # Frequency penalty
--presence-penalty <penalty>    # Presence penalty
--mirostat <n>                  # Mirostat sampling (0=off, 1=v1, 2=v2)
--mirostat-tau <tau>            # Mirostat target entropy
```

#### Complete Example Command

From your test configuration:
```bash
llama-cli \
  -m stories15M-q4_0.gguf \     # Model file
  --seed 42 \                   # Random seed
  -t 1 \                        # 1 thread
  -ngl 0 \                      # No GPU layers (CPU only)
  -n 32 \                       # Generate 32 tokens
  -no-cnv \                     # No conversation mode
  -st \                         # Special tokens enabled
  --no-warmup \                 # Skip warmup
  --file hello-world.txt        # Input prompt file
```

### 0.3 Context Initialization Flow

**When is Context Created?**

**File:** [`common/common.cpp:952`](../dev_env/llama.cpp/common/common.cpp#L952)

The context is created in `common_init_from_params()`:

```cpp
struct common_init_result common_init_from_params(common_params & params) {
    common_init_result iparams;
    
    // Step 1: Convert model parameters
    auto mparams = common_model_params_to_llama(params);
    
    // Step 2: Load model from GGUF file
    llama_model * model = llama_model_load_from_file(
        params.model.path.c_str(),  // e.g., "stories15M-q4_0.gguf"
        mparams);
    
    if (model == NULL) {
        LOG_ERR("failed to load model\n");
        return iparams;
    }
    
    // Step 3: Convert context parameters from command-line args
    auto cparams = common_context_params_to_llama(params);
    
    // Step 4: Create context from model
    llama_context * lctx = llama_init_from_model(model, cparams);
    
    if (lctx == NULL) {
        LOG_ERR("failed to create context\n");
        llama_model_free(model);
        return iparams;
    }
    
    // Step 5: Apply optional adapters (LoRA, control vectors, etc.)
    // ... adapter loading ...
    
    return {model, lctx};  // Return both model and context
}
```

**Timeline:**
1. **Command-line parsing** → `common_params` structure
2. **Model loading** → `llama_model` structure (weights, vocabulary, architecture)
3. **Context creation** → `llama_context` structure (runtime state, backends, KV cache)
4. **Context is fetched** → Stored in `ctx` variable in `main()` at line 143

### 0.4 What is llama_context?

**Context Structure**

**File:** [`src/llama-context.cpp:19`](../dev_env/llama.cpp/src/llama-context.cpp#L19)

The `llama_context` is the **runtime state** for inference. It contains:

```cpp
struct llama_context {
    // 1. Model reference
    const llama_model & model;          // Reference to loaded model (weights, vocab, etc.)
    
    // 2. Context parameters (from command-line)
    llama_cparams cparams;               // Context size, batch size, threads, etc.
    
    // 3. Backend infrastructure
    ggml_backend_sched_ptr sched;       // Backend scheduler (assigns ops to CPU/GPU)
    std::vector<ggml_backend_t> backends; // Available backends (CPU, GPU, etc.)
    ggml_backend_t backend_cpu;          // CPU backend
    
    // 4. Memory management
    llama_memory_ptr memory;             // KV cache for attention keys/values
    llama_batch_allocr_ptr balloc;       // Batch allocator (splits batches)
    
    // 5. Computation graphs
    llm_graph_result_ptr gf_res_prev;    // Previous graph result (for graph reuse)
    
    // 6. Thread pools (attached later in main())
    // These are attached via llama_attach_threadpool()
    
    // 7. Performance tracking
    llama_perf_context perf;            // Performance metrics
};
```

**What Context Contains at Creation Time**

When `llama_init_from_model()` is called, the context is **partially initialized**:

#### ✅ Already Initialized:
1. **Model reference** - Points to loaded model
2. **Context parameters** - From command-line (`n_ctx`, `n_batch`, `n_threads`, etc.)
3. **Backends** - CPU backend registered, GPU backends (if any) registered
4. **Backend scheduler** - Created and ready to assign operations
5. **KV cache memory** - Allocated (but empty, no tokens cached yet)
6. **Batch allocator** - Created and ready to process batches

#### ❌ Not Yet Initialized:
1. **Thread pools** - Created later in `main()` (line 159-195)
2. **Computation graphs** - Built during inference
3. **KV cache data** - Populated during inference

**Context Parameters (from command-line)**

**File:** [`common/common.cpp:common_context_params_to_llama()`](../dev_env/llama.cpp/common/common.cpp)

The context parameters are converted from `common_params`:

```cpp
llama_context_params common_context_params_to_llama(common_params & params) {
    llama_context_params cparams;
    
    // Context size
    cparams.n_ctx = params.n_ctx;              // e.g., 2048 (from -c 2048)
    
    // Batch sizes
    cparams.n_batch = params.n_batch;          // e.g., 512 (from --batch-size)
    cparams.n_ubatch = params.n_ubatch;        // Unified batch size
    
    // Threading
    cparams.n_threads = params.cpuparams.n_threads;        // Decode threads (from -t)
    cparams.n_threads_batch = params.cpuparams_batch.n_threads; // Prefill threads
    
    // GPU offloading
    cparams.n_gpu_layers = params.n_gpu_layers; // GPU layers (from -ngl)
    
    // Memory
    cparams.kv_unified = params.kv_unified;     // Unified KV cache
    
    // Operation offloading
    cparams.op_offload = params.op_offload;     // KQV offload to GPU
    
    // ... more parameters ...
    
    return cparams;
}
```

**Key Parameters:**
- `n_ctx`: Maximum context length (how many tokens can be processed)
- `n_batch`: Batch size for processing multiple tokens
- `n_threads`: Number of threads for decode phase (from `-t`)
- `n_threads_batch`: Number of threads for prefill phase (from `--threads-batch`, defaults to same as `-t`)
- `n_gpu_layers`: How many layers to offload to GPU (0 = CPU only)

**For your command (`-t 2`):**
- `n_threads = 2` → **2 threads for decode phase** (autoregressive token generation)
- `n_threads_batch = 2` → **2 threads for prefill phase** (prompt processing, defaults to same as `-t`)

**Note:** Since both are 2, llama.cpp creates a **single thread pool** shared between prefill and decode phases. If they were different, two separate thread pools would be created.

**Thread Pool Creation Logic:**

**File:** [`tools/main/main.cpp:177-195`](../dev_env/llama.cpp/tools/main/main.cpp#L177)

```cpp
struct ggml_threadpool * threadpool_batch = NULL;
if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
    // Only create separate batch threadpool if parameters differ
    threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
    tpp.paused = true;  // Pause decode threadpool when batch is active
}

struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
llama_attach_threadpool(ctx, threadpool, threadpool_batch);
```

**For `-t 2` (no `--threads-batch` specified):**
- `tpp.n_threads = 2` (from `-t 2`)
- `tpp_batch.n_threads = 2` (defaults to same as `-t`)
- Since they match → **Single thread pool** with 2 threads
- This thread pool is used for **both prefill and decode phases**

### 0.6 Information Registered at Initialization Stage

**What Information is Registered:**

#### 1. Backend Registration

**File:** [`src/llama.cpp:70`](../dev_env/llama.cpp/src/llama.cpp#L70) (`llama_backend_init()`)

```cpp
void llama_backend_init(void) {
    ggml_time_init();
    
    // Initialize f16 tables (needed for half-precision operations)
    struct ggml_init_params params = { 0, NULL, false };
    struct ggml_context * ctx = ggml_init(params);
    ggml_free(ctx);
}
```

**What this does:**
- Initializes GGML backend registry
- Registers available backends (CPU, REF, GPU if compiled in)
- For CPU-only builds: Only CPU and REF backends available
- **No CPU core information** is registered here

#### 2. NUMA Configuration

**File:** [`src/llama.cpp:81`](../dev_env/llama.cpp/src/llama.cpp#L81) (`llama_numa_init()`)

**What is NUMA?**

**NUMA (Non-Uniform Memory Access)** is a computer memory architecture used in multi-processor systems where:
- **Multiple CPU sockets** (each with its own local memory)
- **Memory access time** depends on the memory location relative to the processor
- **Local memory** (same socket) is faster than **remote memory** (different socket)

**NUMA System Example:**
```
┌─────────────────────────────────────────────────────────┐
│ NUMA Node 0 (Socket 0)                                  │
│ ├─> CPU Cores: 0, 1, 2, 3                              │
│ └─> Local Memory: 32 GB (fast access)                  │
├─────────────────────────────────────────────────────────┤
│ NUMA Node 1 (Socket 1)                                  │
│ ├─> CPU Cores: 4, 5, 6, 7                              │
│ └─> Local Memory: 32 GB (fast access)                    │
└─────────────────────────────────────────────────────────┘

Memory Access:
- Core 0 accessing Node 0 memory: ~100 ns (local, fast)
- Core 0 accessing Node 1 memory: ~200 ns (remote, slower)
```

**Why NUMA Matters for llama.cpp:**

1. **Thread Affinity:** Pin threads to cores on the same NUMA node
2. **Memory Allocation:** Allocate memory from the local NUMA node
3. **Performance:** Reduce remote memory access latency

**NUMA Configuration in llama.cpp:**

```cpp
void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) 
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);  // Configure NUMA topology
        }
    }
}
```

**NUMA Strategies:**

The `ggml_numa_strategy` enum typically includes:
- `GGML_NUMA_STRATEGY_DISABLED`: NUMA awareness disabled (default for QEMU)
- `GGML_NUMA_STRATEGY_DISTRIBUTE`: Distribute threads across NUMA nodes
- `GGML_NUMA_STRATEGY_ISOLATE`: Isolate threads to specific NUMA nodes

**What NUMA Configuration Does:**

1. **Detects NUMA Topology:**
   - Identifies number of NUMA nodes
   - Maps CPU cores to NUMA nodes
   - Determines memory allocation policies

2. **Configures Thread Affinity:**
   - Sets up thread-to-core mapping based on NUMA topology
   - Ensures threads are pinned to cores on appropriate NUMA nodes
   - Used later in `set_numa_thread_affinity(ith)` during computation

3. **Memory Allocation Hints:**
   - Configures memory allocator to prefer local NUMA node
   - Reduces cross-node memory access

**NUMA Usage During Computation:**

**File:** `ggml/src/ggml-cpu/ggml-cpu.c` (inside `ggml_graph_compute_thread()`)

```cpp
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    
    // Set thread affinity based on NUMA topology
    set_numa_thread_affinity(state->ith);
    
    // ... rest of computation ...
}
```

**What `set_numa_thread_affinity()` Does:**
- Pins thread to a specific CPU core
- Selects core from the appropriate NUMA node
- Uses OS system calls (e.g., `pthread_setaffinity_np()` on Linux)

**SOC (System-on-Chip) and NUMA Support:**

**Question:** If a SOC has multiple cores, may it support NUMA?

**Answer:** **Most SOCs do NOT support NUMA** - they use **UMA (Uniform Memory Access)**. However, some high-end SOCs can have NUMA-like characteristics.

**1. Traditional SOCs (UMA - Uniform Memory Access):**

**Typical SOC Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│ Single SOC Die                                          │
│ ├─> CPU Cores: 0, 1, 2, 3, 4, 5, 6, 7 (all on same die)│
│ ├─> Single Memory Controller                            │
│ └─> Shared Memory: 8 GB (all cores access at same speed)│
└─────────────────────────────────────────────────────────┘

Memory Access:
- Core 0 accessing memory: ~100 ns (same for all cores)
- Core 7 accessing memory: ~100 ns (same for all cores)
- All cores have EQUAL access time → UMA (not NUMA)
```

**Characteristics:**
- **All cores on single die/chip**
- **Single memory controller**
- **Uniform memory access time** (all cores access memory at same speed)
- **No NUMA support** - NUMA configuration is not applicable

**2. High-End SOCs (Can Have NUMA-Like Characteristics):**

**Modern Server-Class SOCs (e.g., AMD EPYC, Intel Xeon):**

**Multi-Chip Module (MCM) Design:**
```
┌─────────────────────────────────────────────────────────┐
│ Single Package (SOC)                                    │
│ ├─> Die 0 (CCD 0)                                       │
│ │   ├─> Cores: 0, 1, 2, 3                              │
│ │   └─> Local Memory Controller → Memory Channel 0      │
│ ├─> Die 1 (CCD 1)                                       │
│ │   ├─> Cores: 4, 5, 6, 7                              │
│ │   └─> Local Memory Controller → Memory Channel 1      │
│ └─> Shared Memory: 16 GB (split across channels)        │
└─────────────────────────────────────────────────────────┘

Memory Access:
- Core 0 accessing Channel 0 memory: ~100 ns (local, fast)
- Core 0 accessing Channel 1 memory: ~150 ns (remote, slower)
- Different access times → NUMA-like behavior
```

**Characteristics:**
- **Multiple dies/chiplets** in single package
- **Multiple memory controllers** (one per die/chiplet)
- **Non-uniform memory access** (local vs. remote memory)
- **Can support NUMA** - OS can expose multiple NUMA nodes

**3. How to Determine if Your SOC Supports NUMA:**

**Check NUMA Topology (Linux):**
```bash
# Check number of NUMA nodes
numactl --hardware

# Example output for UMA (single node):
available: 1 nodes (0)
node 0 cpus: 0 1 2 3 4 5 6 7
node 0 size: 8192 MB
node 0 free: 4096 MB

# Example output for NUMA (multiple nodes):
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3
node 0 size: 4096 MB
node 1 cpus: 4 5 6 7
node 1 size: 4096 MB
```

**Check via /sys:**
```bash
# List NUMA nodes
ls /sys/devices/system/node/

# For UMA: node0 only
# For NUMA: node0, node1, node2, ...
```

**4. For Your Product (RISC-V SOC):**

**Most Likely Scenario: UMA (Not NUMA)**

**Typical RISC-V SOC Architecture:**
- **Single die/chip** with multiple cores
- **Single memory controller**
- **Uniform memory access** (all cores access memory at same speed)
- **No NUMA support** - NUMA configuration should be disabled

**NUMA Configuration Recommendation:**
- **Set:** `GGML_NUMA_STRATEGY_DISABLED` (default)
- **Reason:** SOC likely uses UMA architecture
- **Impact:** No performance penalty (UMA systems don't benefit from NUMA optimization)

**5. If Your SOC Does Support NUMA:**

**How to Enable:**
1. **Verify NUMA topology** using `numactl --hardware`
2. **If multiple nodes exist:** NUMA can be enabled
3. **Configure llama.cpp:** Set appropriate NUMA strategy
4. **Benefits:** Optimized memory access, reduced latency

**For QEMU Emulation (This Project):**

**NUMA is Typically Disabled:**
- **Reason:** QEMU user-mode emulation doesn't expose NUMA topology
- **Default:** `GGML_NUMA_STRATEGY_DISABLED`
- **Impact:** Threads are not pinned to specific cores
- **OS Scheduler:** Handles core assignment automatically

**Command-Line Control:**

NUMA strategy can be controlled via command-line arguments (if supported):
- Default: Disabled (for QEMU/emulation environments and most SOCs)
- Can be enabled for native systems with multiple NUMA nodes

**Summary:**
- **Most SOCs:** UMA (Uniform Memory Access) - **No NUMA support**
- **High-End SOCs:** Can have NUMA-like characteristics (multi-chip modules)
- **NUMA Configuration:** Sets up thread-to-core mapping and memory allocation policies for multi-socket/multi-die systems
- **Purpose:** Optimize memory access latency by keeping threads and memory on the same NUMA node
- **For This Project:** Typically disabled (QEMU emulation and most SOCs don't support NUMA)
- **Impact:** When disabled, OS scheduler handles thread placement (no performance penalty for UMA systems)

#### 3. CPU Core Detection (for Default Thread Count)

**File:** [`common/arg.cpp:786-794`](../dev_env/llama.cpp/common/arg.cpp#L786)

```cpp
add_opt(common_arg(
    {"-t", "--threads"}, "N",
    "number of CPU threads to use during generation (default: auto-detect)",
    [](common_params & params, int value) {
        params.cpuparams.n_threads = value;
        if (params.cpuparams.n_threads <= 0) {
            // Auto-detect: Use hardware_concurrency()
            params.cpuparams.n_threads = std::thread::hardware_concurrency();
        }
    }
));
```

**CPU Core Detection:**
- **Function**: `std::thread::hardware_concurrency()`
- **Returns**: Number of concurrent threads supported by the system
- **When used**: Only if `-t` is not specified, or set to 0 or negative
- **For your command (`-t 2`)**: CPU core count is **detected but NOT used**

**Example:**
```cpp
// If you don't specify -t:
int cores = std::thread::hardware_concurrency();  // e.g., 8 cores
params.cpuparams.n_threads = cores;              // Uses 8 threads

// If you specify -t 2:
params.cpuparams.n_threads = 2;                   // Uses 2 threads (ignores core count)
```

#### 4. Backend Device Registration

**File:** [`src/llama-context.cpp:266`](../dev_env/llama.cpp/src/llama-context.cpp#L266)

```cpp
// CPU backend (always added)
backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
backends.emplace_back(backend_cpu);
```

**What gets registered:**
- **Backend type**: CPU backend
- **Backend capabilities**: Supported operations (MUL_MAT, ADD, etc.)
- **Buffer types**: CPU buffer types (standard, IMI, etc.)
- **Kernel implementations**: Available CPU kernels

#### 5. Thread Pool Information (Later)

**File:** [`tools/main/main.cpp:170-195`](../dev_env/llama.cpp/tools/main/main.cpp#L170)

```cpp
struct ggml_threadpool_params tpp =
    ggml_threadpool_params_from_cpu_params(params.cpuparams);

struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
```

**Thread pool parameters:**
- `n_threads`: Number of threads (from `-t` or auto-detected)
- `cpumask`: CPU affinity mask (which cores to use)
- `priority`: Thread scheduling priority
- `poll`: Polling level (busywait vs sleep)

**Summary of Registered Information:**

| Information | Source | When Used | For Your Command (`-t 2`) |
|------------|--------|-----------|---------------------------|
| **Backend types** | Backend registry | Always | CPU backend registered |
| **CPU core count** | `std::thread::hardware_concurrency()` | Only if `-t` not specified | Detected but **not used** (you set `-t 2`) |
| **Thread count** | Command-line (`-t`) | Always | **2 threads** (from your `-t 2`) |
| **Backend capabilities** | Backend registration | During operation dispatch | CPU operations registered |
| **Buffer types** | Backend initialization | During tensor allocation | CPU_IMI or standard CPU |
| **NUMA topology** | `llama_numa_init()` | If NUMA enabled | Typically disabled for QEMU |

**Key Answer: CPU Core Count**

**Question:** Will the number of CPU cores be passed to llama.cpp?

**Answer:** 
- **CPU core count is detected** via `std::thread::hardware_concurrency()`
- **But it's NOT directly passed to llama.cpp**
- **Only used as default** if you don't specify `-t`
- **For your command (`-t 2`)**: CPU core count is **ignored** - you explicitly set 2 threads
- **What llama.cpp uses**: The thread count you specify (`-t 2`), not the core count

**The thread count (`-t 2`) is what matters, not the CPU core count.**

### 0.5 Complete Initialization Sequence

**Step-by-Step Flow**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. COMMAND-LINE PARSING                                     │
│    File: tools/main/main.cpp:86                             │
│    ├─> common_params_parse(argc, argv, params)              │
│    └─> Result: common_params structure filled              │
│        ├─> params.model.path = "stories15M-q4_0.gguf"     │
│        ├─> params.n_ctx = 512 (or default)                 │
│        ├─> params.cpuparams.n_threads = 2 (from -t 2)     │
│        ├─> CPU core count detected (but not used)         │
│        ├─> params.n_gpu_layers = 0 (from -ngl 0)          │
│        └─> ... all other options ...                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 2. BACKEND INITIALIZATION                                    │
│    File: tools/main/main.cpp:123-126                        │
│    ├─> llama_backend_init()                                │
│    │   ├─> Initialize GGML backend registry                │
│    │   ├─> Register CPU backend                             │
│    │   ├─> Register REF backend (if enabled)                 │
│    │   └─> Register GPU backends (if available, skipped)    │
│    │                                                         │
│    └─> llama_numa_init(params.numa)                        │
│        └─> NUMA configuration (typically disabled)          │
│                                                             │
│    Registered Information:                                  │
│    ├─> Backend types: CPU (and REF if enabled)              │
│    ├─> Backend capabilities: Supported operations          │
│    ├─> Buffer types: CPU buffer types                      │
│    └─> Kernel implementations: Available CPU kernels        │
│                                                             │
│    Note: CPU core count detected but NOT registered        │
│    └─> Only used as default if -t not specified            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 3. MODEL LOADING                                            │
│    File: common/common.cpp:956                              │
│    ├─> llama_model_load_from_file()                        │
│    │   ├─> Parse GGUF file                                 │
│    │   ├─> Load metadata (architecture, hyperparameters)  │
│    │   ├─> Load vocabulary                                 │
│    │   └─> Load tensor weights → CPU backend buffers      │
│    └─> Result: llama_model * model                        │
│        ├─> model.hparams (hyperparameters)                 │
│        ├─> model.vocab (vocabulary)                         │
│        ├─> model.tensors (weight tensors in CPU buffers)    │
│        └─> model.devices (GPU devices, if any)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 4. CONTEXT CREATION                                         │
│    File: common/common.cpp:965-967                         │
│    ├─> common_context_params_to_llama(params)             │
│    │   └─> Convert command-line params to context params  │
│    │                                                       │
│    └─> llama_init_from_model(model, cparams)             │
│        └─> llama_context::llama_context() constructor     │
│            ├─> Store model reference                       │
│            ├─> Store context parameters                    │
│            ├─> Register backends (CPU, GPU if any)         │
│            ├─> Create backend scheduler                    │
│            ├─> Allocate KV cache memory                    │
│            └─> Create batch allocator                     │
│                                                             │
│    Result: llama_context * ctx                            │
│        ├─> ctx.model = model (reference)                   │
│        ├─> ctx.cparams = cparams (from command-line)      │
│        ├─> ctx.sched = backend scheduler                  │
│        ├─> ctx.memory = KV cache (empty, ready for tokens)│
│        └─> ctx.balloc = batch allocator                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 5. CONTEXT FETCHED IN MAIN()                                │
│    File: tools/main/main.cpp:140-143                       │
│    ├─> common_init_result llama_init = ...                │
│    ├─> model = llama_init.model.get()                     │
│    └─> ctx = llama_init.context.get()  ← CONTEXT FETCHED! │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 6. THREAD POOL ATTACHMENT                                   │
│    File: tools/main/main.cpp:159-195                       │
│    ├─> Create thread pools (decode, prefill)               │
│    ├─> llama_attach_threadpool(ctx, threadpool, ...)      │
│    └─> Context is now fully ready for inference            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 7. READY FOR INFERENCE                                      │
│    Context is complete and ready to:                       │
│    ├─> Process input tokens                                │
│    ├─> Build computation graphs                            │
│    ├─> Execute on CPU backend                              │
│    └─> Generate output tokens                              │
└─────────────────────────────────────────────────────────────┘
```

**What Context Contains When Fetched**

At the point where `ctx = llama_init.context.get()` is called (line 143), the context contains:

#### ✅ Fully Initialized:
1. **Model Reference** - Points to loaded model with all weights
2. **Context Parameters** - All parameters from command-line:
   - `n_ctx`: Context size (e.g., 512 or 2048)
   - `n_batch`: Batch size (e.g., 512)
   - `n_threads`: Decode threads (e.g., 1 from `-t 1`)
   - `n_threads_batch`: Prefill threads
   - `n_gpu_layers`: GPU layers (e.g., 0 from `-ngl 0`)
3. **Backends** - CPU backend registered, GPU backends (if any)
4. **Backend Scheduler** - Ready to assign operations to backends
5. **KV Cache Memory** - Allocated and ready (but empty)
6. **Batch Allocator** - Ready to process batches

#### ⏳ Initialized Later:
1. **Thread Pools** - Created and attached in `main()` lines 159-195
2. **Computation Graphs** - Built during inference
3. **KV Cache Data** - Populated during inference

**Key Insights:**

1. **Context is Runtime State**: The `llama_context` is **not** the model. It's the **runtime state** for inference:
   - **Model** = Static data (weights, vocabulary, architecture)
   - **Context** = Dynamic state (KV cache, computation graphs, backends)

2. **Context is Created from Model**: The context needs the model because it references the model's weights, uses the model's vocabulary, and needs model's hyperparameters.

3. **Context Parameters Come from Command-Line**: All user-configurable options are converted to context parameters:
   - `-t 1` → `cparams.n_threads = 1`
   - `-ngl 0` → `cparams.n_gpu_layers = 0`
   - `-c 2048` → `cparams.n_ctx = 2048`

4. **Context is Fetched Before Thread Pools**: The context is fetched at line 143, but thread pools are attached later (line 195). This allows flexible thread pool configuration.

---

## Phase 1: Model Loading

### 1.2 Model Loading Function

**File:** [`common/common.cpp:952`](../dev_env/llama.cpp/common/common.cpp#L952)

```cpp
struct common_init_result common_init_from_params(common_params & params) {
    // Convert parameters
    auto mparams = common_model_params_to_llama(params);
    
    // Load model from GGUF file
    llama_model * model = llama_model_load_from_file(
        params.model.path.c_str(),  // "stories15M-q4_0.gguf"
        mparams);
    
    // Create context
    auto cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    
    return {model, ctx};
}
```

### 1.3 GGUF File Parsing

**File:** [`src/llama.cpp:155`](../dev_env/llama.cpp/src/llama.cpp#L155)

```cpp
static struct llama_model * llama_model_load_from_file_impl(
        const std::string & path_model,
        std::vector<std::string> & splits,
        struct llama_model_params params) {
    
    llama_model * model = new llama_model(params);
    
    // Load model from GGUF file
    int ret = llama_model_load(path_model, splits, *model, params);
    if (ret != 0) {
        delete model;
        return nullptr;
    }
    
    return model;
}
```

**File:** [`src/llama.cpp:102`](../dev_env/llama.cpp/src/llama.cpp#L102)

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
        params.use_mmap,          // Memory mapping enabled
        params.check_tensors,
        params.kv_overrides,
        params.tensor_buft_overrides);
    
    // Load architecture-specific components
    model.load_arch(ml);          // Load architecture metadata
    model.load_hparams(ml);       // Load hyperparameters
    model.load_vocab(ml);         // Load vocabulary
    
    // Load tensor weights
    if (!model.load_tensors(ml)) {
        return -2;
    }
    
    return 0;
}
```

### 1.4 Tensor Loading and Buffer Allocation

**File:** [`src/llama-model.cpp:load_tensors()`](../dev_env/llama.cpp/src/llama-model.cpp)

```cpp
bool llama_model::load_tensors(llama_model_loader & ml) {
    // Iterate through all tensors in GGUF file
    for (size_t i = 0; i < ml.n_tensors; i++) {
        const std::string & name = ml.get_tensor_name(i);
        const auto & tensor_weight = ml.get_tensor_weight(name);
        
        // Create GGML tensor
        struct ggml_tensor * cur = ggml_new_tensor(
            ctx,                    // GGML context
            tensor_weight.type,     // GGML_TYPE_Q4_0, etc.
            tensor_weight.n_dims,  // Number of dimensions
            tensor_weight.ne);      // Shape [n_embd, vocab_size, ...]
        
        ggml_set_name(cur, name.c_str());
        
        // Determine buffer type for this tensor
        ggml_backend_buffer_type_t buft = get_buffer_type(cur);
        
        // Allocate buffer in backend
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(
            buft,                   // CPU backend buffer type
            ggml_nbytes(cur));      // Size in bytes
        
        // Attach buffer to tensor
        cur->buffer = buffer;
        
        // Load data from GGUF file
        if (ml.use_mmap) {
            // Memory-mapped: create view into file
            ml.load_data_for(cur);
        } else {
            // Direct read: copy data from file
            ml.load_data_for(cur);
        }
        
        // Repack weights if needed (for IMI, AMX, etc.)
        if (buft == ggml_backend_cpu_imi_buffer_type()) {
            // Repack to IMI-optimized format
            ggml_cpu_imi_repack_tensor(cur);
        }
        
        // Store tensor in model
        model.tensors[name] = cur;
    }
    
    return true;
}
```

**Key Points:**
- **Buffer Type Selection**: `get_buffer_type()` determines which backend buffer to use
  - For CPU-only builds: `GGML_BACKEND_CPU_BUFFER_TYPE`
  - For IMI builds: `GGML_BACKEND_CPU_IMI_BUFFER_TYPE` (if enabled)
- **Memory Mapping**: Large models use memory mapping to avoid loading entire file into RAM
- **Weight Repacking**: IMI/AMX backends repack weights into optimized formats during load

### 1.5 Buffer Type Selection

**File:** [`src/llama-model.cpp:get_buffer_type()`](../dev_env/llama.cpp/src/llama-model.cpp)

```cpp
ggml_backend_buffer_type_t llama_model::get_buffer_type(
    const struct ggml_tensor * tensor) const {
    
    // Check for explicit override
    if (tensor_buft_overrides.count(tensor->name)) {
        return tensor_buft_overrides[tensor->name];
    }
    
    // Default: use CPU backend buffer type
    #ifdef GGML_USE_CPU_IMI
        // For IMI builds, use IMI buffer type
        if (params.use_imi) {
            return ggml_backend_cpu_imi_buffer_type();
        }
    #endif
    
    // Standard CPU buffer type
    return ggml_backend_cpu_buffer_type();
}
```

**Result:**
- All model weights are allocated in **CPU backend buffers**
- Buffer type determines which kernels will be used later
- CPU_IMI buffer type triggers IMI-optimized kernels

---

## Phase 2: Context Initialization

### 2.1 Context Creation

**File:** [`src/llama.cpp:llama_init_from_model()`](../dev_env/llama.cpp/src/llama.cpp)

```cpp
llama_context * llama_init_from_model(
        struct llama_model * model,
        struct llama_context_params params) {
    
    return new llama_context(*model, params);
}
```

### 2.2 Context Constructor

**File:** [`src/llama-context.cpp:19`](../dev_env/llama.cpp/src/llama-context.cpp#L19)

```cpp
llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    
    // Store context parameters
    cparams.n_ctx = params.n_ctx;              // Context size (e.g., 2048)
    cparams.n_batch = params.n_batch;          // Batch size (e.g., 512)
    cparams.n_threads = params.n_threads;      // Decode threads (e.g., 1)
    cparams.n_threads_batch = params.n_threads_batch;  // Prefill threads
    
    // Initialize backends
    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    
    // GPU backends (if available) - SKIPPED in CPU-only builds
    // For CPU-only execution: model.devices is empty, so this loop doesn't execute
    for (auto * dev : model.devices) {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        backends.emplace_back(backend);
        backend_buft.emplace_back(ggml_backend_get_default_buffer_type(backend));
    }
    
    // CPU backend (always added, and ONLY backend in CPU-only builds)
    backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    backends.emplace_back(backend_cpu);
    backend_buft.emplace_back(ggml_backend_get_default_buffer_type(backend_cpu));
    
    // Result: backends = [CPU] (single backend for CPU-only execution)
    
    // Create backend scheduler
    sched.reset(ggml_backend_sched_new(
        backends.data(),           // Backend array [CPU, REF]
        backend_buft.data(),       // Buffer types
        backends.size(),           // Number of backends
        max_nodes,                 // Max graph nodes (~200)
        false,                     // Pipeline parallelism (disabled)
        cparams.op_offload));      // Operation offload config
    
    // Initialize KV cache memory
    memory.reset(model.create_memory(params_mem, cparams));
}
```

**Key Components:**
- **Backends**: Array of available backends (CPU only in this project, REF optional for validation)
- **Scheduler**: Manages operation assignment (all operations → CPU backend)
- **Memory**: KV cache for attention keys/values (allocated in CPU memory)
- **Batch Allocator**: Manages batch splitting and unified batches

**For CPU-only execution:**
- `backends.size() == 1` (only CPU backend)
- `backends[0] == backend_cpu` (CPU backend)
- All operations are assigned to CPU backend
- No GPU backends, no cross-backend transfers

### 2.3 Backend Scheduler Initialization

**File:** [`ggml/src/ggml-backend.cpp:678`](../dev_env/llama.cpp/ggml/src/ggml-backend.cpp#L678)

```cpp
ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t max_nodes,
        bool pipeline_parallel,
        bool op_offload) {
    
    struct ggml_backend_sched * sched = new ggml_backend_sched;
    
    sched->n_backends = n_backends;
    for (int i = 0; i < n_backends; i++) {
        sched->backends[i] = backends[i];
        sched->bufts[i] = bufts[i];
    }
    
    // Initialize graph allocator
    sched->galloc = ggml_gallocr_new(sched->bufts, n_backends);
    
    // Initialize hash table for tensor-to-backend mapping
    sched->hash_set = ggml_hash_set_new(max_nodes);
    sched->hv_tensor_backend_ids = new int[max_nodes];
    sched->hv_tensor_copies = new ggml_tensor*[max_nodes];
    
    // Initialize node backend assignments
    sched->node_backend_ids = new int[max_nodes];
    sched->leaf_backend_ids = new int[max_nodes];
    
    sched->op_offload = op_offload;
    
    return sched;
}
```

**Scheduler Responsibilities:**
1. **Tensor-to-Backend Mapping**: Track which backend owns each tensor (all → CPU)
2. **Operation Assignment**: Decide which backend executes each operation (all → CPU)
3. **Memory Management**: Allocate tensors in appropriate backend buffers (all → CPU buffers)
4. **Graph Splitting**: Split graph at backend boundaries (single split for CPU-only)

**For CPU-only execution:**
- All tensors mapped to CPU backend (backend_id = 0)
- All operations assigned to CPU backend
- Single graph split (all nodes → CPU)
- No cross-backend data transfers needed

---

## Phase 3: Graph Construction

### 3.1 Inference Entry Point

**File:** [`tools/main/main.cpp:573`](../dev_env/llama.cpp/tools/main/main.cpp#L573)

```cpp
while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
    // Tokenize input if needed
    if (!embd.empty()) {
        // Process tokens in batches
        for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
            int n_eval = (int) embd.size() - i;
            if (n_eval > params.n_batch) {
                n_eval = params.n_batch;
            }
            
            // Create batch
            llama_batch batch = llama_batch_get_one(
                &embd[i], n_eval, n_past, 0, true);
            
            // Decode batch
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("failed to decode\n");
                return 1;
            }
            
            n_past += n_eval;
        }
    }
    
    // Sample next token
    const llama_token id = common_sampler_sample(smpl, ctx, -1);
    common_sampler_accept(smpl, id, true);
    
    embd.push_back(id);
    --n_remain;
}
```

### 3.2 Decode Function

**File:** [`src/llama-context.cpp:2773`](../dev_env/llama.cpp/src/llama-context.cpp#L2773)

```cpp
int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    return ctx->decode(batch);
}
```

**File:** [`src/llama-context.cpp:109`](../dev_env/llama.cpp/src/llama-context.cpp#L109)

```cpp
int llama_context::decode(const llama_batch & batch_inp) {
    // Initialize batch allocator
    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, 
                      cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, 
                      output_all)) {
        return -1;
    }
    
    // Initialize memory context
    llama_memory_context_ptr mctx;
    mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
    
    // Process unified batches
    do {
        const auto & ubatch = mctx->get_ubatch();
        
        // Process unified batch
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

### 3.3 Graph Building

**File:** [`src/llama-context.cpp:103`](../dev_env/llama.cpp/src/llama-context.cpp#L103)

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

**File:** [`src/llama-graph.cpp:graph_build()`](../dev_env/llama.cpp/src/llama-graph.cpp)

```cpp
llm_graph_result * llama_context::graph_build(
        const llama_ubatch & ubatch,
        llm_graph_type gtype,
        llama_memory_context_i * mctx) {
    
    // Create new graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    
    // Input: token embeddings
    auto tok_embd = ggml_get_rows(model.tok_embeddings, ubatch.token);
    // tok_embd shape: [n_tokens, n_embd]
    
    auto hidden = tok_embd;
    
    // Process each transformer layer
    for (int i = 0; i < model.hparams.n_layer; i++) {
        // Pre-attention layer norm
        hidden = ggml_rms_norm(ctx, hidden, model.layers[i].attn_norm);
        
        // QKV projections
        auto q = ggml_mul_mat(ctx, model.layers[i].attn_q, hidden);
        auto k = ggml_mul_mat(ctx, model.layers[i].attn_k, hidden);
        auto v = ggml_mul_mat(ctx, model.layers[i].attn_v, hidden);
        
        // Apply RoPE
        q = ggml_rope(ctx, q, ubatch.pos, n_ctx, freq_base, freq_scale);
        k = ggml_rope(ctx, k, ubatch.pos, n_ctx, freq_base, freq_scale);
        
        // Attention computation
        auto attn_out = compute_attention(ctx, q, k, v, mctx, i);
        
        // Output projection
        attn_out = ggml_mul_mat(ctx, model.layers[i].attn_o, attn_out);
        
        // Residual connection
        hidden = ggml_add(ctx, hidden, attn_out);
        
        // Pre-FFN layer norm
        hidden = ggml_rms_norm(ctx, hidden, model.layers[i].ffn_norm);
        
        // Feed-forward network
        auto gate = ggml_mul_mat(ctx, model.layers[i].ffn_gate, hidden);
        auto up = ggml_mul_mat(ctx, model.layers[i].ffn_up, hidden);
        gate = ggml_silu(ctx, gate);
        auto ffn_out = ggml_mul(ctx, gate, up);
        ffn_out = ggml_mul_mat(ctx, model.layers[i].ffn_down, ffn_out);
        
        // Residual connection
        hidden = ggml_add(ctx, hidden, ffn_out);
    }
    
    // Output layer
    hidden = ggml_rms_norm(ctx, hidden, model.norm);
    auto logits = ggml_mul_mat(ctx, model.output, hidden);
    
    // Set graph output
    gf->nodes[gf->n_nodes - 1] = logits;
    gf->n_leafs = 1;
    gf->leafs[0] = ubatch.token;
    
    return gf;
}
```

**Graph Structure:**
- **Input**: Token IDs `[n_tokens]`
- **Embedding**: Token embeddings `[n_tokens, n_embd]`
- **Layers**: N transformer layers, each with:
  - Attention (QKV projections, RoPE, attention computation)
  - Feed-forward (gate, up, down projections)
- **Output**: Logits `[n_tokens, vocab_size]`

**Total Nodes:** ~3 + (n_layers × 25) + 2 ≈ 155 nodes for 6-layer model

---

## Phase 4: Backend Scheduling

### 4.1 Graph Allocation

**File:** [`ggml/src/ggml-backend.cpp:1282`](../dev_env/llama.cpp/ggml/src/ggml-backend.cpp#L1282)

```cpp
bool ggml_backend_sched_alloc_graph(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph) {
    
    // Reset scheduler state
    ggml_backend_sched_reset(sched);
    
    // Pass 1: Assign backends to all operations
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);
        
        if (backend_id == -1) {
            // Default to CPU backend (last backend)
            backend_id = sched->n_backends - 1;
        }
        
        // Store assignment
        sched->node_backend_ids[i] = backend_id;
        ggml_hash_set_insert(sched->hash_set, node, backend_id);
    }
    
    // Pass 2: Split graph at backend boundaries
    ggml_backend_sched_split_graph(sched, graph);
    
    // Pass 3: Allocate tensors in backend buffers
    if (!ggml_gallocr_alloc_graph(sched->galloc, graph)) {
        return false;
    }
    
    return true;
}
```

### 4.2 Backend Assignment Algorithm

**File:** [`ggml/src/ggml-backend.cpp:776`](../dev_env/llama.cpp/ggml/src/ggml-backend.cpp#L776)

```cpp
int ggml_backend_sched_backend_id_from_cur(
        ggml_backend_sched_t sched,
        struct ggml_tensor * tensor) {
    
    // STEP 1: Check if tensor already has a buffer
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        return cur_backend_id;  // Use existing buffer's backend
    }
    
    // STEP 2: Handle view tensors (use source tensor's backend)
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(
            sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            return cur_backend_id;
        }
    }
    
    // STEP 3: Graph inputs default to CPU backend
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        return sched->n_backends - 1;  // CPU backend (last backend)
    }
    
    // STEP 4: Data locality - prefer backend where weights are located
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) continue;
        
        // If source is a weight tensor, use its backend
        if (src->buffer != NULL &&
            src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(
                sched, src, tensor);
            return src_backend_id;  // Use weight's backend
        }
    }
    
    // Default: CPU backend
    return sched->n_backends - 1;
}
```

**Key Principle: Data Locality**
- Operations run on the backend where their **weights are stored**
- **For CPU-only execution**: All operations → CPU backend (only backend available)
- **No data movement**: All tensors stay in CPU memory, no cross-backend transfers
- **Simplified execution**: Single backend means no complex scheduling decisions

### 4.3 Graph Splitting

**File:** [`ggml/src/ggml-backend.cpp:912`](../dev_env/llama.cpp/ggml/src/ggml-backend.cpp#L912)

```cpp
void ggml_backend_sched_split_graph(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph) {
    
    int n_splits = 0;
    struct ggml_backend_sched_split * cur_split = &sched->splits[0];
    
    // Initialize first split
    cur_split->backend_id = sched->node_backend_ids[0];
    cur_split->i_start = 0;
    cur_split->n_inputs = 0;
    
    int cur_backend_id = cur_split->backend_id;
    
    // Group consecutive nodes with same backend
    for (int i = 0; i < graph->n_nodes; i++) {
        const int node_backend_id = sched->node_backend_ids[i];
        
        // Need new split if backend changed
        if (node_backend_id != cur_backend_id) {
            // Finalize current split
            cur_split->i_end = i;
            
            // Create new split
            n_splits++;
            cur_split = &sched->splits[n_splits];
            cur_split->backend_id = node_backend_id;
            cur_split->i_start = i;
            cur_split->n_inputs = 0;
            cur_backend_id = node_backend_id;
        }
        
        // Track cross-backend dependencies
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = graph->nodes[i]->src[j];
            if (src && ggml_backend_sched_backend_id(sched, src) != cur_backend_id) {
                cur_split->inputs[cur_split->n_inputs++] = src;  // Needs copy
            }
        }
    }
    
    cur_split->i_end = graph->n_nodes;
    sched->n_splits = n_splits + 1;
}
```

**For CPU-only execution:**
- **Single split**: All nodes → CPU backend (only backend available)
- **No cross-backend copies**: All tensors already in CPU memory
- **Simple execution path**: Direct dispatch to CPU backend, no scheduling complexity
- **Graph structure**: All ~155 nodes processed in one continuous split

---

## Phase 5: CPU Backend Execution

### 5.1 Graph Execution (CPU-Only)

**File:** [`ggml/src/ggml-backend.cpp:1282`](../dev_env/llama.cpp/ggml/src/ggml-backend.cpp#L1282)

```cpp
ggml_status ggml_backend_sched_graph_compute_async(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph) {
    
    // Execute each split
    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        ggml_backend_t backend = sched->backends[split->backend_id];
        
        // Copy inputs from other backends if needed
        // For CPU-only: This loop is empty (no cross-backend dependencies)
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * src = split->inputs[j];
            int src_backend_id = ggml_backend_sched_backend_id(sched, src);
            
            if (src_backend_id != split->backend_id) {
                // Copy tensor to this backend
                // For CPU-only: This condition is never true
                ggml_backend_tensor_copy(
                    sched->backends[src_backend_id],
                    backend,
                    src, split->copies[j]);
            }
        }
        
        // Create subgraph for this split
        struct ggml_cgraph * subgraph = create_subgraph(graph, split);
        
        // Execute on backend (CPU backend for CPU-only execution)
        ggml_backend_graph_compute(backend, subgraph);
    }
    
    return GGML_STATUS_SUCCESS;
}
```

**For CPU-only execution:**
- `sched->n_splits == 1` (single split, all nodes → CPU)
- `split->n_inputs == 0` (no cross-backend inputs)
- `backend == backend_cpu` (CPU backend)
- **No data copies**: All tensors already in CPU memory
- **Direct execution**: Graph executed directly on CPU backend

### 5.2 CPU Backend Graph Compute

**File:** [`ggml/src/ggml-backend-cpu.cpp:ggml_backend_cpu_graph_compute()`](../dev_env/llama.cpp/ggml/src/ggml-backend-cpu.cpp)

```cpp
static void ggml_backend_cpu_graph_compute(
        ggml_backend_t backend,
        struct ggml_cgraph * graph) {
    
    // Get thread pool from context
    struct ggml_threadpool * threadpool = 
        ggml_backend_cpu_get_threadpool(backend);
    
    // Create compute plan
    struct ggml_cplan cplan = ggml_graph_plan(graph, threadpool);
    
    // Allocate work buffer
    if (cplan.work_size > 0) {
        cplan.work_data = malloc(cplan.work_size);
    }
    
    // Execute graph with thread pool
    ggml_graph_compute_threaded(graph, &cplan, threadpool);
    
    // Free work buffer
    if (cplan.work_data) {
        free(cplan.work_data);
    }
}
```

### 5.3 Thread Pool Execution

**File:** [`ggml/src/ggml-threading.cpp:ggml_graph_compute_threaded()`](../dev_env/llama.cpp/ggml/src/ggml-threading.cpp)

```cpp
void ggml_graph_compute_threaded(
        struct ggml_cgraph * graph,
        struct ggml_cplan * cplan,
        struct ggml_threadpool * threadpool) {
    
    int n_threads = threadpool->n_threads;
    
    // Create compute state
    struct ggml_compute_state * states = 
        malloc(n_threads * sizeof(struct ggml_compute_state));
    
    // Initialize states
    for (int i = 0; i < n_threads; i++) {
        states[i].ith = i;
        states[i].nth = n_threads;
        states[i].cgraph = graph;
        states[i].cplan = cplan;
        states[i].threadpool = threadpool;
    }
    
    // Wake/spawn worker threads
    for (int i = 1; i < n_threads; i++) {
        ggml_threadpool_work(threadpool, ggml_graph_compute_thread, &states[i]);
    }
    
    // Main thread also processes
    ggml_graph_compute_thread(&states[0]);
    
    // Wait for all threads
    ggml_threadpool_wait(threadpool);
    
    free(states);
}
```

### 5.4 Thread Work Function

**File:** [`ggml/src/ggml-threading.cpp:ggml_graph_compute_thread()`](../dev_env/llama.cpp/ggml/src/ggml-threading.cpp)

```cpp
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_cgraph * cgraph = state->cgraph;
    struct ggml_threadpool * tp = state->threadpool;
    
    // Set thread affinity (NUMA-aware)
    set_numa_thread_affinity(state->ith);
    
    struct ggml_compute_params params = {
        .ith = state->ith,           // Thread ID (0, 1, 2, 3, ...)
        .nth = tp->n_threads,        // Total threads
        .wsize = state->cplan->work_size,
        .wdata = state->cplan->work_data,
        .threadpool = tp,
    };
    
    // ALL threads process ALL nodes sequentially
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];
        
        // Compute this node (work is split among threads)
        ggml_compute_forward(&params, node);
        
        // Synchronize between nodes
        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(tp);
        }
    }
    
    return 0;
}
```

**Key Points:**
- **All threads process all nodes** sequentially
- **Within each node**, work is split among threads
- **Barrier synchronization** ensures correctness between nodes

---

## Phase 5.5: Multi-Threading Workload Distribution (CPU-Only)

### 5.5.1 How `-t` (Thread Count) Impacts Workload Distribution

**Question:** How does the number of threads (`-t`) impact the distribution of inference workload in CPU-only llama.cpp?

**Answer:** Thread count affects **work partitioning WITHIN each operation**, not the assignment of operations to threads.

#### Execution Model

**Key Principle:** All threads process ALL nodes sequentially, but each thread processes a different portion of each node.

```
Graph Execution Flow (with -t 2):
┌─────────────────────────────────────────────────────────────┐
│ Graph: 145 nodes (all assigned to CPU backend)             │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────▼─────────────┐
        │ For each node (sequential)│
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │ Node N: MUL_MAT operation │
        │ Input: [batch, 288]       │
        │ Weight: [288, 288]        │
        │ Output: [batch, 288]       │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │ Work Partitioning:        │
        │                            │
        │ Thread 0 (ith=0, nth=2):   │
        │   Process rows [0, 2, 4...]│
        │                            │
        │ Thread 1 (ith=1, nth=2):   │
        │   Process rows [1, 3, 5...]│
        │                            │
        │ Both threads work on      │
        │ SAME node simultaneously   │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │ Barrier Synchronization    │
        │ Wait for all threads       │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │ Move to next node          │
        └────────────────────────────┘
```

### 5.5.2 When Does Distribution/Partition Start?

**Question:** In which phase does the distribution and partition start?

**Answer:** Distribution starts **immediately when graph execution begins**, in both **prefill** and **decode** phases.

#### Phase 1: Prefill Phase (Initial Prompt Processing)

**File:** [`src/llama-context.cpp:process_ubatch()`](../dev_env/llama.cpp/src/llama-context.cpp#L1092)

```cpp
// Prefill phase: Process entire prompt
const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);

// Inside process_ubatch():
// 1. Build computation graph (all prompt tokens)
// 2. Allocate tensors
// 3. Execute graph → Thread distribution starts HERE
ggml_backend_sched_graph_compute_async(sched, &graph);
```

**Thread Pool Used:** `threadpool_batch` (if different from decode pool) or `threadpool` (if same)

**When Distribution Starts:**
- **Graph building**: Sequential (single thread)
- **Tensor allocation**: Sequential (single thread)
- **Graph execution**: **Multi-threaded distribution starts HERE**

#### Phase 2: Decode Phase (Autoregressive Token Generation)

**File:** [`src/llama-context.cpp:decode()`](../dev_env/llama.cpp/src/llama-context.cpp#L850)

```cpp
// Decode phase: Generate one token at a time
int decode(struct llama_context * ctx, const llama_batch & batch) {
    // 1. Build computation graph (single token)
    // 2. Allocate tensors
    // 3. Execute graph → Thread distribution starts HERE
    ggml_backend_sched_graph_compute_async(sched, &graph);
}
```

**Thread Pool Used:** `threadpool` (decode thread pool)

**When Distribution Starts:**
- **Graph building**: Sequential (single thread)
- **Tensor allocation**: Sequential (single thread)
- **Graph execution**: **Multi-threaded distribution starts HERE**

**Summary:**
- **Distribution starts**: When `ggml_backend_graph_compute()` is called
- **Applies to**: Both prefill and decode phases
- **Location**: Inside `ggml_backend_cpu_graph_compute()` → `ggml_graph_compute_threaded()`

### 5.5.3 How Do Multiple Threads Coordinate?

**Question:** How do multiple threads coordinate with each other?

**Answer:** Threads coordinate through **barrier synchronization** between graph nodes and **work partitioning** within each operation.

#### Coordination Mechanism 1: Barrier Synchronization

**File:** [`ggml/src/ggml-cpu/ggml-cpu.c:ggml_barrier()`](../dev_env/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c#L543)

```cpp
void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed);
    if (n_threads == 1) {
        return;  // No synchronization needed for single thread
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    // Custom barrier using atomics and condition variables
    // Ensures all threads reach this point before continuing
    atomic_fetch_add(&tp->barrier_counter, 1);
    while (atomic_load(&tp->barrier_counter) < n_threads) {
        // Spin-wait or sleep
    }
    atomic_store(&tp->barrier_counter, 0);
#endif
}
```

**Barrier Usage:**
```cpp
// In ggml_graph_compute_thread():
for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
    struct ggml_tensor * node = cgraph->nodes[node_n];
    
    // All threads compute their portion of this node
    ggml_compute_forward(&params, node);
    
    // BARRIER: Wait for all threads to finish this node
    if (node_n + 1 < cgraph->n_nodes) {
        ggml_barrier(tp);  // ← Coordination point
    }
}
```

**Why Barriers Are Needed:**
- **Data dependencies**: Node N+1 may depend on Node N's output
- **Correctness**: All threads must finish Node N before starting Node N+1
- **Memory consistency**: Ensures all writes from Node N are visible to all threads

#### Coordination Mechanism 2: Work Partitioning Within Operations

**Example: Matrix Multiplication (MUL_MAT)**

**File:** [`ggml/src/ggml-cpu/ggml-cpu.c:ggml_compute_forward_mul_mat()`](../dev_env/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c#L1221)

```cpp
void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    
    const int ith = params->ith;  // Thread ID: 0, 1, 2, ...
    const int nth = params->nth;  // Total threads: 2 (if -t 2)
    
    const int64_t nr0 = ne0;  // Output rows
    const int64_t nr1 = ne1;  // Output cols
    
    // Divide work into chunks
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    
    // Each thread processes specific chunks
    int current_chunk = ith;  // Thread 0: chunk 0, Thread 1: chunk 1, ...
    
    while (current_chunk < nchunk0 * nchunk1) {
        // Calculate which rows/cols this chunk covers
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;
        
        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);
        
        // Process this chunk
        ggml_compute_forward_mul_mat_one_chunk(params, dst,
            ir0_start, ir0_end, ir1_start, ir1_end);
        
        // Work stealing: Get next chunk (atomic increment)
        current_chunk = atomic_fetch_add(&tp->current_chunk, nth);
    }
}
```

**Work Distribution Example (with `-t 2`):**

```
Operation: MUL_MAT
Input: [batch=1, 288]
Weight: [288, 288]
Output: [batch=1, 288]  (288 rows to compute)

With 2 threads:
├─> Thread 0 (ith=0, nth=2):
│   ├─> Processes chunks: 0, 2, 4, 6, 8, ...
│   └─> Computes rows: [0-143] (approximately)
│
└─> Thread 1 (ith=1, nth=2):
    ├─> Processes chunks: 1, 3, 5, 7, 9, ...
    └─> Computes rows: [144-287] (approximately)

Both threads:
├─> Work on SAME operation simultaneously
├─> Write to DIFFERENT portions of output tensor
└─> No data races (different memory regions)
```

**Work Stealing:**
- If one thread finishes early, it can grab additional chunks
- Uses atomic operations for thread-safe chunk assignment
- Improves load balancing

#### Coordination Mechanism 3: Thread Pool Management

**File:** [`ggml/src/ggml-threading.cpp:ggml_graph_compute_threaded()`](../dev_env/llama.cpp/ggml/src/ggml-threading.cpp)

```cpp
void ggml_graph_compute_threaded(
        struct ggml_cgraph * graph,
        struct ggml_cplan * cplan,
        struct ggml_threadpool * threadpool) {
    
    int n_threads = threadpool->n_threads;  // e.g., 2 (from -t 2)
    
    // Create compute state for each thread
    struct ggml_compute_state * states = 
        malloc(n_threads * sizeof(struct ggml_compute_state));
    
    // Initialize states
    for (int i = 0; i < n_threads; i++) {
        states[i].ith = i;              // Thread ID
        states[i].nth = n_threads;     // Total threads
        states[i].cgraph = graph;
        states[i].cplan = cplan;
        states[i].threadpool = threadpool;
    }
    
    // Wake/spawn worker threads (threads 1, 2, 3, ...)
    for (int i = 1; i < n_threads; i++) {
        ggml_threadpool_work(threadpool, ggml_graph_compute_thread, &states[i]);
    }
    
    // Main thread (thread 0) also processes
    ggml_graph_compute_thread(&states[0]);
    
    // Wait for all threads to complete
    ggml_threadpool_wait(threadpool);
    
    free(states);
}
```

**Thread Lifecycle:**
1. **Spawn**: Worker threads are woken/spawned
2. **Execute**: All threads run `ggml_graph_compute_thread()` in parallel
3. **Synchronize**: Barriers ensure coordination between nodes
4. **Complete**: Main thread waits for all workers via `ggml_threadpool_wait()`

### 5.5.4 Complete Multi-Threading Flow (Example: `-t 2`)

**Complete Execution Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. GRAPH EXECUTION STARTS                                   │
│    File: ggml_backend_cpu_graph_compute()                  │
│    Thread pool: 2 threads (from -t 2)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 2. THREAD SPAWNING                                           │
│    ├─> Main thread (thread 0): Starts immediately           │
│    └─> Worker thread (thread 1): Woken from thread pool     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 3. FOR EACH NODE (Sequential, all threads participate)     │
│                                                              │
│    Node 0: MUL_MAT (Q projection)                           │
│    ├─> Thread 0: Computes rows [0, 2, 4, ...]             │
│    ├─> Thread 1: Computes rows [1, 3, 5, ...]             │
│    └─> BARRIER: Wait for both threads                       │
│                                                              │
│    Node 1: MUL_MAT (K projection)                           │
│    ├─> Thread 0: Computes rows [0, 2, 4, ...]             │
│    ├─> Thread 1: Computes rows [1, 3, 5, ...]             │
│    └─> BARRIER: Wait for both threads                       │
│                                                              │
│    ... (continue for all 145 nodes)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 4. THREAD COMPLETION                                         │
│    ├─> All threads finish processing all nodes              │
│    └─> Main thread waits for worker thread                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 5. GRAPH EXECUTION COMPLETE                                  │
│    Return to llama_context::decode() or process_ubatch()    │
└─────────────────────────────────────────────────────────────┘
```

### 5.5.5 Summary: Multi-Threading in CPU-Only llama.cpp

| Aspect | Details |
|--------|---------|
| **When distribution starts** | Immediately when `ggml_backend_graph_compute()` is called (both prefill and decode) |
| **What gets distributed** | Work WITHIN each operation (not operations themselves) |
| **How work is split** | Each thread processes different rows/chunks of the same operation |
| **Thread coordination** | Barrier synchronization between nodes, work partitioning within nodes |
| **Thread pools** | Separate pools for prefill (`threadpool_batch`) and decode (`threadpool`) |
| **Synchronization points** | After each graph node completes (barrier) |
| **Load balancing** | Work stealing for dynamic load balancing |

**Key Takeaways:**
1. **All threads process all nodes** - No node-to-thread assignment
2. **Work is split within operations** - Each thread processes a portion of each operation
3. **Barriers ensure correctness** - All threads must finish a node before starting the next
4. **Distribution starts at graph execution** - Not during graph building or allocation

### 5.5.7 Complete Summary: How Multi-Threading Works on Multi-Core CPU

**Based on our discussion, here's a comprehensive summary of how llama.cpp uses multiple threads on multi-core CPUs:**

#### 1. Thread Creation and Configuration

**Command-Line Control:**
- **`-t N`**: Sets number of threads for decode phase (autoregressive token generation)
- **`--threads-batch N`**: Sets number of threads for prefill phase (prompt processing)
- If `--threads-batch` is not specified, it defaults to the same value as `-t`
- **Example:** `-t 2` means 2 threads for both decode and prefill (unless `--threads-batch` is specified separately)

**Thread Pool Creation:**
- **Location:** `tools/main/main.cpp:170-195`
- Two thread pools can be created:
  - `threadpool`: For decode phase (autoregressive generation)
  - `threadpool_batch`: For prefill phase (if different from decode)
- If thread counts are identical, only one pool is created
- Threads are attached to the context via `llama_attach_threadpool()`

#### 2. Execution Model: All Threads Process All Nodes

**Critical Principle:**
- **All threads process ALL graph nodes sequentially**
- **No node-to-thread assignment** (unlike GPU where different nodes can run on different devices)
- **Work is split WITHIN each node**, not across nodes

**Visual Representation:**
```
Graph: 145 nodes (all assigned to CPU backend)

Execution Flow (with -t 2):
┌─────────────────────────────────────────────────────────┐
│ For each node (sequential):                             │
│                                                          │
│ Node 0: MUL_MAT (Q projection)                          │
│   ├─> Thread 0 (ith=0): Processes rows [0, 2, 4, ...]  │
│   ├─> Thread 1 (ith=1): Processes rows [1, 3, 5, ...]  │
│   └─> BARRIER: Wait for both threads                    │
│                                                          │
│ Node 1: MUL_MAT (K projection)                          │
│   ├─> Thread 0: Processes rows [0, 2, 4, ...]           │
│   ├─> Thread 1: Processes rows [1, 3, 5, ...]           │
│   └─> BARRIER: Wait for both threads                    │
│                                                          │
│ ... (continue for all 145 nodes)                        │
└─────────────────────────────────────────────────────────┘
```

#### 3. Work Distribution: Within-Operation Parallelization

**How Work is Split:**

1. **Thread Parameters:**
   - `ith`: Thread ID (0, 1, 2, ..., N-1)
   - `nth`: Total number of threads (from `-t`)

2. **Operation-Level Partitioning:**
   - Each operation (e.g., `MUL_MAT`, `ADD`, `RMS_NORM`) receives `ith` and `nth`
   - Operation splits its work into chunks
   - Each thread processes specific chunks based on its `ith`

3. **Example: Matrix Multiplication (MUL_MAT)**
   - **Input:** Matrix [batch, 288]
   - **Weight:** Matrix [288, 288]
   - **Output:** Matrix [batch, 288] (288 rows to compute)
   - **With 2 threads:**
     - Thread 0: Computes rows [0-143] (approximately)
     - Thread 1: Computes rows [144-287] (approximately)
   - **Both threads work simultaneously on the same operation**
   - **No data races:** Each thread writes to different memory regions

4. **Work Stealing:**
   - If one thread finishes early, it can grab additional chunks
   - Uses atomic operations for thread-safe chunk assignment
   - Improves load balancing across threads

**Code Location:** `ggml/src/ggml-cpu/ggml-cpu.c:1221` (`ggml_compute_forward_mul_mat()`)

#### 4. Thread Coordination: Barrier Synchronization

**Why Barriers Are Needed:**
- **Data dependencies:** Node N+1 may depend on Node N's output
- **Correctness:** All threads must finish Node N before starting Node N+1
- **Memory consistency:** Ensures all writes from Node N are visible to all threads

**Barrier Implementation:**
- **Location:** `ggml/src/ggml-cpu/ggml-cpu.c:543` (`ggml_barrier()`)
- **Mechanism:**
  - Uses atomic operations for synchronization
  - Last thread to arrive resets the barrier counter
  - Other threads spin-wait until barrier is released
  - **No OpenMP:** Uses custom barrier (OpenMP disabled for cross-compilation)

**Barrier Usage:**
- Called after **every graph node** completes
- **Location:** `ggml/src/ggml-cpu/ggml-cpu.c:2934` (inside `ggml_graph_compute_thread()`)
- All threads must reach the barrier before proceeding to the next node

#### 5. When Distribution Starts

**Timeline:**
1. **Graph Building:** Sequential (single thread)
   - Computation graph is constructed
   - All nodes are created
   - Dependencies are established

2. **Tensor Allocation:** Sequential (single thread)
   - Memory is allocated for all tensors
   - Backend assignment happens (all → CPU)

3. **Graph Execution:** **Multi-threaded distribution starts HERE**
   - **Entry point:** `ggml_backend_cpu_graph_compute()`
   - **Function:** `ggml_graph_compute_threaded()`
   - **Location:** `ggml/src/ggml-cpu/ggml-cpu.c` (around line 2900)
   - Threads are spawned/woken
   - All threads start processing nodes in parallel

**Applies to Both Phases:**
- **Prefill phase:** Uses `threadpool_batch` (or `threadpool` if same)
- **Decode phase:** Uses `threadpool`

#### 6. Thread Lifecycle

**Complete Lifecycle:**

```
1. THREAD SPAWNING
   ├─> Main thread (thread 0): Starts immediately
   └─> Worker threads (1, 2, ..., N-1): Woken from thread pool
   
2. THREAD EXECUTION
   ├─> All threads run ggml_graph_compute_thread() in parallel
   ├─> Each thread processes ALL nodes sequentially
   └─> Within each node, work is split based on ith/nth
   
3. SYNCHRONIZATION
   ├─> After each node: ggml_barrier() ensures all threads finish
   └─> Before next node: All threads wait at barrier
   
4. COMPLETION
   ├─> All threads finish processing all nodes
   └─> Main thread waits for workers via ggml_threadpool_wait()
```

#### 7. Performance Characteristics

**Speedup Factors:**
- **Ideal speedup:** Up to N× with N threads (for compute-bound operations)
- **Actual speedup:** Less than N× due to:
  - Barrier overhead (synchronization cost)
  - Memory bandwidth limitations
  - Cache contention (multiple threads competing for cache)
  - Load imbalance (some threads may finish earlier)

**Optimization Strategies:**
- **Work stealing:** Faster threads grab additional chunks
- **NUMA awareness:** Threads pinned to specific CPU cores
- **Chunk size tuning:** Adaptive chunk sizes based on matrix dimensions
- **Polling level:** Configurable busy-wait vs. sleep (via `--poll` parameter)

#### 8. Key Differences from GPU Execution

| Aspect | CPU Multi-Threading | GPU Execution |
|--------|---------------------|---------------|
| **Node assignment** | All threads process all nodes | Different nodes can run on different devices |
| **Work distribution** | Within each operation | Across operations (pipeline parallelism) |
| **Synchronization** | Barriers between nodes | Asynchronous execution with events |
| **Memory** | Shared memory (all threads access same tensors) | Separate device memory (requires copies) |
| **Parallelism** | Thread-level (within operation) | Operation-level (across operations) |

#### 9. Code Locations Summary

| Component | File | Line Range |
|-----------|------|------------|
| **Barrier implementation** | `ggml/src/ggml-cpu/ggml-cpu.c` | 543-568 |
| **Work partitioning (MUL_MAT)** | `ggml/src/ggml-cpu/ggml-cpu.c` | 1221-1410 |
| **Thread work function** | `ggml/src/ggml-cpu/ggml-cpu.c` | ~2900 |
| **Thread pool creation** | `tools/main/main.cpp` | 170-195 |
| **Graph execution entry** | `ggml/src/ggml-backend-cpu.cpp` | (varies) |

#### 10. Practical Example: `-t 2` Execution

**Your Command:**
```bash
llama-cli -m model.gguf -t 2 -ngl 0 ...
```

**What Happens:**
1. **2 threads** are created in the thread pool
2. **All 145 graph nodes** are assigned to CPU backend
3. **For each node:**
   - Thread 0 processes approximately half the work (rows 0, 2, 4, ...)
   - Thread 1 processes approximately half the work (rows 1, 3, 5, ...)
   - Both threads work simultaneously
   - Barrier synchronizes before next node
4. **Result:** Up to ~2× speedup (depending on operation type and memory bandwidth)

**Key Insight:**
- CPU core count is **detected** but **not directly used**
- What matters is the **thread count** you specify (`-t 2`)
- The system uses **2 threads** regardless of how many CPU cores are available
- Each thread may run on a different CPU core (if available), but that's handled by the OS scheduler

---

**Final Summary:**
Multi-threading in llama.cpp on multi-core CPUs works by having **all threads process all graph nodes sequentially**, but **splitting work within each operation** based on thread ID. **Barrier synchronization** ensures correctness between nodes, and **work stealing** improves load balancing. The distribution starts when graph execution begins, not during graph building or allocation.

### 5.5.6 Code Implementation Locations

**Where are the actual code implementations for the three coordination mechanisms?**

#### A. Barrier Synchronization Implementation

**File:** [`ggml/src/ggml-cpu/ggml-cpu.c:543`](../dev_env/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c#L543)

```c
void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed);
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (fill seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }
#endif
}
```

**Barrier Usage Location:**
- **File:** [`ggml/src/ggml-cpu/ggml-cpu.c:2934`](../dev_env/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c#L2934) (inside `ggml_graph_compute_thread()`)
- Called after each graph node completes to synchronize all threads

#### B. Work Partitioning Within Operations Implementation

**File:** [`ggml/src/ggml-cpu/ggml-cpu.c:1221`](../dev_env/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c#L1221) (`ggml_compute_forward_mul_mat()`)

**Key Code Sections:**

1. **Thread ID and Count Extraction** (lines 1230-1231):
```c
const int ith = params->ith;  // Thread ID: 0, 1, 2, ...
const int nth = params->nth;  // Total threads: 2 (if -t 2)
```

2. **Work Chunking** (lines 1352-1382):
```c
// Dimensions of result
const int64_t nr0 = ne0;      // First dimension
const int64_t nr1 = ne1 * ne2 * ne3;  // Rest of dimensions

// Select chunk size
int chunk_size = 16;
if (nr0 == 1 || nr1 == 1) {
    chunk_size = 64;
}

// Calculate number of chunks
int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

// Re-chunk for better thread utilization or NUMA
if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
    nchunk0 = nr0 > nr1 ? nth : 1;  // parallelize by src0 rows
    nchunk1 = nr0 > nr1 ? 1 : nth;  // parallelize by src1 rows
}

// Elements per chunk
const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;
```

3. **Work Stealing Loop** (lines 1385-1410):
```c
// Start with thread's own chunk
int current_chunk = ith;

while (current_chunk < nchunk0 * nchunk1) {
    const int64_t ith0 = current_chunk % nchunk0;
    const int64_t ith1 = current_chunk / nchunk0;

    const int64_t ir0_start = dr0 * ith0;
    const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

    const int64_t ir1_start = dr1 * ith1;
    const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

    // Process this chunk
    // ... (actual computation code) ...

    // Work stealing: Get next chunk (atomic increment)
    current_chunk = atomic_fetch_add(&params->threadpool->current_chunk, nth);
}
```

**Barrier Before Work Distribution** (line 1325):
```c
if (ith == 0) {
    // Initialize chunk counter
    atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
}
ggml_barrier(params->threadpool);  // Sync all threads before starting work
```

#### C. Thread Pool Management Implementation

**Note:** The thread pool management functions (`ggml_graph_compute_threaded`, `ggml_graph_compute_thread`, `ggml_threadpool_work`, `ggml_threadpool_wait`) are implemented in the GGML backend infrastructure. The exact file locations may vary depending on the GGML version, but they are typically in:

- **Thread spawning/coordination:** `ggml/src/ggml-cpu/ggml-cpu.c` (around line 2900-3000)
- **Thread work function:** `ggml/src/ggml-cpu/ggml-cpu.c:ggml_graph_compute_thread()` (around line 2900)

**Key Functions:**

1. **Thread Work Function** (`ggml_graph_compute_thread`):
   - **Location:** [`ggml/src/ggml-cpu/ggml-cpu.c`](../dev_env/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c) (around line 2900)
   - **Purpose:** Each thread executes this function, processing all graph nodes sequentially
   - **Key behavior:**
     - Sets thread affinity (NUMA-aware)
     - Iterates through all graph nodes
     - Calls `ggml_compute_forward()` for each node (work is split within)
     - Calls `ggml_barrier()` after each node

2. **Thread Pool Work/Wait Functions:**
   - **`ggml_threadpool_work()`**: Wakes/spawns worker threads
   - **`ggml_threadpool_wait()`**: Waits for all threads to complete
   - **Location:** These are part of the thread pool infrastructure, typically implemented in `ggml/src/ggml-cpu/ggml-cpu.c` or a separate threading module

**Summary of Code Locations:**

| Mechanism | Function | File | Line Range |
|-----------|----------|------|------------|
| **A. Barrier** | `ggml_barrier()` | `ggml/src/ggml-cpu/ggml-cpu.c` | 543-568 |
| **B. Work Partitioning** | `ggml_compute_forward_mul_mat()` | `ggml/src/ggml-cpu/ggml-cpu.c` | 1221-1410 |
| **B. Work Partitioning** | Thread ID extraction | `ggml/src/ggml-cpu/ggml-cpu.c` | 1230-1231 |
| **B. Work Partitioning** | Chunk calculation | `ggml/src/ggml-cpu/ggml-cpu.c` | 1352-1382 |
| **B. Work Partitioning** | Work stealing loop | `ggml/src/ggml-cpu/ggml-cpu.c` | 1385-1410 |
| **C. Thread Management** | `ggml_graph_compute_thread()` | `ggml/src/ggml-cpu/ggml-cpu.c` | ~2900 |
| **C. Thread Management** | Barrier usage | `ggml/src/ggml-cpu/ggml-cpu.c` | 2934 |

**How to Find These Functions:**

1. **Search in codebase:**
   ```bash
   # Find barrier implementation
   grep -r "void ggml_barrier" dev_env/llama.cpp/ggml/src/
   
   # Find MUL_MAT work partitioning
   grep -r "ggml_compute_forward_mul_mat" dev_env/llama.cpp/ggml/src/
   
   # Find thread work function
   grep -r "ggml_graph_compute_thread" dev_env/llama.cpp/ggml/src/
   ```

2. **Direct file access:**
   - All CPU backend code is in: `dev_env/llama.cpp/ggml/src/ggml-cpu/`
   - Main file: `ggml-cpu.c` (contains most of the threading and operation code)

### 5.5.8 Response: How is Work Distributed Across Cores?

**Question:** "How is the work distributed across cores?"

**Short Answer:**
Work is distributed **at the thread level**, not directly at the core level. llama.cpp creates threads (based on the `-t` parameter), and the **OS scheduler** distributes these threads across available CPU cores. Within each operation, work is split among threads based on thread ID (`ith`) and total threads (`nth`).

#### Detailed Explanation

**1. Thread-Based Distribution (Not Core-Based)**

**Key Point:** llama.cpp works with **threads**, not cores directly.

- **Thread Creation:** When you specify `-t 2`, llama.cpp creates **2 threads**
- **Core Assignment:** The **OS scheduler** decides which CPU core each thread runs on
- **No Direct Core Control:** llama.cpp doesn't explicitly assign threads to specific cores (unless using CPU affinity masks)

**2. Work Distribution Mechanism**

**Two-Level Distribution:**

**Level 1: Operation-Level Work Splitting**

For each operation (e.g., matrix multiplication), work is split among threads:

```
Operation: MUL_MAT (Matrix Multiplication)
Input: [batch, 288]
Weight: [288, 288]
Output: [288 rows to compute]

With -t 2 (2 threads):
├─> Thread 0 (ith=0): Computes rows [0, 2, 4, 6, ...]  (~144 rows)
└─> Thread 1 (ith=1): Computes rows [1, 3, 5, 7, ...]  (~144 rows)

Both threads work on the SAME operation simultaneously
Each thread writes to different memory regions (no data races)
```

**Code Location:** [`ggml/src/ggml-cpu/ggml-cpu.c:1221`](../dev_env/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c#L1221) (`ggml_compute_forward_mul_mat()`)

**How it works:**
- Each operation receives `ith` (thread ID: 0, 1, 2, ...) and `nth` (total threads)
- Operation divides work into chunks
- Each thread processes chunks starting from its `ith` value
- Work stealing: Faster threads can grab additional chunks

**Level 2: OS-Level Core Assignment**

**OS Scheduler Responsibility:**
- The OS scheduler distributes threads across available CPU cores
- Typically: Thread 0 → Core 0, Thread 1 → Core 1, etc. (if cores available)
- Can migrate threads between cores for load balancing

**Thread Affinity (Optional):**
- llama.cpp can set thread affinity (NUMA-aware) via `set_numa_thread_affinity()`
- This hints to the OS which core to use, but OS still makes final decision
- **Location:** `ggml/src/ggml-cpu/ggml-cpu.c` (inside `ggml_graph_compute_thread()`)

**3. Core Distribution Example**

**Scenario:** System with 8 CPU cores, running with `-t 2`

```
llama.cpp creates: 2 threads
├─> Thread 0 (ith=0)
└─> Thread 1 (ith=1)

OS scheduler distributes:
├─> Thread 0 → Core 0 (or Core 2, 4, 6 - OS decides)
└─> Thread 1 → Core 1 (or Core 3, 5, 7 - OS decides)

Work distribution:
├─> Thread 0: Processes ~50% of each operation
└─> Thread 1: Processes ~50% of each operation

Result:
- Only 2 cores are actively used (out of 8 available)
- Other 6 cores remain idle (unless other processes use them)
```

**Important:** If you want to use more cores, you need to increase the thread count (`-t 4` to use 4 cores, `-t 8` to use 8 cores, etc.)

**4. CPU Affinity Control (Advanced)**

**Optional CPU Mask:**
- llama.cpp supports `--cpu-mask` to specify which cores to use
- **Example:** `--cpu-mask 0x3` (binary: 0011) uses cores 0 and 1 only
- **Location:** [`common/common.h:64-71`](../dev_env/llama.cpp/common/common.h#L64) (`struct cpu_params`)

**NUMA Awareness:**
- `set_numa_thread_affinity(ith)` sets thread affinity based on NUMA topology
- Helps optimize memory access patterns on NUMA systems
- **Location:** `ggml/src/ggml-cpu/ggml-cpu.c` (inside `ggml_graph_compute_thread()`)

**5. Summary**

**How work is distributed across cores:**

1. **Application Level (llama.cpp):**
   - Creates N threads (based on `-t` parameter)
   - Splits work within each operation based on thread ID (`ith`)
   - Each thread processes different chunks/rows of the same operation

2. **OS Level (Kernel Scheduler):**
   - Distributes threads across available CPU cores
   - Typically assigns one thread per core (if cores available)
   - Can migrate threads for load balancing

3. **Result:**
   - Work is distributed **by thread**, not directly by core
   - Core assignment is handled by the OS scheduler
   - To use more cores, increase thread count (`-t N`)

**Key Takeaway:** llama.cpp controls **thread-level work distribution**, while the OS controls **core-level thread assignment**. The number of threads (`-t`) determines how many cores will be utilized (up to the number of available cores).

---

## Phase 6: Kernel Dispatch

### 6.1 Operation Dispatch

**File:** [`ggml/src/ggml.c:ggml_compute_forward()`](../dev_env/llama.cpp/ggml/src/ggml.c)

```cpp
void ggml_compute_forward(
        struct ggml_compute_params * params,
        struct ggml_tensor * tensor) {
    
    // Check if extra buffer type handles this (IMI, AMX, etc.)
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;  // IMI/AMX handled it
    }
    
    // Use standard CPU kernels
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            ggml_compute_forward_mul_mat(params, tensor);
            break;
        case GGML_OP_ADD:
            ggml_compute_forward_add(params, tensor);
            break;
        case GGML_OP_RMS_NORM:
            ggml_compute_forward_rms_norm(params, tensor);
            break;
        // ... other operations ...
        default:
            GGML_ABORT("unknown op");
    }
}
```

### 6.2 Extra Buffer Type Check (IMI/AMX)

**File:** [`ggml/src/ggml-cpu/traits.cpp:12`](../dev_env/llama.cpp/ggml/src/ggml-cpu/traits.cpp#L12)

```cpp
bool ggml_cpu_extra_compute_forward(
        struct ggml_compute_params * params,
        struct ggml_tensor * op) {
    
    // Iterate through all extra buffer types
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra = (ggml::cpu::extra_buffer_type *) extra->context;
            
            // Get tensor-specific traits (IMI kernel)
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;  // IMI/AMX handled it
            }
        }
    }
    
    return false;  // Not handled, use standard kernels
}
```

**Decision Flow:**
```
Operation: MUL_MAT
    │
    ├─> Is src[0]->buffer->buft == CPU_IMI?  ──No──> Use standard CPU kernel
    │   └─> Yes
    │
    ├─> Does src[0]->extra contain IMI kernel?  ──No──> Use standard CPU
    │   └─> Yes
    │
    └─> Call IMI kernel: ggml_imi_gemm_q4_0_q8_0_8x4(...)
```

### 6.3 Standard CPU Kernel (MUL_MAT Example)

**File:** [`ggml/src/ggml-cpu/ggml-cpu.c:1221`](../dev_env/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c#L1221)

```cpp
void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    
    const int ith = params->ith;  // Thread ID
    const int nth = params->nth;   // Total threads
    
    const struct ggml_tensor * src0 = dst->src[0];  // Weight matrix
    const struct ggml_tensor * src1 = dst->src[1];   // Input matrix
    
    const int64_t ne0 = src0->ne[0];  // Output dimension
    const int64_t ne1 = src1->ne[1];  // Number of tokens
    
    // Divide work among threads
    const int64_t nr = ne1;  // Rows to process
    const int64_t dr = (nr + nth - 1) / nth;  // Rows per thread
    const int64_t ir0 = dr * ith;             // Start row
    const int64_t ir1 = MIN(ir0 + dr, nr);   // End row
    
    // Process assigned rows
    for (int64_t ir = ir0; ir < ir1; ir++) {
        // Matrix-vector multiplication (decode) or matrix-matrix (prefill)
        if (ne1 == 1) {
            // Decode: GEMV
            ggml_vec_dot_q4_0_q8_0(
                ne0,
                (float *) ((char *) dst->data + ir * dst->nb[1]),
                (void *) src0->data,
                (float *) ((char *) src1->data + ir * src1->nb[1]));
        } else {
            // Prefill: GEMM
            ggml_compute_forward_mul_mat_f32(
                params, dst, ir, ir1);
        }
    }
}
```

**Thread Work Distribution:**
- **Thread 0**: Processes rows 0 to (dr-1)
- **Thread 1**: Processes rows dr to (2*dr-1)
- **Thread 2**: Processes rows (2*dr) to (3*dr-1)
- **Thread 3**: Processes rows (3*dr) to (4*dr-1)

### 6.4 IMI Kernel Execution

**File:** [`ggml/src/ggml-cpu/imi/opt-kernels.cpp`](../dev_env/llama.cpp/ggml/src/ggml-cpu/imi/opt-kernels.cpp)

```cpp
// IMI-optimized kernel for Q4_0 × F32 → F32
void ggml_imi_gemm_q4_0_q8_0_8x4(
        const int n,
        const int k,
        const float * src1,      // Input [k, n]
        const block_q4_0 * src0, // Weight [n, k] (Q4_0, repacked)
        float * dst) {            // Output [n]
    
    // Use RISC-V Vector extensions
    size_t vl;
    
    // Process in blocks of 8 rows
    for (int i = 0; i < n; i += 8) {
        // Load 8 rows of weights
        vfloat32m1_t acc0 = vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc1 = vfmv_v_f_f32m1(0.0f, vl);
        // ... acc2-acc7 ...
        
        // Dot product with input vector
        for (int j = 0; j < k; j += vl) {
            // Load input vector
            vfloat32m1_t v_in = vle32_v_f32m1(&src1[j], vl);
            
            // Load and dequantize weights
            // ... IMI-specific dequantization ...
            
            // Fused multiply-add
            acc0 = vfmacc_vv_f32m1(acc0, v_w0, v_in, vl);
            acc1 = vfmacc_vv_f32m1(acc1, v_w1, v_in, vl);
            // ... acc2-acc7 ...
        }
        
        // Store results
        vse32_v_f32m1(&dst[i], acc0, vl);
        vse32_v_f32m1(&dst[i+1], acc1, vl);
        // ... store acc2-acc7 ...
    }
}
```

**IMI Optimizations:**
- **Repacked weights**: Optimized memory layout
- **Vector intrinsics**: RISC-V Vector (RVV) extensions
- **Fused operations**: Combine dequantization and multiply-add
- **Block processing**: Process 8 rows at a time

---

## Phase 7: Major Model Inference Computations and Data Structures

This section details how major model inference computations work with their typical data structures (K, Q, V matrices, softmax, RMSNorm, SiLU, etc.) and temporal data structures (KV cache) in a multi-threaded environment.

### 7.1 Overview: Transformer Layer Computation Flow

**Complete Transformer Layer (Per Layer):**

```
Input: hidden [n_tokens, n_embd]
    ↓
[1] Pre-Attention RMSNorm
    ↓
[2] QKV Projections (3 separate matrix multiplications)
    ├─> Q [n_tokens, n_q * head_dim]
    ├─> K [n_tokens, n_kv * head_dim]
    └─> V [n_tokens, n_kv * head_dim]
    ↓
[3] RoPE (Rotary Position Embedding) - Applied to Q and K
    ↓
[4] Attention Computation
    ├─> Concatenate with KV Cache
    ├─> Compute QK^T scores
    ├─> Scale by sqrt(head_dim)
    ├─> Apply causal mask
    ├─> Softmax
    └─> Multiply with V
    ↓
[5] Output Projection
    ↓
[6] Residual Connection (hidden = hidden + attn_out)
    ↓
[7] Pre-FFN RMSNorm
    ↓
[8] Feed-Forward Network (FFN)
    ├─> Gate projection
    ├─> Up projection
    ├─> SiLU activation
    ├─> Element-wise multiply (gate * up)
    └─> Down projection
    ↓
[9] Residual Connection (hidden = hidden + ffn_out)
    ↓
Output: hidden [n_tokens, n_embd]
```

**File:** [`src/llama-graph.cpp`](../dev_env/llama.cpp/src/llama-graph.cpp) (graph building logic)

### 7.2 Q, K, V Matrix Computation

#### 7.2.1 QKV Projections

**Purpose:** Transform hidden states into Query, Key, and Value representations for attention.

**File:** [`src/llama-graph.cpp`](../dev_env/llama.cpp/src/llama-graph.cpp) (around line 1100-1200)

**Computation:**
```cpp
// Input: hidden [n_tokens, n_embd]
// After RMSNorm: hidden_norm [n_tokens, n_embd]

// Q projection
auto q = ggml_mul_mat(ctx0, model.layers[i].attn_q, hidden_norm);
// q shape: [n_tokens, n_q * head_dim]
// Weight: attn_q [n_embd, n_q * head_dim]

// K projection
auto k = ggml_mul_mat(ctx0, model.layers[i].attn_k, hidden_norm);
// k shape: [n_tokens, n_kv * head_dim]
// Weight: attn_k [n_embd, n_kv * head_dim]

// V projection
auto v = ggml_mul_mat(ctx0, model.layers[i].attn_v, hidden_norm);
// v shape: [n_tokens, n_kv * head_dim]
// Weight: attn_v [n_embd, n_kv * head_dim]
```

**Data Structures:**
- **Q, K, V tensors:** Stored as `ggml_tensor` structures
- **Memory layout:** Row-major (C-style)
- **Data type:** Typically F32 or quantized (Q4_0, Q8_0, etc.)

**Multi-Threading:**
- **Operation:** `ggml_mul_mat` (matrix multiplication)
- **Work splitting:** Each thread processes different rows of the output matrix
- **Example with `-t 2`:**
  - Thread 0: Computes rows [0, 2, 4, ...] of Q, K, V
  - Thread 1: Computes rows [1, 3, 5, ...] of Q, K, V
- **Code:** [`ggml/src/ggml-cpu/ggml-cpu.c:1221`](../dev_env/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c#L1221) (`ggml_compute_forward_mul_mat()`)

#### 7.2.2 Reshape and Permute for Multi-Head Attention

**Purpose:** Reorganize Q, K, V into multi-head format.

```cpp
// Reshape Q: [n_tokens, n_q * head_dim] → [head_dim, n_q, n_tokens]
q = ggml_reshape_3d(ctx0, q, head_dim, n_q, n_tokens);

// Permute: [head_dim, n_q, n_tokens] → [n_q, n_tokens, head_dim]
q = ggml_permute(ctx0, q, 0, 2, 1, 3);

// Similar for K and V
k = ggml_reshape_3d(ctx0, k, head_dim, n_kv, n_tokens);
k = ggml_permute(ctx0, k, 0, 2, 1, 3);

v = ggml_reshape_3d(ctx0, v, head_dim, n_kv, n_tokens);
v = ggml_permute(ctx0, v, 0, 2, 1, 3);
```

**Final Shapes:**
- **Q:** `[n_q, n_tokens, head_dim]` (e.g., [6, 1, 48])
- **K:** `[n_kv, n_tokens, head_dim]` (e.g., [2, 1, 48])
- **V:** `[n_kv, n_tokens, head_dim]` (e.g., [2, 1, 48])

**Multi-Threading:**
- Reshape and permute operations are typically **not parallelized** (simple memory reordering)
- They run sequentially on a single thread

#### 7.2.3 RoPE (Rotary Position Embedding)

**Purpose:** Apply rotary position embeddings to Q and K before attention computation.

**File:** [`src/llama-graph.cpp`](../dev_env/llama.cpp/src/llama-graph.cpp)

```cpp
// Apply RoPE to Q and K
q = ggml_rope(ctx0, q, ubatch.pos, n_ctx, freq_base, freq_scale);
k = ggml_rope(ctx0, k, ubatch.pos, n_ctx, freq_base, freq_scale);
```

**How RoPE Works:**
- Rotates Q and K vectors based on their position in the sequence
- Uses complex number rotation: `(x, y) → (x*cos - y*sin, x*sin + y*cos)`
- Different frequencies for different dimensions
- **No addition:** Unlike learned position embeddings, RoPE rotates existing vectors

**Multi-Threading:**
- RoPE computation can be parallelized across tokens
- Each thread processes different token positions
- **Code:** `ggml/src/ggml-cpu/ggml-cpu.c` (rope implementation)

### 7.3 Attention Mechanism with KV Cache

#### 7.3.1 KV Cache: Temporal Data Structure

**Purpose:** Store previously computed K and V values to avoid recomputation during autoregressive generation.

**Data Structure:**
```cpp
// KV Cache per layer
struct llama_kv_cache_layer {
    ggml_tensor * k;  // [n_kv, n_ctx, head_dim] - Cached keys
    ggml_tensor * v;  // [n_kv, n_ctx, head_dim] - Cached values
};

// Full KV Cache
struct llama_kv_cache {
    llama_kv_cache_layer layers[n_layer];  // One cache per layer
    int n_pos;  // Current position in cache
};
```

**Memory Layout:**
- **Per layer:** `n_kv * n_ctx * head_dim * sizeof(float)` bytes
- **Total:** `n_layer * n_kv * n_ctx * head_dim * sizeof(float)` bytes
- **Example:** 6 layers, 2 KV heads, 512 context, 48 head_dim = ~1.2 MB (F32)

**File:** [`src/llama-kv-cache.h`](../dev_env/llama.cpp/src/llama-kv-cache.h)

#### 7.3.2 Concatenating with KV Cache

**During Inference:**
```cpp
// Retrieve cached K, V from previous tokens
auto k_cache = memory->get_k(i);  // [n_kv, n_prev, head_dim]
auto v_cache = memory->get_v(i);  // [n_kv, n_prev, head_dim]

// Current K, V for new tokens
auto k_new = k;  // [n_kv, n_tokens, head_dim]
auto v_new = v;  // [n_kv, n_tokens, head_dim]

// Concatenate along sequence dimension
auto k_all = ggml_concat(ctx0, k_cache, k_new, /*dim=*/1);
// k_all shape: [n_kv, n_prev + n_tokens, head_dim]

auto v_all = ggml_concat(ctx0, v_cache, v_new, /*dim=*/1);
// v_all shape: [n_kv, n_prev + n_tokens, head_dim]
```

**Multi-Threading:**
- Concatenation is typically **not parallelized** (simple memory copy)
- KV cache update happens **after** attention computation (see below)

#### 7.3.3 Attention Score Computation

**Step 1: Compute QK^T**

```cpp
// QK^T: [n_q, n_tokens, head_dim] × [n_kv, n_prev + n_tokens, head_dim]^T
// Result: [n_q, n_tokens, n_prev + n_tokens]
auto scores = ggml_mul_mat(ctx0, q, k_all);
```

**Step 2: Scale by sqrt(head_dim)**

```cpp
// Scale scores for numerical stability
float scale = 1.0f / sqrtf(head_dim);
scores = ggml_scale(ctx0, scores, scale);
```

**Step 3: Apply Causal Mask**

```cpp
// Causal mask: -inf for future positions, 0 for past/current
// Mask shape: [n_tokens, n_prev + n_tokens]
auto mask = ggml_causal_mask(ctx0, n_tokens, n_prev + n_tokens);
scores = ggml_add(ctx0, scores, mask);
```

**Mask Pattern Example** (n_tokens=1, n_prev=2):
```
Position:  0  1  2 | 3
Token 3:   0  0  0 | 0   (can see positions 0,1,2,3)
Token 4:   0  0  0 | 0   (hypothetical - would see 0,1,2,3,4)
```

**Multi-Threading:**
- **QK^T computation:** Parallelized across Q heads and tokens
- **Scaling and masking:** Simple element-wise operations (can be parallelized)
- **Code:** `ggml/src/ggml-cpu/ggml-cpu.c` (attention implementations)

#### 7.3.4 Softmax Operation

**Purpose:** Convert attention scores to probability distributions.

**Mathematical Formula:**
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

**Implementation:**
```cpp
// Softmax: [n_q, n_tokens, n_prev + n_tokens]
scores = ggml_soft_max(ctx0, scores);
```

**Softmax Algorithm (for numerical stability):**
1. Find maximum value in each row: `max = max(scores[i, :])`
2. Subtract maximum: `scores[i, j] = scores[i, j] - max`
3. Compute exponentials: `exp_scores = exp(scores)`
4. Sum exponentials: `sum = sum(exp_scores)`
5. Normalize: `softmax = exp_scores / sum`

**Multi-Threading:**
- **Parallelization:** Each thread processes different rows (query positions)
- **With `-t 2`:**
  - Thread 0: Processes softmax for rows [0, 2, 4, ...]
  - Thread 1: Processes softmax for rows [1, 3, 5, ...]
- **Synchronization:** Each row's softmax is independent (no barriers needed within softmax)
- **Code:** `ggml/src/ggml-cpu/ggml-cpu.c` (softmax implementation)

**Flash Attention (Memory-Efficient):**
- Uses **online softmax** algorithm
- Processes Q rows in chunks
- Accumulates V*exp(scores) incrementally
- **Memory:** O(n) instead of O(n²)
- **Code:** `ggml/src/ggml-cpu/ops.cpp` (flash attention implementation)

#### 7.3.5 Attention Output Computation

**Step 1: Multiply Scores with V**

```cpp
// scores: [n_q, n_tokens, n_prev + n_tokens]
// v_all: [n_kv, n_prev + n_tokens, head_dim]
// Result: [n_q, n_tokens, head_dim]
auto attn_out = ggml_mul_mat(ctx0, scores, v_all);
```

**Step 2: Reshape and Permute Back**

```cpp
// Reshape: [n_q, n_tokens, head_dim] → [head_dim, n_q, n_tokens]
attn_out = ggml_permute(ctx0, attn_out, 0, 2, 1, 3);

// Reshape: [head_dim, n_q, n_tokens] → [n_tokens, n_q * head_dim]
attn_out = ggml_reshape_2d(ctx0, attn_out, n_q * head_dim, n_tokens);
```

**Step 3: Output Projection**

```cpp
// Output projection: [n_tokens, n_q * head_dim] × [n_q * head_dim, n_embd]
attn_out = ggml_mul_mat(ctx0, model.layers[i].attn_o, attn_out);
// attn_out shape: [n_tokens, n_embd]
```

**Multi-Threading:**
- **Matrix multiplication:** Parallelized across output rows
- **Reshape/permute:** Sequential (memory reordering)

#### 7.3.6 KV Cache Update

**After Attention Computation:**

```cpp
// Update KV cache with new K, V values
memory->set_k(i, k_new);  // Store new K values
memory->set_v(i, v_new);  // Store new V values

// Increment position counter
memory->n_pos += n_tokens;
```

**Multi-Threading:**
- **KV cache update:** Typically **sequential** (single thread)
- **Reason:** Cache is shared across threads, requires synchronization
- **Location:** `src/llama-memory-*.cpp` (memory management)

### 7.4 RMSNorm (Root Mean Square Normalization)

**Purpose:** Normalize hidden states before attention and FFN.

**Mathematical Formula:**
```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

**Implementation:**
```cpp
// Pre-attention normalization
hidden_norm = ggml_rms_norm(ctx0, hidden, model.layers[i].attn_norm);
// hidden_norm shape: [n_tokens, n_embd]

// Pre-FFN normalization
hidden_norm = ggml_rms_norm(ctx0, hidden, model.layers[i].ffn_norm);
```

**Computation Steps:**
1. Compute mean of squares: `mean_sq = mean(x²)`
2. Compute RMS: `rms = sqrt(mean_sq + eps)`
3. Normalize: `x_norm = x / rms`
4. Scale: `output = x_norm * weight`

**Multi-Threading:**
- **Parallelization:** Each thread processes different rows (tokens)
- **With `-t 2`:**
  - Thread 0: Normalizes rows [0, 2, 4, ...]
  - Thread 1: Normalizes rows [1, 3, 5, ...]
- **Synchronization:** Each row's normalization is independent
- **Code:** `ggml/src/ggml-cpu/ggml-cpu.c` (RMSNorm implementation)

### 7.5 SiLU (Sigmoid Linear Unit) Activation

**Purpose:** Activation function used in feed-forward networks.

**Mathematical Formula:**
```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

**Implementation:**
```cpp
// Apply SiLU to gate projection
gate = ggml_silu(ctx0, gate);
// gate shape: [n_tokens, ffn_dim]

// Element-wise multiply with up projection
ffn_out = ggml_mul(ctx0, gate, up);
```

**Computation Steps:**
1. Compute sigmoid: `s = 1 / (1 + exp(-x))`
2. Multiply: `output = x * s`

**Multi-Threading:**
- **Parallelization:** Each thread processes different elements
- **Element-wise operation:** Highly parallelizable
- **With `-t 2`:**
  - Thread 0: Processes elements [0, 2, 4, ...]
  - Thread 1: Processes elements [1, 3, 5, ...]
- **Code:** `ggml/src/ggml-cpu/ggml-cpu.c` (SiLU implementation)

### 7.6 Feed-Forward Network (FFN)

**Complete FFN Computation:**

```cpp
// Step 1: Gate projection
auto gate = ggml_mul_mat(ctx0, model.layers[i].ffn_gate, hidden_norm);
// gate shape: [n_tokens, ffn_dim]

// Step 2: Up projection
auto up = ggml_mul_mat(ctx0, model.layers[i].ffn_up, hidden_norm);
// up shape: [n_tokens, ffn_dim]

// Step 3: SiLU activation
gate = ggml_silu(ctx0, gate);

// Step 4: Element-wise multiply (SwiGLU)
auto ffn_out = ggml_mul(ctx0, gate, up);
// ffn_out shape: [n_tokens, ffn_dim]

// Step 5: Down projection
ffn_out = ggml_mul_mat(ctx0, model.layers[i].ffn_down, ffn_out);
// ffn_out shape: [n_tokens, n_embd]
```

**Multi-Threading:**
- **Matrix multiplications:** Parallelized across output rows
- **SiLU and element-wise multiply:** Parallelized across elements
- **All operations:** Use same thread distribution pattern

### 7.7 Residual Connections

**Purpose:** Add input to output (skip connections) for gradient flow.

**Implementation:**
```cpp
// Attention residual
hidden = ggml_add(ctx0, hidden, attn_out);
// hidden shape: [n_tokens, n_embd]

// FFN residual
hidden = ggml_add(ctx0, hidden, ffn_out);
```

**Multi-Threading:**
- **Element-wise addition:** Highly parallelizable
- **Each thread:** Processes different elements independently
- **No synchronization needed:** No data dependencies

### 7.8 Temporal Data Structures in Multi-Threaded Environment

#### 7.8.1 KV Cache Access Patterns

**Read Pattern (During Attention):**
```
Thread 0: Reads K_cache[0:n_prev], V_cache[0:n_prev]
Thread 1: Reads K_cache[0:n_prev], V_cache[0:n_prev]
...
All threads read the SAME cached values (read-only)
```

**Write Pattern (After Attention):**
```
Thread 0: Writes K_new[0], V_new[0] to cache
Thread 1: Writes K_new[1], V_new[1] to cache
...
Each thread writes DIFFERENT positions (no conflicts)
```

**Synchronization:**
- **Read access:** No synchronization needed (read-only, shared)
- **Write access:** Typically sequential (single thread updates cache)
- **Barrier:** All threads must finish attention before cache update

#### 7.8.2 Intermediate Tensor Lifecycle

**Temporal Characteristics:**

1. **Short-lived tensors:**
   - Q, K, V projections: Created → Used → Discarded
   - Attention scores: Created → Softmax → Discarded
   - Gate/Up projections: Created → SiLU → Discarded

2. **Persistent tensors:**
   - Hidden states: Updated each layer (residual connections)
   - KV cache: Grows during inference (appended, not replaced)

3. **Memory Management:**
   - **Allocation:** Sequential (single thread, during graph building)
   - **Computation:** Parallel (multiple threads, during graph execution)
   - **Deallocation:** Sequential (after graph execution completes)

#### 7.8.3 Thread Safety Considerations

**Safe Operations (No Synchronization Needed):**
- **Read-only access:** Multiple threads can read same tensor simultaneously
- **Independent writes:** Each thread writes to different memory regions
- **Element-wise operations:** No cross-element dependencies

**Operations Requiring Synchronization:**
- **KV cache updates:** Single thread updates (or requires mutex)
- **Reduction operations:** Sum, mean, max across elements (requires barriers)
- **Shared accumulators:** Flash attention online softmax (atomic operations)

**Barrier Points:**
- **After each graph node:** `ggml_barrier()` ensures all threads finish before next node
- **Before KV cache update:** All threads must complete attention computation
- **Before next layer:** All threads must finish current layer

### 7.9 Complete Multi-Threaded Attention Flow (Example: `-t 2`)

**Step-by-Step with Thread Distribution:**

```
1. QKV Projections (Parallel)
   ├─> Thread 0: Computes Q[0], K[0], V[0]
   └─> Thread 1: Computes Q[1], K[1], V[1] (if n_tokens > 1)
   └─> BARRIER: Wait for all threads

2. RoPE (Parallel)
   ├─> Thread 0: Applies RoPE to Q[0], K[0]
   └─> Thread 1: Applies RoPE to Q[1], K[1]
   └─> BARRIER: Wait for all threads

3. Concatenate with KV Cache (Sequential)
   └─> Single thread: Concatenates k_cache + k_new, v_cache + v_new

4. QK^T Computation (Parallel)
   ├─> Thread 0: Computes scores[0, :] (row 0 of attention matrix)
   └─> Thread 1: Computes scores[1, :] (row 1, if n_tokens > 1)
   └─> BARRIER: Wait for all threads

5. Scale and Mask (Parallel)
   ├─> Thread 0: Scales and masks scores[0, :]
   └─> Thread 1: Scales and masks scores[1, :]
   └─> BARRIER: Wait for all threads

6. Softmax (Parallel)
   ├─> Thread 0: Computes softmax(scores[0, :])
   └─> Thread 1: Computes softmax(scores[1, :])
   └─> BARRIER: Wait for all threads

7. Attention Output (Parallel)
   ├─> Thread 0: Computes attn_out[0, :]
   └─> Thread 1: Computes attn_out[1, :]
   └─> BARRIER: Wait for all threads

8. KV Cache Update (Sequential)
   └─> Single thread: Updates cache with new K, V values
```

### 7.10 Summary: Data Structures and Multi-Threading

| Operation | Data Structure | Multi-Threading | Synchronization |
|-----------|---------------|-----------------|-----------------|
| **QKV Projections** | `[n_tokens, n_embd]` → `[n_tokens, head_dim]` | Parallel (rows) | Barrier after |
| **RoPE** | `[n_q, n_tokens, head_dim]` | Parallel (tokens) | Barrier after |
| **KV Cache** | `[n_kv, n_ctx, head_dim]` | Read: shared, Write: sequential | Barrier before write |
| **QK^T** | `[n_q, n_tokens, n_prev+n_tokens]` | Parallel (rows) | Barrier after |
| **Softmax** | `[n_q, n_tokens, n_prev+n_tokens]` | Parallel (rows) | Barrier after |
| **Attention Output** | `[n_q, n_tokens, head_dim]` | Parallel (rows) | Barrier after |
| **RMSNorm** | `[n_tokens, n_embd]` | Parallel (rows) | Barrier after |
| **SiLU** | `[n_tokens, ffn_dim]` | Parallel (elements) | Barrier after |
| **FFN Projections** | `[n_tokens, ffn_dim]` | Parallel (rows) | Barrier after |
| **Residual Add** | `[n_tokens, n_embd]` | Parallel (elements) | Barrier after |

**Key Insights:**
1. **Most operations are parallelized** across rows or elements
2. **Barriers ensure correctness** between dependent operations
3. **KV cache is shared** (read by all, written by one)
4. **Temporal data structures** (cache) persist across inference steps
5. **Intermediate tensors** are short-lived (created and discarded per step)

### 7.11 Data Sharing and Synchronization in Multi-Threaded Computation

**Question:** During multi-threaded computation, do threads share any data, or do they only need synchronization?

**Answer:** Threads **share most data structures** (model weights, input tensors, KV cache), but **write to different memory regions** of output tensors. They need **barrier synchronization** between operations, but **no locks/mutexes** for most operations because threads work on separate memory regions.

#### 7.11.1 Shared Data Structures (Read-Only or Read-Mostly)

**1. Model Weights (Read-Only, Shared by All Threads)**

```cpp
// Model weights are loaded once and shared by all threads
struct llama_model {
    ggml_tensor * layers[0].attn_q;  // [n_embd, n_q * head_dim]
    ggml_tensor * layers[0].attn_k;  // [n_embd, n_kv * head_dim]
    ggml_tensor * layers[0].attn_v;  // [n_embd, n_kv * head_dim]
    // ... all other weights
};
```

**Access Pattern:**
- **All threads read the same weights** simultaneously
- **No synchronization needed:** Read-only access is safe
- **Memory location:** Single copy in CPU memory, shared by all threads

**Example (MUL_MAT with `-t 2`):**
```
Operation: Q = hidden × attn_q
Weight: attn_q [n_embd, n_q * head_dim]  ← SHARED by all threads

Thread 0: Reads attn_q, computes Q[0, :] = hidden[0, :] × attn_q
Thread 1: Reads attn_q, computes Q[1, :] = hidden[1, :] × attn_q

Both threads read the SAME weight matrix simultaneously
No conflicts because it's read-only
```

**2. Input Tensors (Read-Only, Shared by All Threads)**

```cpp
// Input tensors are shared
ggml_tensor * hidden;  // [n_tokens, n_embd] - Input to layer
ggml_tensor * k_cache; // [n_kv, n_ctx, head_dim] - KV cache
```

**Access Pattern:**
- **All threads read the same input tensors**
- **Thread 0** reads `hidden[0, :]` to compute its portion
- **Thread 1** reads `hidden[1, :]` to compute its portion
- **KV cache:** All threads read the same cached K, V values

**Example:**
```
Input: hidden [2, 288]  ← SHARED by all threads

Thread 0: Reads hidden[0, :] (row 0)
Thread 1: Reads hidden[1, :] (row 1)

Both threads can read simultaneously (read-only)
```

**3. KV Cache (Read by All, Written by One)**

```cpp
// KV cache structure
struct llama_kv_cache {
    ggml_tensor * k;  // [n_kv, n_ctx, head_dim]
    ggml_tensor * v;  // [n_kv, n_ctx, head_dim]
};
```

**Access Pattern:**
- **Read access:** All threads read the same KV cache during attention
- **Write access:** Single thread updates cache after attention completes
- **Synchronization:** Barrier ensures all threads finish reading before write

**Example:**
```
KV Cache: k_cache [2, 512, 48]  ← SHARED

During Attention (Read):
├─> Thread 0: Reads k_cache[:, 0:512, :] (all cached keys)
└─> Thread 1: Reads k_cache[:, 0:512, :] (all cached keys)
└─> Both threads read simultaneously (read-only, safe)

After Attention (Write):
└─> Single thread: Writes k_new to k_cache[:, 512, :] (append new keys)
└─> Barrier ensures all reads complete before write
```

#### 7.11.2 Partitioned Data Structures (Each Thread Writes to Different Regions)

**1. Output Tensors (Partitioned by Threads)**

**Key Principle:** Each thread writes to **different memory regions** of the same output tensor.

```cpp
// Output tensor: Shared structure, partitioned memory regions
ggml_tensor * q;  // [n_tokens, n_q * head_dim] - Output tensor

// Memory layout:
// q->data points to shared memory buffer
// Thread 0 writes to: q->data[0 * row_size : 1 * row_size]
// Thread 1 writes to: q->data[1 * row_size : 2 * row_size]
```

**Example: Matrix Multiplication (MUL_MAT)**

```cpp
// Operation: Q = hidden × attn_q
// Output: Q [2, 288]  ← SHARED tensor, but partitioned writes

Thread 0 (ith=0):
├─> Reads: hidden[0, :] (row 0 of input)
├─> Reads: attn_q (entire weight matrix)
└─> Writes: Q[0, :] (row 0 of output)  ← Different memory region

Thread 1 (ith=1):
├─> Reads: hidden[1, :] (row 1 of input)
├─> Reads: attn_q (entire weight matrix)
└─> Writes: Q[1, :] (row 1 of output)  ← Different memory region

No data races: Threads write to different rows
No synchronization needed during computation
```

**Memory Layout Visualization:**
```
Output Tensor Q [2, 288]:
┌─────────────────────────────────────┐
│ Row 0: [288 elements]              │ ← Thread 0 writes here
├─────────────────────────────────────┤
│ Row 1: [288 elements]              │ ← Thread 1 writes here
└─────────────────────────────────────┘

Memory addresses:
Q->data[0 * 288 * sizeof(float) : 1 * 288 * sizeof(float)]  ← Thread 0
Q->data[1 * 288 * sizeof(float) : 2 * 288 * sizeof(float)]  ← Thread 1
```

**2. Intermediate Tensors (Partitioned Writes)**

**All intermediate tensors follow the same pattern:**
- **Attention scores:** Each thread computes different rows
- **Softmax output:** Each thread processes different rows
- **FFN outputs:** Each thread computes different rows

**Example: Softmax Operation**

```cpp
// Input: scores [2, 512]  ← SHARED (read by all)
// Output: scores_softmax [2, 512]  ← SHARED (partitioned writes)

Thread 0:
├─> Reads: scores[0, :] (row 0)
├─> Computes: softmax(scores[0, :])
└─> Writes: scores_softmax[0, :]  ← Row 0 only

Thread 1:
├─> Reads: scores[1, :] (row 1)
├─> Computes: softmax(scores[1, :])
└─> Writes: scores_softmax[1, :]  ← Row 1 only

No conflicts: Each thread writes to different rows
```

#### 7.11.3 Synchronization Mechanisms

**1. Barrier Synchronization (Primary Mechanism)**

**Purpose:** Ensure all threads finish one operation before starting the next.

**When Used:**
- **After each graph node** completes
- **Before dependent operations** start

**Why Needed:**
- **Data dependencies:** Node N+1 may read Node N's output
- **Memory consistency:** Ensure all writes are visible to all threads
- **Correctness:** Prevent race conditions

**Implementation:**
```cpp
// After each node computation
ggml_compute_forward(&params, node);

// BARRIER: All threads must reach here before continuing
ggml_barrier(tp);

// Next node can now safely read previous node's output
ggml_compute_forward(&params, next_node);
```

**Example:**
```
Node 0: Q = hidden × attn_q
├─> Thread 0: Writes Q[0, :]
├─> Thread 1: Writes Q[1, :]
└─> BARRIER: Wait for both threads

Node 1: K = hidden × attn_k
├─> Thread 0: Can now safely read Q[0, :] and Q[1, :]
├─> Thread 1: Can now safely read Q[0, :] and Q[1, :]
└─> Both threads see complete Q matrix
```

**2. Atomic Operations (For Work Stealing)**

**Purpose:** Thread-safe chunk assignment for dynamic load balancing.

**When Used:**
- **Work stealing:** Faster threads grab additional chunks
- **Chunk counter:** Atomic increment for chunk assignment

**Implementation:**
```cpp
// Initialize chunk counter (thread 0 only)
if (ith == 0) {
    atomic_store_explicit(&tp->current_chunk, nth, memory_order_relaxed);
}

// Work stealing loop
int current_chunk = ith;  // Start with own chunk
while (current_chunk < nchunk0 * nchunk1) {
    // Process chunk...
    
    // Get next chunk (atomic increment)
    current_chunk = atomic_fetch_add(&tp->current_chunk, nth);
}
```

**Why Atomic:**
- Multiple threads may try to grab the same chunk
- Atomic operations ensure only one thread gets each chunk
- No locks needed: Atomic operations are lock-free

**3. Sequential Operations (No Parallelization)**

**Some operations run on a single thread:**

- **KV cache concatenation:** Single thread concatenates k_cache + k_new
- **KV cache update:** Single thread writes new K, V to cache
- **Graph building:** Sequential (before multi-threading starts)
- **Tensor allocation:** Sequential (before multi-threading starts)

**Why Sequential:**
- **Complexity:** Operation is too simple to benefit from parallelization
- **Synchronization overhead:** Would require more synchronization than benefit
- **Data dependencies:** Operation depends on all threads completing previous step

#### 7.11.4 What is NOT Shared

**1. Thread-Local Variables**

Each thread has its own:
- **Thread ID (`ith`):** Unique to each thread
- **Local accumulators:** For operations like softmax
- **Stack variables:** Function-local variables

**Example:**
```cpp
// Inside ggml_compute_forward_mul_mat()
const int ith = params->ith;  // Thread-local (different for each thread)
const int nth = params->nth;  // Shared (same for all threads)

// Local variables (each thread has its own copy)
int current_chunk = ith;  // Thread 0: 0, Thread 1: 1
float tmp[16];  // Thread-local temporary buffer
```

**2. Work State**

Each thread tracks its own:
- **Current chunk being processed**
- **Progress through work items**
- **Local computation results** (before writing to shared output)

#### 7.11.5 Complete Data Sharing and Synchronization Summary

**Shared Data (Read by All Threads):**
| Data Structure | Access Pattern | Synchronization |
|----------------|----------------|-----------------|
| **Model weights** | Read-only, all threads | None (read-only safe) |
| **Input tensors** | Read-only, all threads | None (read-only safe) |
| **KV cache (read)** | Read-only, all threads | None (read-only safe) |
| **Intermediate tensors (read)** | Read-only, all threads | None (read-only safe) |

**Shared Data (Partitioned Writes):**
| Data Structure | Access Pattern | Synchronization |
|----------------|----------------|-----------------|
| **Output tensors** | Each thread writes different rows | Barrier after write |
| **Attention scores** | Each thread writes different rows | Barrier after write |
| **Softmax output** | Each thread writes different rows | Barrier after write |
| **FFN outputs** | Each thread writes different rows | Barrier after write |

**Shared Data (Sequential Write):**
| Data Structure | Access Pattern | Synchronization |
|----------------|----------------|-----------------|
| **KV cache (write)** | Single thread writes | Barrier before write |

**Synchronization Mechanisms:**
| Mechanism | Purpose | When Used |
|-----------|---------|-----------|
| **Barrier** | Ensure all threads finish before next operation | After each graph node |
| **Atomic operations** | Thread-safe chunk assignment | Work stealing |
| **Sequential execution** | Operations that can't be parallelized | KV cache update, concatenation |

#### 7.11.6 Why This Design Works

**1. No Locks/Mutexes Needed (Most Operations)**

**Reason:** Threads write to **different memory regions**, so no conflicts.

```
Output Tensor Q [2, 288]:
├─> Thread 0 writes: Q[0, :]  (memory region A)
└─> Thread 1 writes: Q[1, :]  (memory region B)

No overlap → No conflicts → No locks needed
```

**2. Barriers Are Sufficient**

**Reason:** Barriers ensure **ordering** (all threads finish before next step), not **mutual exclusion**.

```
Timeline:
T0: Thread 0 writes Q[0, :]
T1: Thread 1 writes Q[1, :]
T2: BARRIER (both threads wait)
T3: Both threads can now read complete Q matrix
```

**3. Read-Only Sharing is Safe**

**Reason:** Multiple threads reading the same data simultaneously is always safe.

```
Model Weight attn_q [288, 288]:
├─> Thread 0 reads: attn_q (entire matrix)
└─> Thread 1 reads: attn_q (entire matrix)

No writes → No conflicts → No synchronization needed
```

#### 7.11.7 Example: Complete Data Flow with `-t 2`

**Operation: Q = hidden × attn_q**

```
Shared Data Structures:
├─> hidden [2, 288]        ← SHARED (read by both threads)
├─> attn_q [288, 288]       ← SHARED (read by both threads)
└─> Q [2, 288]             ← SHARED (partitioned writes)

Thread 0 (ith=0):
├─> Reads: hidden[0, :] (row 0)           ← From shared tensor
├─> Reads: attn_q (entire matrix)          ← From shared tensor
├─> Computes: Q[0, :] = hidden[0, :] × attn_q
└─> Writes: Q[0, :] (row 0)               ← To shared tensor, row 0

Thread 1 (ith=1):
├─> Reads: hidden[1, :] (row 1)           ← From shared tensor
├─> Reads: attn_q (entire matrix)         ← From shared tensor
├─> Computes: Q[1, :] = hidden[1, :] × attn_q
└─> Writes: Q[1, :] (row 1)               ← To shared tensor, row 1

Synchronization:
└─> BARRIER: Both threads wait here
└─> After barrier: Both threads see complete Q matrix [2, 288]
```

**Key Points:**
1. **All data structures are shared** (single copy in memory)
2. **Threads read from shared data** (no conflicts, read-only)
3. **Threads write to different regions** (no conflicts, different memory addresses)
4. **Barriers ensure ordering** (all writes complete before reads)
5. **No locks/mutexes needed** (no overlapping writes)

---

## Complete Data Flow Diagram (CPU-Only Execution)

## Complete Data Flow Diagram (CPU-Only Execution)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. MODEL LOADING                                            │
│    File: stories15M-q4_0.gguf                                │
│    ├─> llama_model_loader::llama_model_loader()            │
│    │   └─> Parse GGUF metadata (20 key-value pairs)        │
│    │   └─> Load 57 tensors (13 F32, 43 Q4_0, 1 Q8_0)       │
│    │                                                         │
│    └─> llama_model::load_tensors()                          │
│        ├─> For each tensor:                                 │
│        │   ├─> ggml_new_tensor() - Create tensor metadata  │
│        │   ├─> get_buffer_type() - Select CPU buffer type  │
│        │   │   └─> CPU_IMI buffer type (for IMI builds)     │
│        │   ├─> ggml_backend_buft_alloc_buffer() - Allocate │
│        │   │   └─> Allocate in CPU backend buffer          │
│        │   └─> load_data_for() - Load from file            │
│        │       └─> Repack weights to IMI format (if IMI)  │
│        └─> Result: All weights in CPU backend buffers      │
│            └─> All tensors in CPU memory (RISC-V)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 2. CONTEXT INITIALIZATION (CPU-ONLY)                       │
│    ├─> llama_context::llama_context()                       │
│    │   ├─> Register CPU backend (ONLY backend)             │
│    │   │   └─> backends = [CPU]                            │
│    │   ├─> Register REF backend (optional, for validation)  │
│    │   ├─> ggml_backend_sched_new() - Create scheduler     │
│    │   │   └─> Single backend scheduler                    │
│    │   └─> model.create_memory() - Initialize KV cache     │
│    │       └─> KV cache allocated in CPU memory            │
│    │                                                         │
│    └─> llama_attach_threadpool()                           │
│        └─> Create thread pools (2 threads for -t 2)         │
│            └─> Single thread pool (decode + prefill)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 3. INFERENCE REQUEST                                        │
│    Input: "Hello, world!"                                   │
│    ├─> llama_tokenize() - Convert to token IDs             │
│    │   └─> Result: [9906, 11, 1917, 0] (tokens)          │
│    │                                                         │
│    └─> llama_batch_get_one() - Create batch                │
│        └─> Result: llama_batch with 4 tokens              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 4. GRAPH CONSTRUCTION                                       │
│    llama_context::graph_build()                             │
│    ├─> Input: Token embeddings                             │
│    │   └─> ggml_get_rows(model.tok_embeddings, tokens)     │
│    │       └─> Embeddings in CPU memory                    │
│    │                                                         │
│    ├─> Layer 0:                                            │
│    │   ├─> RMSNorm                                          │
│    │   ├─> QKV projections (3 × MUL_MAT)                    │
│    │   │   └─> Weights in CPU buffers                      │
│    │   ├─> RoPE                                            │
│    │   ├─> Attention computation                           │
│    │   ├─> Output projection (MUL_MAT)                     │
│    │   ├─> Residual add                                    │
│    │   ├─> FFN: gate, up, down (3 × MUL_MAT)               │
│    │   └─> Residual add                                    │
│    │                                                         │
│    ├─> Layers 1-5: (same as Layer 0)                       │
│    │                                                         │
│    └─> Output:                                             │
│        ├─> Final RMSNorm                                   │
│        └─> Output projection (MUL_MAT) → logits            │
│                                                             │
│    Result: ~155 nodes in computation graph                 │
│    └─> All nodes will execute on CPU backend               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 5. BACKEND SCHEDULING (CPU-ONLY)                           │
│    ggml_backend_sched_alloc_graph()                       │
│    ├─> Pass 1: Assign backends                            │
│    │   └─> For each node:                                  │
│    │       ├─> Check weight tensor location               │
│    │       │   └─> All weights in CPU buffers             │
│    │       └─> Assign to CPU backend (backend_id = 0)     │
│    │           └─> All 155 nodes → CPU backend             │
│    │                                                         │
│    ├─> Pass 2: Split graph                                │
│    │   └─> Result: Single split (all 155 nodes → CPU)    │
│    │       └─> No backend boundaries to split at         │
│    │                                                         │
│    └─> Pass 3: Allocate tensors                            │
│        └─> All output tensors in CPU backend buffers      │
│            └─> No cross-backend allocations               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 6. CPU BACKEND EXECUTION                                   │
│    ggml_backend_cpu_graph_compute()                        │
│    ├─> Get thread pool (2 threads for -t 2)               │
│    ├─> Create compute plan                                │
│    └─> ggml_graph_compute_threaded()                      │
│        └─> Spawn 2 worker threads                          │
│            └─> Execute on RISC-V CPU (via QEMU)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 7. THREAD EXECUTION (2 THREADS)                           │
│    ggml_graph_compute_thread() (per thread)               │
│    ├─> Thread 0-1: Process all nodes sequentially         │
│    │   ├─> Node 0: Token embedding lookup                │
│    │   │   └─> Barrier sync                               │
│    │   ├─> Node 1: Layer 0 RMSNorm                        │
│    │   │   └─> Barrier sync                               │
│    │   ├─> Node 2: Layer 0 Q projection (MUL_MAT)        │
│    │   │   ├─> Thread 0: Processes half of rows           │
│    │   │   └─> Thread 1: Processes other half             │
│    │   │   └─> Barrier sync                               │
│    │   └─> ... (continue for all 155 nodes)              │
│    │                                                         │
│    └─> Wait for all threads                                │
│        └─> All execution on RISC-V CPU                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 8. CPU KERNEL DISPATCH (RISC-V)                            │
│    ggml_compute_forward_mul_mat()                         │
│    ├─> Check: IMI buffer type?                            │
│    │   ├─> Yes: ggml_imi_gemm_q4_0_q8_0_8x4()            │
│    │   │   └─> Use RISC-V Vector (RVV) intrinsics        │
│    │   │       └─> I-Machines custom instructions         │
│    │   └─> No: ggml_vec_dot_q4_0_q8_0()                  │
│    │       └─> Standard CPU kernel (RISC-V)               │
│    │                                                         │
│    └─> Result: Output tensor computed on CPU              │
│        └─> All computation on RISC-V CPU backend          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 9. OUTPUT                                                  │
│    Logits: [n_tokens, vocab_size]                         │
│    ├─> llama_get_logits_ith(ctx, -1) - Get last token    │
│    │   └─> Logits in CPU memory                           │
│    ├─> common_sampler_sample() - Sample next token        │
│    └─> Result: Next token ID                              │
│        └─> All processing on CPU                           │
└─────────────────────────────────────────────────────────────┘
```

**CPU-Only Execution Characteristics:**
- **Single Backend**: Only CPU backend registered and used
- **No GPU**: All operations execute on RISC-V CPU (via QEMU)
- **No Cross-Backend Transfers**: All tensors stay in CPU memory
- **Simplified Scheduling**: All operations assigned to CPU (no choice)
- **Single Graph Split**: All nodes in one continuous split
- **Direct Execution**: Graph executed directly on CPU backend

---

## Key Code Locations

### Model Loading
- **GGUF Parsing**: [`src/llama-model-loader.cpp:471`](../dev_env/llama.cpp/src/llama-model-loader.cpp#L471) (`llama_model_loader` constructor)
- **Tensor Loading**: [`src/llama-model.cpp:load_tensors()`](../dev_env/llama.cpp/src/llama-model.cpp)
- **Buffer Allocation**: [`ggml/src/ggml-backend.cpp:ggml_backend_buft_alloc_buffer()`](../dev_env/llama.cpp/ggml/src/ggml-backend.cpp)

### Context Initialization
- **Context Creation**: [`src/llama-context.cpp:19`](../dev_env/llama.cpp/src/llama-context.cpp#L19) (`llama_context` constructor)
- **Backend Registration**: [`src/llama-context.cpp:157-184`](../dev_env/llama.cpp/src/llama-context.cpp#L157)
- **Scheduler Creation**: [`ggml/src/ggml-backend.cpp:678`](../dev_env/llama.cpp/ggml/src/ggml-backend.cpp#L678) (`ggml_backend_sched_new`)

### Graph Construction
- **Graph Building**: [`src/llama-graph.cpp:graph_build()`](../dev_env/llama.cpp/src/llama-graph.cpp)
- **Decode Entry**: [`src/llama-context.cpp:109`](../dev_env/llama.cpp/src/llama-context.cpp#L109) (`llama_context::decode()`)
- **Batch Processing**: [`src/llama-context.cpp:103`](../dev_env/llama.cpp/src/llama-context.cpp#L103) (`process_ubatch()`)

### Backend Scheduling
- **Graph Allocation**: [`ggml/src/ggml-backend.cpp:1282`](../dev_env/llama.cpp/ggml/src/ggml-backend.cpp#L1282) (`ggml_backend_sched_alloc_graph`)
- **Backend Assignment**: [`ggml/src/ggml-backend.cpp:776`](../dev_env/llama.cpp/ggml/src/ggml-backend.cpp#L776) (`ggml_backend_sched_backend_id_from_cur`)
- **Graph Splitting**: [`ggml/src/ggml-backend.cpp:912`](../dev_env/llama.cpp/ggml/src/ggml-backend.cpp#L912) (`ggml_backend_sched_split_graph`)

### CPU Backend Execution
- **Graph Compute**: [`ggml/src/ggml-backend-cpu.cpp:ggml_backend_cpu_graph_compute()`](../dev_env/llama.cpp/ggml/src/ggml-backend-cpu.cpp)
- **Thread Execution**: [`ggml/src/ggml-threading.cpp:ggml_graph_compute_thread()`](../dev_env/llama.cpp/ggml/src/ggml-threading.cpp)
- **Operation Dispatch**: [`ggml/src/ggml.c:ggml_compute_forward()`](../dev_env/llama.cpp/ggml/src/ggml.c)

### Kernel Execution
- **Standard CPU**: [`ggml/src/ggml-cpu/ggml-cpu.c:1221`](../dev_env/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c#L1221) (`ggml_compute_forward_mul_mat`)
- **IMI Check**: [`ggml/src/ggml-cpu/traits.cpp:12`](../dev_env/llama.cpp/ggml/src/ggml-cpu/traits.cpp#L12) (`ggml_cpu_extra_compute_forward`)
- **IMI Kernels**: [`ggml/src/ggml-cpu/imi/opt-kernels.cpp`](../dev_env/llama.cpp/ggml/src/ggml-cpu/imi/opt-kernels.cpp)

---

## Summary

### Key Takeaways (CPU-Only Execution)

1. **Model Loading**: GGUF file → Parse metadata → Load tensors → Allocate in **CPU backend buffers only**
2. **Context Init**: Register **CPU backend only** → Create scheduler → Initialize KV cache in CPU memory → Attach thread pools
3. **Graph Building**: Build forward pass → Create ~155 nodes for 6-layer model → All nodes execute on CPU
4. **Scheduling**: All operations assigned to **CPU backend** (only backend available) → Allocate tensors in CPU buffers
5. **Execution**: Thread pool (2 threads for `-t 2`) processes all nodes sequentially → Work split within each node
6. **Kernels**: IMI kernels used if weights in IMI buffer type, else standard CPU kernels (both run on RISC-V CPU)

### CPU-Only Execution Characteristics

**Single Backend Architecture:**
- **Only CPU backend**: No GPU backends registered or used
- **All operations on CPU**: Every computation runs on RISC-V CPU (via QEMU emulation)
- **No cross-backend transfers**: All tensors stay in CPU memory throughout execution
- **Simplified scheduling**: No backend selection needed (only one choice)
- **Single graph split**: All ~155 nodes processed in one continuous split

### Data Locality Principle (CPU-Only)

**All operations run on CPU backend:**
- Model weights loaded into CPU backend buffers (CPU_IMI or standard CPU)
- All operations assigned to CPU backend (only backend available)
- All intermediate tensors allocated in CPU buffers
- **No data movement**: Everything stays in CPU memory
- **Maximum efficiency**: No overhead from cross-backend transfers

### Threading Model (CPU-Only)

- **All threads process all nodes** sequentially
- **Within each node**, work is split among threads (2 threads for `-t 2`)
- **Barrier synchronization** ensures correctness between operations
- **Dynamic work stealing** for load balancing
- **RISC-V execution**: All threads execute on RISC-V CPU via QEMU

### RISC-V CPU Backend

**Execution Environment:**
- **Architecture**: RISC-V with I-Machines extensions
- **Emulation**: QEMU user-mode (`qemu-riscv64`)
- **CPU Model**: `imicpu-v1` (I-Machines CPU model)
- **Vector Extensions**: RISC-V Vector (RVV) with VLEN=128
- **Custom Instructions**: I-Machines optimized instructions for matrix operations

**Kernel Selection:**
- **IMI kernels**: Used when weights in CPU_IMI buffer type (RVV-optimized)
- **Standard CPU kernels**: Fallback for operations without IMI optimization
- **Both run on CPU**: All kernels execute on RISC-V CPU, no GPU involved

This architecture ensures efficient execution on **CPU backend only**, with all computation happening on the RISC-V CPU via QEMU emulation.
