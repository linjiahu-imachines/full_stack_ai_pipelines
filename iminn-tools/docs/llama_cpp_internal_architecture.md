# llama.cpp Internal Architecture Deep Dive

**Date:** December 9, 2025  
**Location:** `/home/linhu/repo/iminn-tools/dev_env/llama.cpp`

## Overview

This document provides a comprehensive analysis of llama.cpp's internal architecture, covering:

1. **GGUF Model Loading** - How models are loaded from GGUF files, metadata parsing, and tensor allocation
2. **Batching System** - How batches are created, managed, split into unified batches, and processed
3. **KV Cache Management** - Memory-efficient storage of attention keys/values, cache strategies, and eviction
4. **Attention Mechanism** - Multi-head attention computation, QKV projections, causal masking, and Flash Attention
5. **Parallelization & Threading** - Thread pools, parallel execution strategies, and synchronization
6. **Backend Scheduling** - How operations are scheduled across CPU, GPU, and other backends
7. **Computation Graph** - Graph construction, layer-by-layer building, and graph reuse
8. **Tokenization & Vocabulary** - Token encoding/decoding, vocabulary management, and special tokens
9. **Sampling Strategies** - Token sampling algorithms, repetition penalties, and advanced techniques
10. **Key Data Structures** - Core structures for models, contexts, and memory management

---

## Table of Contents

1. [GGUF Model Loading](#1-gguf-model-loading)
2. [Batching System](#2-batching-system)
3. [KV Cache Management](#3-kv-cache-management)
4. [Attention Mechanism](#4-attention-mechanism)
5. [Parallelization & Threading](#5-parallelization--threading)
6. [Backend Scheduling](#6-backend-scheduling)
7. [Computation Graph](#7-computation-graph)
8. [Tokenization & Vocabulary](#8-tokenization--vocabulary)
9. [Sampling Strategies](#9-sampling-strategies)
10. [Key Data Structures](#10-key-data-structures)

---

## 1. GGUF Model Loading

### 1.1 GGUF File Format

**GGUF (GPT-Generated Unified Format)** is the file format used by llama.cpp to store model weights and metadata.

**File Structure:**
- **Header**: Magic number, version, metadata
- **Metadata**: Key-value pairs (architecture, hyperparameters, tokenizer info)
- **Tensors**: Model weights in various quantization formats

**Supported Versions:**
```cpp
enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,  // Latest
};
```

### 1.2 Model Loader Architecture

**Key Class:** `llama_model_loader` (`src/llama-model-loader.h`)

**Core Components:**

```cpp
struct llama_model_loader {
    // File information
    llama_files files;              // GGUF file(s) - supports split files
    llama_ftype ftype;              // File type (quantization format)
    llama_fver  fver;               // GGUF version
    
    // Metadata
    gguf_context_ptr meta;          // GGUF metadata context
    std::string arch_name;          // Architecture name (e.g., "llama")
    LLM_KV llm_kv;                  // Key-value metadata
    
    // Tensor weights
    std::map<std::string, llama_tensor_weight> weights_map;
    
    // Memory mapping
    llama_mmaps mappings;           // Memory-mapped file regions
    bool use_mmap;                  // Use memory mapping for large files
};
```

### 1.3 Loading Process

**Step 1: Initialize Loader** (`src/llama-model-loader.cpp:471`)

```cpp
llama_model_loader::llama_model_loader(
    const std::string & fname,
    std::vector<std::string> & splits,
    bool use_mmap,
    bool check_tensors,
    const llama_model_kv_override * param_overrides_p,
    const llama_model_tensor_buft_override * param_tensor_buft_overrides_p)
```

**Process:**
1. Open main GGUF file
2. Parse header and metadata
3. Load additional split files if present
4. Build tensor weight map

**Step 2: Parse Metadata**

Metadata is accessed via template functions:

```cpp
template<typename T>
bool get_key(const std::string & key, T & result, bool required = true);

template<typename T>
bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);
```

**Key Metadata Fields:**
- `general.architecture` - Model architecture (llama, gpt, etc.)
- `general.name` - Model name
- `llama.context_length` - Training context size
- `llama.embedding_length` - Embedding dimension
- `llama.block_count` - Number of transformer blocks
- `tokenizer.ggml.tokens` - Vocabulary tokens
- `tokenizer.ggml.scores` - Token scores

**Step 3: Load Tensor Weights**

**Tensor Weight Structure:**
```cpp
struct llama_tensor_weight {
    uint16_t  idx;        // Source file index (for split files)
    size_t   offs;        // Tensor data offset in file
    ggml_tensor * tensor; // GGML tensor metadata
};
```

**Loading Process:**
1. **Create tensor metadata** - Shape, type, name
2. **Map file offset** - Calculate byte offset in GGUF file
3. **Memory mapping** - If `use_mmap=true`, map file region to memory
4. **Load data** - Copy or map tensor data to backend buffer

**Step 4: Allocate Backend Buffers**

Tensors are allocated in backend-specific buffers:

```cpp
void load_data_for(struct ggml_tensor * cur) const {
    const llama_tensor_weight & w = require_weight(ggml_get_name(cur));
    
    // Allocate buffer in appropriate backend
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(
        get_buffer_type(cur), ggml_nbytes(cur));
    
    // Load data from file
    if (use_mmap) {
        // Memory-mapped access
    } else {
        // Direct file read
    }
}
```

### 1.4 Split File Support

llama.cpp supports **split GGUF files** for large models:

```cpp
// Example: model-00001-of-00004.gguf, model-00002-of-00004.gguf, ...
std::vector<std::string> llama_get_list_splits(
    const std::string & path, 
    const int idx, 
    const int n_split);
```

**Split File Loading:**
1. Parse split index from filename
2. Discover all split files
3. Load metadata from all splits
4. Map tensors to appropriate split file

### 1.5 Quantization Formats

**Supported Formats** (from `llama-model-loader.cpp:24-66`):

| Format | Description | Bits per Weight |
|--------|-------------|-----------------|
| `ALL_F32` | Full precision float32 | 32 |
| `MOSTLY_F16` | Half precision float16 | 16 |
| `MOSTLY_BF16` | Brain float16 | 16 |
| `MOSTLY_Q4_0` | 4-bit quantization | 4 |
| `MOSTLY_Q8_0` | 8-bit quantization | 8 |
| `MOSTLY_Q2_K` | K-quantization (2-bit) | ~2 |
| `MOSTLY_Q4_K` | K-quantization (4-bit) | ~4 |
| `MOSTLY_IQ2_XXS` | Imatrix quantization | ~2 |
| `MOSTLY_MXFP4` | Microscaling FP4 | 4 |

---

## 2. Batching System

### 2.1 Batch Structure

**Key Structure:** `llama_batch` (`include/llama.h`)

```cpp
struct llama_batch {
    int32_t n_tokens;           // Number of tokens in batch
    
    llama_token  * token;       // Token IDs [n_tokens]
    float        * embd;        // Embeddings [n_embd, n_tokens] (optional)
    llama_pos    * pos;         // Positions [n_tokens]
    int32_t      * n_seq_id;    // Number of sequences per token [n_tokens]
    llama_seq_id ** seq_id;     // Sequence IDs [n_tokens][n_seq_id[i]]
    int8_t       * logits;      // Output flags [n_tokens]
    
    // Internal fields
    uint8_t logits_all;         // All tokens produce logits
    uint8_t all_pos_0;          // All positions are 0
    uint8_t all_pos_1;          // All positions are 1
    uint8_t all_seq_id;         // All tokens belong to same sequence
};
```

### 2.2 Unified Batch (ubatch)

**Key Structure:** `llama_ubatch` (`src/llama-batch.h:15-67`)

A **unified batch** is a processed batch with equal-length sequence sets:

```cpp
struct llama_ubatch {
    uint32_t n_tokens;          // Total tokens
    uint32_t n_seq_tokens;      // Tokens per sequence set
    uint32_t n_seqs;            // Number of sequence sets
    uint32_t n_seqs_unq;        // Unique sequence IDs
    
    bool equal_seqs() const;    // All sequences have equal length
    
    llama_token  * token;       // Token IDs
    float        * embd;        // Embeddings
    llama_pos    * pos;         // Positions (can be multi-dimensional)
    llama_seq_id ** seq_id;     // Sequence IDs
    int8_t       * output;      // Output flags
};
```

### 2.3 Batch Allocator

**Key Class:** `llama_batch_allocr` (`src/llama-batch.h:70-171`)

**Purpose:** Sanitize, fulfill, and split batches into unified batches.

**Initialization** (`src/llama-batch.cpp:25-200`):

```cpp
bool llama_batch_allocr::init(
    const llama_batch & batch_inp,
    const llama_vocab & vocab,
    const llama_memory_i * memory,
    uint32_t n_embd,
    uint32_t n_seq_max,
    bool output_all)
```

**Process:**
1. **Validate input** - Check token IDs, sequence IDs
2. **Auto-generate missing fields:**
   - `n_seq_id` - If missing, default to 1 sequence per token
   - `seq_id` - If missing, assign default sequence ID
   - `pos` - If missing, compute from memory or start at 0
   - `logits` - If missing, set based on `output_all`
3. **Compute statistics:**
   - Count outputs
   - Track sequence positions
   - Identify coupled sequences (tokens belonging to multiple sequences)

### 2.4 Batch Splitting

**Splitting Methods:**

#### 2.4.1 Simple Split

```cpp
llama_ubatch split_simple(uint32_t n_ubatch);
```

- Unknown number of sequence sets
- Unequal sequence lengths
- Creates ubatches up to `n_ubatch` tokens

#### 2.4.2 Equal Split

```cpp
llama_ubatch split_equal(uint32_t n_ubatch, bool sequential);
```

- Creates ubatches with **equal-length sequence sets**
- All sequences in ubatch have same length
- `sequential=true`: Tokens have increasing sequence IDs

#### 2.4.3 Sequence Split

```cpp
llama_ubatch split_seq(uint32_t n_ubatch);
```

- Each ubatch contains a **single sequence set**
- Used for sequence-level processing

### 2.5 Sequence Coupling

**Coupled Sequences:** Sequences that share tokens in the batch.

```cpp
// seq_cpl[s0][s1]: sequence s1 is coupled to sequence s0
std::vector<std::vector<bool>> seq_cpl;
```

**Use Case:** Multi-modal models where text and image tokens share positions.

### 2.6 Batch Processing Flow

```
Input Batch (llama_batch)
    ↓
Batch Allocator (init)
    ├─> Validate tokens, sequences
    ├─> Auto-generate missing fields
    └─> Compute statistics
    ↓
Split into Unified Batches (ubatch)
    ├─> split_simple()    - Unequal lengths
    ├─> split_equal()     - Equal lengths
    └─> split_seq()       - Single sequence
    ↓
Process Each ubatch
    ├─> Build computation graph
    ├─> Execute on backends
    └─> Collect outputs
```

---

## 3. KV Cache Management

### 3.1 Purpose of KV Cache

The **KV (Key-Value) Cache** stores previously computed attention keys and values to avoid recomputing them during autoregressive generation. This is critical for performance in decode phase, where each token depends on all previous tokens.

**Without KV Cache:**
- Each token generation requires recomputing attention for all previous tokens
- O(n²) computation complexity for n tokens

**With KV Cache:**
- Store K and V from previous tokens
- Only compute K and V for the new token
- O(n) computation complexity

### 3.2 KV Cache Structure

**Key Components:**

```cpp
struct llama_kv_cache {
    // Per-layer KV cache
    std::vector<ggml_tensor *> k;  // Key tensors [n_layers]
    std::vector<ggml_tensor *> v;  // Value tensors [n_layers]
    
    // Cache dimensions
    uint32_t size;        // Total cache size (in tokens/positions)
    uint32_t n_cells;     // Number of cache cells allocated
    uint32_t n_layers;    // Number of transformer layers
    
    // Memory backend
    ggml_backend_buffer_t buffer;  // Backend buffer storing cache
};
```

**Memory Layout:**

For each layer, KV cache stores:
- **K tensor**: Shape `[n_ctx, n_kv, head_dim]`
  - `n_ctx`: Maximum context length (e.g., 4096)
  - `n_kv`: Number of key-value heads (may differ from query heads)
  - `head_dim`: Dimension per attention head
  
- **V tensor**: Shape `[n_ctx, n_kv, head_dim]`
  - Same dimensions as K tensor

**Memory Size Calculation:**
```
KV cache size per layer = 2 * n_ctx * n_kv * head_dim * sizeof(f16)
Total KV cache = n_layers * KV cache size per layer
```

Example: 6 layers, 4096 context, 6 KV heads, 48 head_dim, F16:
```
Per layer: 2 * 4096 * 6 * 48 * 2 bytes = 4.5 MiB
Total: 6 * 4.5 MiB = 27.00 MiB
```

### 3.3 Memory Management Strategies

#### 3.3.1 Hybrid Memory (`llama_memory_hybrid`)

**File:** `src/llama-memory-hybrid.cpp`

**Characteristics:**
- Fixed-size allocation
- Per-sequence KV cache slots
- Supports multiple sequences with independent caches
- Used for standard transformer architectures

**Initialization:**
```cpp
llama_memory_context_ptr llama_memory_hybrid::init_batch(
    const llama_batch_allocr & allocr,
    uint32_t n_ubatch,
    bool output_all)
{
    // Allocate KV cache for each sequence
    // Track position offsets for each sequence
    // Support sequence-level parallelism
}
```

**Memory Layout:**
```
┌─────────────────────────────────────────┐
│ Sequence 0 KV Cache (Layer 0)           │
│   K: [0..n_pos_0]                       │
│   V: [0..n_pos_0]                       │
├─────────────────────────────────────────┤
│ Sequence 1 KV Cache (Layer 0)           │
│   K: [0..n_pos_1]                       │
│   V: [0..n_pos_1]                       │
├─────────────────────────────────────────┤
│ ...                                     │
└─────────────────────────────────────────┘
```

#### 3.3.2 Recurrent Memory (`llama_memory_recurrent`)

**File:** `src/llama-memory-recurrent.cpp`

**Characteristics:**
- Dynamic memory management
- Used for RNN-like architectures (e.g., Mamba)
- Supports variable-length sequences
- More efficient for very long contexts

### 3.4 KV Cache Updates

**Update Process** (during forward pass):

1. **Compute new K, V** for current tokens:
   ```cpp
   // QKV projection
   auto q = ggml_mul_mat(model.layers[i].attn_q, hidden);
   auto k = ggml_mul_mat(model.layers[i].attn_k, hidden);
   auto v = ggml_mul_mat(model.layers[i].attn_v, hidden);
   ```

2. **Append to cache**:
   ```cpp
   // Concatenate with cached K, V
   auto k_all = ggml_concat(cache.k[i], k, /*dim=*/0);
   auto v_all = ggml_concat(cache.v[i], v, /*dim=*/0);
   ```

3. **Update cache state**:
   - Increment position counter
   - Update cache pointers
   - Track sequence positions

**Position Management:**
- Each sequence maintains its own position offset
- Cache positions map to sequence-relative positions
- Supports concurrent sequences with different lengths

### 3.5 Cache Eviction and Management

**Context Window Limits:**
- When context exceeds `n_ctx`, oldest tokens are evicted
- Strategies:
  - **FIFO**: Remove oldest tokens
  - **Sliding Window**: Shift cache positions
  - **Selective Eviction**: Remove based on attention scores

**Memory Efficiency:**
- KV cache is often the largest memory consumer after model weights
- Compression techniques:
  - Quantization (F16 → Q8_0 or Q4_0)
  - Sparse caching (only cache certain layers)
  - Dynamic allocation based on sequence length

### 3.6 Unified KV Cache

**Configuration:** `cparams.kv_unified`

When enabled (`kv_unified=true`):
- Single shared KV cache across all sequences
- More memory efficient for batched inference
- Requires careful position management

When disabled (`kv_unified=false`):
- Separate KV cache per sequence
- Better isolation but higher memory usage
- Default for multi-sequence scenarios

---

## 4. Attention Mechanism

### 4.1 Multi-Head Attention Overview

The attention mechanism in transformers computes relationships between tokens. llama.cpp implements several attention variants optimized for different scenarios.

**Core Attention Formula:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where:
- **Q** (Query): What information are we looking for?
- **K** (Key): What information is available?
- **V** (Value): What is the actual content?

### 4.2 Attention Computation Steps

#### Step 1: QKV Projections

**File:** `src/llama-graph.cpp`

```cpp
// Input: hidden states [n_tokens, n_embd]
auto q = ggml_mul_mat(model.layers[i].attn_q, hidden);  // [n_tokens, n_q * head_dim]
auto k = ggml_mul_mat(model.layers[i].attn_k, hidden);  // [n_tokens, n_kv * head_dim]
auto v = ggml_mul_mat(model.layers[i].attn_v, hidden);  // [n_tokens, n_kv * head_dim]
```

**Weight Shapes:**
- `attn_q`: `[n_embd, n_q * head_dim]`
- `attn_k`: `[n_embd, n_kv * head_dim]`
- `attn_v`: `[n_embd, n_kv * head_dim]`

Where:
- `n_q`: Number of query heads (e.g., 6)
- `n_kv`: Number of key-value heads (may be less for grouped-query attention)
- `head_dim`: Dimension per head (e.g., `n_embd / n_q = 48`)

#### Step 2: Reshape into Multi-Head Format

```cpp
// Reshape to [n_tokens, n_q, head_dim]
q = ggml_reshape_3d(q, head_dim, n_q, n_tokens);
q = ggml_permute(q, 0, 2, 1, 3);  // [n_q, n_tokens, head_dim]

// Similar for K and V
k = ggml_reshape_3d(k, head_dim, n_kv, n_tokens);
k = ggml_permute(k, 0, 2, 1, 3);  // [n_kv, n_tokens, head_dim]

v = ggml_reshape_3d(v, head_dim, n_kv, n_tokens);
v = ggml_permute(v, 0, 2, 1, 3);  // [n_kv, n_tokens, head_dim]
```

#### Step 3: Concatenate with KV Cache

```cpp
// Retrieve cached K, V from previous tokens
auto k_cache = memory->get_k(i);  // [n_kv, n_prev, head_dim]
auto v_cache = memory->get_v(i);  // [n_kv, n_prev, head_dim]

// Concatenate: [n_kv, n_prev + n_tokens, head_dim]
auto k_all = ggml_concat(k_cache, k, /*dim=*/1);
auto v_all = ggml_concat(v_cache, v, /*dim=*/1);
```

#### Step 4: Compute Attention Scores

```cpp
// QK^T: [n_q, n_tokens, head_dim] × [n_kv, n_prev + n_tokens, head_dim]^T
// Result: [n_q, n_tokens, n_prev + n_tokens]
auto scores = ggml_mul_mat(q, k_all);

// Scale by sqrt(head_dim)
scores = ggml_scale(scores, 1.0f / sqrtf(head_dim));
```

#### Step 5: Apply Causal Mask

**Causal Attention:** Tokens can only attend to previous tokens (including themselves).

```cpp
// Create causal mask: -inf for future positions, 0 for past/current
// Mask shape: [n_tokens, n_prev + n_tokens]
auto mask = ggml_causal_mask(n_tokens, n_prev + n_tokens);

// Apply mask: scores[mask] = -inf
scores = ggml_add(scores, mask);
```

**Mask Pattern** (for n_tokens=3, n_prev=2):
```
Position:  0  1  2 | 3  4
Token 3:   0  0  0 | 0 -∞ -∞  (can see 0,1,2,3; not 4,5)
Token 4:   0  0  0 | 0  0 -∞  (can see 0,1,2,3,4; not 5)
Token 5:   0  0  0 | 0  0  0   (can see all previous)
```

#### Step 6: Softmax and Apply to Values

```cpp
// Softmax: convert scores to probabilities
scores = ggml_soft_max(scores);  // [n_q, n_tokens, n_prev + n_tokens]

// Multiply by V: [n_q, n_tokens, n_prev + n_tokens] × [n_kv, n_prev + n_tokens, head_dim]
// Result: [n_q, n_tokens, head_dim]
auto attn_out = ggml_mul_mat(scores, v_all);
```

#### Step 7: Output Projection

```cpp
// Reshape back: [n_q, n_tokens, head_dim] → [n_tokens, n_q * head_dim]
attn_out = ggml_permute(attn_out, 0, 2, 1, 3);
attn_out = ggml_reshape_2d(attn_out, n_q * head_dim, n_tokens);

// Output projection: [n_tokens, n_q * head_dim] × [n_q * head_dim, n_embd]
attn_out = ggml_mul_mat(model.layers[i].attn_o, attn_out);  // [n_tokens, n_embd]
```

### 4.3 Attention Variants

#### 4.3.1 Standard Multi-Head Attention

- **n_q == n_kv**: Equal number of query and key-value heads
- **Full attention matrix**: O(n²) memory for n tokens
- Used in: Llama 1, GPT-2

#### 4.3.2 Grouped-Query Attention (GQA)

- **n_q > n_kv**: More query heads than key-value heads
- **Shared KV heads**: Multiple query heads share same K, V
- **Memory efficient**: Reduces KV cache size
- Used in: Llama 2, Llama 3

Example: `n_q = 6`, `n_kv = 2` (3:1 ratio)
- Query heads: 0, 1, 2, 3, 4, 5
- KV heads: 0, 1
- Q heads 0-2 share KV head 0
- Q heads 3-5 share KV head 1

#### 4.3.3 Flash Attention

**File:** `ggml/src/ggml-cpu/ops.cpp:ggml_compute_forward_flash_attn_ext`

**Benefits:**
- Memory efficient: O(n) instead of O(n²)
- Tiled computation: Process attention in blocks
- Fused operations: Combine softmax and matrix multiply

**When Enabled:**
- Automatically enabled for supported backends
- Requires backend support (CUDA, Metal, CPU with specific optimizations)
- Can be disabled via `flash_attn=false` parameter

**Implementation:**
```cpp
// Tiled attention computation
for (int block_i = 0; block_i < n_blocks; block_i++) {
    for (int block_j = 0; block_j < n_blocks; block_j++) {
        // Compute QK^T for tile
        auto q_block = q[block_i];
        auto k_block = k_cache[block_j];
        auto scores = matmul(q_block, k_block);
        
        // Apply causal mask for tile
        if (block_j > block_i) scores = -inf;
        
        // Softmax and multiply with V block
        auto attn_block = softmax(scores) * v_block;
        output[block_i] += attn_block;
    }
}
```

### 4.4 Rotary Position Embedding (RoPE)

**File:** `src/llama-graph.cpp`

Instead of adding position embeddings, RoPE rotates Q and K vectors based on position.

**Implementation:**
```cpp
// Apply RoPE to Q and K before attention
q = ggml_rope(q, pos, n_ctx, freq_base, freq_scale);
k = ggml_rope(k, pos, n_ctx, freq_base, freq_scale);
```

**Benefits:**
- Better extrapolation to longer contexts
- Relative position encoding
- Computationally efficient

**Parameters:**
- `freq_base`: Base frequency (e.g., 10000.0)
- `freq_scale`: Frequency scaling factor for context extension

---

## 5. Parallelization & Threading

### 3.1 Thread Configuration

**Context Parameters** (`src/llama-cparams.h`):

```cpp
struct llama_cparams {
    uint32_t n_threads;        // Threads for decode (autoregressive)
    uint32_t n_threads_batch;  // Threads for prefill (parallel)
};
```

**Initialization** (`src/llama-context.cpp:38-39`):

```cpp
cparams.n_threads       = params.n_threads;
cparams.n_threads_batch = params.n_threads_batch;
```

### 3.2 Thread Pools

**GGML Thread Pools** (`ggml/src/ggml-threading.cpp`):

llama.cpp uses GGML's thread pool system:

```cpp
struct ggml_threadpool {
    int n_threads;
    // ... thread management
};
```

**Thread Pool Attachment** (`src/llama-context.cpp:69-73`):

```cpp
void llama_context::attach_threadpool(
    ggml_threadpool_t threadpool,
    ggml_threadpool_t threadpool_batch)
```

**Two Thread Pools:**
1. **`threadpool`** - For decode phase (autoregressive, sequential)
2. **`threadpool_batch`** - For prefill phase (parallel processing)

### 3.3 Parallelization Strategies

#### 3.3.1 Decode Phase (Autoregressive)

**Characteristics:**
- Sequential token generation
- Each token depends on previous tokens
- Limited parallelism

**Thread Usage:**
- `n_threads` threads
- Parallelize within single token computation:
  - Matrix-vector operations (GEMV)
  - Attention computation
  - Feed-forward layers

#### 3.3.2 Prefill Phase (Parallel)

**Characteristics:**
- Process entire prompt in parallel
- All tokens available upfront
- High parallelism

**Thread Usage:**
- `n_threads_batch` threads
- Parallelize across tokens:
  - Matrix-matrix operations (GEMM)
  - Batch attention
  - Parallel feed-forward

### 3.4 Operation-Level Parallelization

**GGML Graph Computation** (`ggml/src/ggml-threading.cpp`):

```cpp
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    
    // Iterate through graph nodes
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];
        
        // Compute node with thread-specific parameters
        ggml_compute_forward(&params, node);
        
        // Synchronize between operations
        ggml_barrier(state->threadpool);
    }
}
```

**Thread Parameters:**
```cpp
struct ggml_compute_params {
    int ith;                    // Thread index
    int nth;                    // Total threads
    size_t wsize;               // Work size
    void * wdata;               // Work data
    ggml_threadpool_t threadpool;
};
```

### 3.5 Multi-Threaded Batch Processing

**Batch Processing** (`src/llama-context.cpp:1092-1143`):

```cpp
do {
    const auto & ubatch = mctx->get_ubatch();
    
    // Process ubatch with graph computation
    const auto * res = process_ubatch(
        ubatch, 
        LLM_GRAPH_TYPE_DECODER, 
        mctx.get(), 
        status);
    
    // Continue with next ubatch
} while (mctx->next_ubatch());
```

**Parallel Execution:**
- Each ubatch processed independently
- Multiple ubatches can be processed in parallel (if supported)
- Thread pools shared across ubatches

---

## 6. Backend Scheduling

### 4.1 Backend Architecture

**Backend Types** (`ggml/include/ggml-backend.h`):

```cpp
enum ggml_backend_device_type {
    GGML_BACKEND_DEVICE_TYPE_CPU,
    GGML_BACKEND_DEVICE_TYPE_GPU,
    GGML_BACKEND_DEVICE_TYPE_ACCEL,  // BLAS, etc.
};
```

**Backend Initialization** (`src/llama-context.cpp:157-184`):

```cpp
// GPU backends
for (auto * dev : model.devices) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    backends.emplace_back(backend);
}

// ACCEL backends (BLAS)
for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        backends.emplace_back(backend);
    }
}

// CPU backend (always added last)
backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
backends.emplace_back(backend_cpu);
```

### 4.2 Backend Scheduler

**Key Structure:** `ggml_backend_sched` (`ggml/src/ggml-backend.cpp`)

**Scheduler Creation** (`src/llama-context.cpp:285`):

```cpp
sched.reset(ggml_backend_sched_new(
    backend_ptrs.data(),        // Backend pointers
    backend_buft.data(),        // Buffer types
    backend_ptrs.size(),        // Number of backends
    max_nodes,                  // Max graph nodes
    pipeline_parallel,          // Enable pipeline parallelism
    cparams.op_offload));      // Operation offload config
```

### 4.3 Operation Scheduling

**Scheduling Process:**

1. **Graph Analysis:**
   - Analyze computation graph
   - Identify tensor dependencies
   - Determine operation requirements

2. **Backend Assignment:**
   - Assign operations to backends based on:
     - **Tensor location** - Prefer backend where tensors are stored
     - **Operation support** - Backend must support operation
     - **Performance** - Estimate execution time
     - **Memory** - Consider memory constraints

3. **Memory Management:**
   - Allocate buffers in appropriate backends
   - Transfer tensors between backends as needed
   - Manage temporary buffers

### 4.4 Pipeline Parallelism

**Pipeline Parallelism** (`src/llama-context.cpp:259-289`):

Enabled when:
- Multiple devices available (`model.n_devices() > 1`)
- GPU layers exceed model layers
- Split mode is layer-based
- KQV offloading enabled
- All devices support async compute and events

**Benefits:**
- Overlap computation across devices
- Process different layers on different devices simultaneously
- Improve throughput for large models

### 4.5 Operation Offloading

**Offload Configuration** (`src/llama-cparams.h`):

```cpp
struct llama_cparams {
    llama_offload_kqv op_offload;  // KQV offload strategy
};
```

**Offload Strategies:**
- **KQV Offload** - Offload Key/Query/Value operations to GPU
- **Layer Offload** - Offload entire layers to GPU
- **Selective Offload** - Offload specific operations

### 4.6 Backend Buffer Types

**Buffer Type Selection** (`src/llama-context.cpp:228-246`):

```cpp
for (auto & backend : backends) {
    auto * buft = ggml_backend_get_default_buffer_type(backend.get());
    auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
    
    if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
        // Use host buffer of first device for faster transfers
        auto * dev = model.devices[0];
        auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (host_buft) {
            buft = host_buft;
        }
    }
    
    backend_buft.push_back(buft);
}
```

### 4.7 Graph Execution

**Graph Execution** (`src/llama-context.cpp:process_ubatch`):

```cpp
llm_graph_result * process_ubatch(
    const llama_ubatch & ubatch,
    llm_graph_type gtype,
    llama_memory_context_i * mctx,
    ggml_status & ret)
{
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
    
    // Synchronize and get results
    ggml_backend_sched_synchronize(sched.get());
    return gf_res_prev.get();
}
```

---

## 7. Computation Graph

### 5.1 Graph Structure

**GGML Computation Graph** (`ggml/include/ggml.h`):

```cpp
struct ggml_cgraph {
    int n_nodes;                // Number of nodes
    int n_leafs;                // Number of leaf nodes (inputs)
    struct ggml_tensor ** nodes; // Node array
    struct ggml_tensor ** grads; // Gradient array (for training)
    struct ggml_cplan cplan;     // Computation plan
};
```

### 7.2 Graph Building Process

**Graph Building** (`src/llama-graph.cpp`):

The graph is built layer by layer through forward pass construction:

#### 7.2.1 Input Layer

```cpp
// Token embeddings: [n_tokens] → [n_tokens, n_embd]
auto tok_embd = ggml_get_rows(model.tok_embeddings, batch.token);

// Position embeddings (RoPE applied later, or learned embeddings)
auto pos_embd = compute_position_embeddings(batch.pos, n_ctx);

// Combine: hidden = tok_embd + pos_embd (if learned positions)
auto hidden = tok_embd;  // RoPE doesn't require addition
```

**Key Operations:**
- `ggml_get_rows`: Lookup embeddings for token IDs
- `ggml_rope`: Apply rotary position embedding (if RoPE)
- Input shape: `[n_tokens, n_embd]`

#### 7.2.2 Transformer Block (Per Layer)

For each transformer layer `i` (0 to `n_layers-1`):

**Step 1: Pre-Attention Layer Norm**
```cpp
// RMSNorm: normalize hidden states
auto hidden_norm = ggml_rms_norm(hidden, model.layers[i].attn_norm);
```

**Step 2: Attention Computation**
```cpp
// QKV projections
auto q = ggml_mul_mat(model.layers[i].attn_q, hidden_norm);
auto k = ggml_mul_mat(model.layers[i].attn_k, hidden_norm);
auto v = ggml_mul_mat(model.layers[i].attn_v, hidden_norm);

// Apply RoPE
q = ggml_rope(q, batch.pos, n_ctx, freq_base, freq_scale);
k = ggml_rope(k, batch.pos, n_ctx, freq_base, freq_scale);

// Attention computation (see Section 4.2 for details)
auto attn_out = compute_attention(q, k, v, memory, i);

// Output projection
attn_out = ggml_mul_mat(model.layers[i].attn_o, attn_out);
```

**Step 3: Residual Connection (Attention)**
```cpp
hidden = ggml_add(hidden, attn_out);  // Residual: hidden = hidden + attn_out
```

**Step 4: Pre-FFN Layer Norm**
```cpp
hidden_norm = ggml_rms_norm(hidden, model.layers[i].ffn_norm);
```

**Step 5: Feed-Forward Network**
```cpp
// Gate projection
auto gate = ggml_mul_mat(model.layers[i].ffn_gate, hidden_norm);

// Up projection
auto up = ggml_mul_mat(model.layers[i].ffn_up, hidden_norm);

// Activation (SwiGLU: gate * silu(gate) * up)
gate = ggml_silu(gate);  // SiLU activation
auto ffn_out = ggml_mul(gate, up);  // Element-wise multiply

// Down projection
ffn_out = ggml_mul_mat(model.layers[i].ffn_down, ffn_out);
```

**Step 6: Residual Connection (FFN)**
```cpp
hidden = ggml_add(hidden, ffn_out);  // Residual: hidden = hidden + ffn_out
```

**Graph Nodes Created Per Layer:**
- RMSNorm (×2)
- Matrix multiplications (×6: Q, K, V, O, gate, up, down)
- RoPE (×2)
- Attention computation (multiple nodes)
- Activation (SiLU)
- Element-wise operations (×2: mul, add)
- **Total: ~20-30 nodes per layer**

#### 7.2.3 Output Layer

```cpp
// Final layer norm
hidden = ggml_rms_norm(hidden, model.norm);

// Output projection: [n_tokens, n_embd] × [n_embd, vocab_size]
auto logits = ggml_mul_mat(model.output, hidden);  // [n_tokens, vocab_size]
```

#### 7.2.4 Graph Construction Summary

**Total Graph Nodes:**
- Input: ~3 nodes
- Per layer: ~25 nodes
- Output: ~2 nodes
- **Total: ~3 + (n_layers × 25) + 2**

For 6-layer model: **~155 nodes**

**Graph Structure:**
```
Input (tokens, positions)
  ↓
Embedding Lookup
  ↓
Layer 0: Norm → Attention → Add → Norm → FFN → Add
  ↓
Layer 1: Norm → Attention → Add → Norm → FFN → Add
  ↓
...
  ↓
Layer N-1: Norm → Attention → Add → Norm → FFN → Add
  ↓
Output Norm
  ↓
Output Projection
  ↓
Logits
```

### 5.3 Graph Reuse

**Graph Reuse** (`src/llama-context.cpp:109-115`):

```cpp
const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? 
    (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;
```

**Reuse Conditions:**
- Same input tensor shapes
- Same graph type
- Same memory context
- Input tensors haven't changed

**Benefits:**
- Avoid rebuilding graph for each inference
- Faster execution
- Reduced memory allocations

---

## 8. Tokenization & Vocabulary

### 8.1 Vocabulary Structure

**Key File:** `src/llama-vocab.cpp`

The vocabulary maps between tokens (integers) and text strings.

**Core Structure:**
```cpp
struct llama_vocab {
    std::vector<llama_vocab::id> id_to_token;  // Token ID → token string
    std::map<llama_token_data, llama_vocab::id> token_to_id;  // Token string → ID
    
    // Token metadata
    std::vector<float> token_scores;  // Score for each token (for BPE)
    std::vector<llama_vocab_type> token_types;  // Token type (normal, control, etc.)
    
    // Special tokens
    llama_token eos_token;      // End-of-sequence
    llama_token bos_token;      // Beginning-of-sequence
    llama_token pad_token;      // Padding
    llama_token nl_token;       // Newline
    // ... other special tokens
};
```

**Token Types:**
```cpp
enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NORMAL = 0,      // Regular text token
    LLAMA_VOCAB_TYPE_UNKNOWN = 1,     // Unknown token
    LLAMA_VOCAB_TYPE_CONTROL = 2,     // Control token (<|im_start|>)
    LLAMA_VOCAB_TYPE_USER_DEFINED = 4, // User-defined token
    LLAMA_VOCAB_TYPE_UNUSED = 8,      // Unused token
    LLAMA_VOCAB_TYPE_BYTE = 16,       // Byte token (for byte-level BPE)
};
```

### 8.2 Tokenization Methods

llama.cpp supports multiple tokenization algorithms:

#### 8.2.1 BPE (Byte-Pair Encoding)

**Used by:** GPT models, most Llama models

**Process:**
1. Start with character-level vocabulary
2. Iteratively merge most frequent token pairs
3. Build vocabulary of subword tokens

**Example:**
```
Original: "hello world"
Character-level: ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
After BPE: ['hello', ' world']  (if "hello" and " world" are in vocabulary)
Token IDs: [9906, 1917]
```

**Implementation:** `src/llama-vocab.cpp:llama_vocab_load()`

#### 8.2.2 SentencePiece

**Used by:** T5, some multilingual models

**Features:**
- Unicode normalization
- Subword segmentation
- Language-agnostic

#### 8.2.3 WordPiece

**Used by:** BERT, ALBERT

**Similar to BPE but with different merge strategy**

### 8.3 Tokenization Process

**Function:** `llama_tokenize()` (`include/llama.h`)

```cpp
int llama_tokenize(
    const llama_model * model,
    const char * text,
    llama_token * tokens,
    int n_max_tokens,
    bool add_bos,
    bool special);
```

**Steps:**

1. **Preprocessing:**
   ```cpp
   // Normalize text (NFKC normalization)
   std::string normalized = normalize_utf8(text);
   ```

2. **Tokenization:**
   ```cpp
   // Apply tokenizer (BPE/SentencePiece/etc.)
   std::vector<llama_token> token_ids = tokenizer.encode(normalized);
   ```

3. **Special Token Handling:**
   ```cpp
   if (add_bos && model->vocab.bos_token != -1) {
       tokens.insert(tokens.begin(), model->vocab.bos_token);
   }
   ```

4. **Validation:**
   - Check token IDs are valid
   - Respect `n_max_tokens` limit

**Example:**
```cpp
const char * text = "Hello, world!";
llama_token tokens[32];
int n_tokens = llama_tokenize(model, text, tokens, 32, true, false);
// Result: tokens = [1, 9906, 11, 1917, 0]  (BOS=1, tokens, EOS=0)
```

### 8.4 Detokenization

**Function:** `llama_token_to_piece()` (`include/llama.h`)

Converts token ID back to text string:

```cpp
int llama_token_to_piece(
    const llama_model * model,
    llama_token token,
    char * buf,
    int length);
```

**Process:**
1. Look up token ID in vocabulary
2. Convert to UTF-8 string
3. Handle special tokens (BOS, EOS, control tokens)
4. Handle byte tokens (convert byte IDs to characters)

**Example:**
```cpp
llama_token token = 9906;
char buf[256];
int len = llama_token_to_piece(model, token, buf, 256);
// Result: buf = "Hello", len = 5
```

### 8.5 Vocabulary Loading from GGUF

**File:** `src/llama-model-loader.cpp`

**Metadata Keys:**
- `tokenizer.ggml.tokens`: Array of token strings
- `tokenizer.ggml.scores`: Array of token scores (for BPE)
- `tokenizer.ggml.token_type`: Array of token types
- `tokenizer.ggml.bos_token_id`: Beginning-of-sequence token ID
- `tokenizer.ggml.eos_token_id`: End-of-sequence token ID
- `tokenizer.ggml.unk_token_id`: Unknown token ID
- `tokenizer.ggml.pad_token_id`: Padding token ID

**Loading Process:**
```cpp
// Read token strings
std::vector<std::string> tokens;
ml.get_arr("tokenizer.ggml.tokens", tokens);

// Read scores (for BPE ranking)
std::vector<float> scores;
ml.get_arr("tokenizer.ggml.scores", scores);

// Build vocabulary
for (size_t i = 0; i < tokens.size(); i++) {
    vocab.id_to_token[i] = tokens[i];
    vocab.token_scores[i] = scores[i];
    vocab.token_to_id[tokens[i]] = i;
}
```

### 8.6 Special Token Handling

**Control Tokens:**
- System prompts: `<|im_start|>`, `<|im_end|>`
- Function calling: `<|function_call|>`, `<|function_result|>`
- Code blocks: `<|code|>`, `<|/code|>`

**Behavior:**
- Special tokens are preserved during tokenization
- Can be enabled/disabled via `special` parameter
- Used for structured generation (chat, function calling)

---

## 9. Sampling Strategies

### 9.1 Overview

After computing logits (probability distribution over vocabulary), sampling selects the next token. llama.cpp implements numerous sampling strategies for diverse text generation.

**Function:** `common_sampler_sample()` (`common/sampling.cpp`)

### 9.2 Sampling Pipeline

**Order of Operations:**

```
Logits
  ↓
Logit Bias (penalize/boost specific tokens)
  ↓
Penalties (repeat, frequency, presence)
  ↓
Dual YARA (DRY) - reduces repetition
  ↓
Top-N-Sigma (statistical filtering)
  ↓
Top-K (keep only K most likely tokens)
  ↓
Typical P (remove atypical tokens)
  ↓
Top-P (Nucleus sampling - keep tokens until cumulative prob > P)
  ↓
Min-P (minimum probability threshold)
  ↓
XTC (extreme temperature compensation)
  ↓
Temperature (apply temperature scaling)
  ↓
Distribution → Sample token
```

### 9.3 Core Sampling Strategies

#### 9.3.1 Greedy Sampling

Simply select the token with highest probability:

```cpp
llama_token id = std::max_element(logits.begin(), logits.end()) - logits.begin();
```

**Characteristics:**
- Deterministic
- Fast
- Often produces repetitive text

#### 9.3.2 Temperature Sampling

Scale logits by temperature before softmax:

```cpp
for (auto & logit : logits) {
    logit /= temperature;  // temperature > 0
}
auto probs = softmax(logits);
llama_token id = sample_from_distribution(probs);
```

**Temperature Effects:**
- `temperature = 0`: Deterministic (greedy)
- `temperature < 1`: More focused, less random
- `temperature = 1`: Standard probability distribution
- `temperature > 1`: More random, more diverse

#### 9.3.3 Top-K Sampling

Keep only the K tokens with highest probability:

```cpp
// Sort logits and keep top K
std::partial_sort(logits.begin(), logits.begin() + k, logits.end(), std::greater<>());
// Set others to -inf
for (int i = k; i < vocab_size; i++) {
    logits[i] = -INFINITY;
}
auto probs = softmax(logits);
llama_token id = sample_from_distribution(probs);
```

**Parameters:**
- `top_k`: Number of tokens to keep (e.g., 40)
- Typical range: 10-100

#### 9.3.4 Top-P (Nucleus) Sampling

Keep tokens until cumulative probability exceeds P:

```cpp
// Sort tokens by probability (descending)
std::vector<std::pair<float, int>> sorted;
for (int i = 0; i < vocab_size; i++) {
    sorted.push_back({probs[i], i});
}
std::sort(sorted.begin(), sorted.end(), std::greater<>());

// Keep tokens until cumulative prob > top_p
float cumsum = 0.0f;
int cutoff = 0;
for (const auto & [prob, idx] : sorted) {
    cumsum += prob;
    cutoff++;
    if (cumsum >= top_p) break;
}

// Zero out tokens after cutoff
for (int i = cutoff; i < vocab_size; i++) {
    logits[sorted[i].second] = -INFINITY;
}
```

**Parameters:**
- `top_p`: Cumulative probability threshold (e.g., 0.95)
- Typical range: 0.7-0.99

#### 9.3.5 Min-P Sampling

Remove tokens below minimum probability threshold:

```cpp
float max_logit = *std::max_element(logits.begin(), logits.end());
float min_threshold = max_logit + log(min_p);

for (auto & logit : logits) {
    if (logit < min_threshold) {
        logit = -INFINITY;
    }
}
```

**Parameters:**
- `min_p`: Minimum probability relative to max (e.g., 0.05)

#### 9.3.6 Typical P Sampling

Keep tokens with "typical" probability (based on entropy):

```cpp
// Compute entropy of distribution
float entropy = -sum(probs * log(probs));

// Keep tokens with probability close to typical value
float typical_prob = exp(-entropy);
float threshold = typical_p * typical_prob;

for (int i = 0; i < vocab_size; i++) {
    if (probs[i] < threshold) {
        logits[i] = -INFINITY;
    }
}
```

**Parameters:**
- `typical_p`: Typical probability multiplier (e.g., 1.0)

### 9.4 Repetition Penalties

#### 9.4.1 Repeat Penalty

Penalize tokens that appeared recently:

```cpp
// Track last N tokens
std::vector<llama_token> recent_tokens = get_last_n_tokens(n);

// Apply penalty
for (llama_token token_id : recent_tokens) {
    if (logits[token_id] > 0) {
        logits[token_id] /= repeat_penalty;  // Decrease probability
    } else {
        logits[token_id] *= repeat_penalty;  // Increase negative logits
    }
}
```

**Parameters:**
- `repeat_penalty`: Penalty factor (1.0 = no penalty, >1.0 = penalize)
- `repeat_last_n`: Number of recent tokens to consider (e.g., 64)

#### 9.4.2 Frequency Penalty

Penalize tokens based on frequency in entire sequence:

```cpp
// Count token frequencies
std::map<llama_token, int> freq = count_token_frequencies(all_tokens);

// Apply penalty
for (const auto & [token_id, count] : freq) {
    logits[token_id] -= frequency_penalty * count;
}
```

**Parameters:**
- `frequency_penalty`: Penalty per occurrence (e.g., 0.0-1.0)

#### 9.4.3 Presence Penalty

Penalize tokens that appeared at least once:

```cpp
// Track which tokens appeared
std::set<llama_token> present_tokens = get_present_tokens(all_tokens);

// Apply penalty
for (llama_token token_id : present_tokens) {
    logits[token_id] -= presence_penalty;
}
```

**Parameters:**
- `presence_penalty`: Penalty for presence (e.g., 0.0-1.0)

### 9.5 Dual YARA (DRY)

**Purpose:** Reduce repetition by analyzing token patterns

**Process:**
1. Identify repeated n-grams (sequences of tokens)
2. Compute "dryness" score based on repetition frequency
3. Apply penalty to tokens that continue repetitive patterns

**Parameters:**
- `dry_multiplier`: Strength of DRY penalty
- `dry_base`: Base threshold
- `dry_allowed_length`: Maximum allowed n-gram length

### 9.6 Advanced Sampling

#### 9.6.1 Mirostat

Adaptive temperature that maintains target entropy:

```cpp
// Mirostat v1: Maintain target surprise
float mu = mirostat_mu;  // Learning rate
float tau = mirostat_tau;  // Target surprise

float error = tau - compute_surprise(sampled_token);
mu -= alpha * error;  // Update learning rate
temperature = mu;  // Use as temperature
```

**Parameters:**
- `mirostat`: 0=disabled, 1=v1, 2=v2
- `mirostat_tau`: Target surprise (e.g., 5.0)
- `mirostat_eta`: Learning rate (e.g., 0.1)

#### 9.6.2 Logit Bias

Manually adjust probabilities for specific tokens:

```cpp
// Apply custom biases
for (const auto & [token_id, bias] : logit_bias_map) {
    logits[token_id] += bias;  // bias can be positive or negative
}
```

**Use Cases:**
- Force certain tokens (bias = +inf)
- Prevent certain tokens (bias = -inf)
- Boost/dampen specific tokens

### 9.7 Sampling Configuration Example

**From llama-cli output:**
```
sampler params: 
    repeat_last_n = 64, repeat_penalty = 1.000
    frequency_penalty = 0.000, presence_penalty = 0.000
    top_k = 40, top_p = 0.950, min_p = 0.050
    typical_p = 1.000, temp = 0.800
    mirostat = 0
```

This configuration:
- Uses top-k=40, top-p=0.95, min-p=0.05
- Applies temperature=0.8 (slightly focused)
- No repetition penalties (1.0 = disabled)
- No mirostat (0 = disabled)

---

## 10. Key Data Structures

### 6.1 Model Structure

```cpp
struct llama_model {
    llama_hparams hparams;              // Hyperparameters
    llama_vocab vocab;                  // Vocabulary
    std::vector<ggml_backend_dev_t> devices; // GPU devices
    llama_model_params params;          // Model parameters
    // ... tensors, buffers, etc.
};
```

### 6.2 Context Structure

```cpp
struct llama_context {
    const llama_model & model;          // Reference to model
    llama_cparams cparams;              // Context parameters
    ggml_backend_sched_ptr sched;       // Backend scheduler
    llama_memory_ptr memory;            // KV cache memory
    llama_batch_allocr_ptr balloc;      // Batch allocator
    // ... graphs, buffers, etc.
};
```

### 6.3 Memory Context

```cpp
struct llama_memory_context_i {
    virtual llama_ubatch get_ubatch() = 0;
    virtual bool next_ubatch() = 0;
    virtual llama_memory_status get_status() = 0;
    // ... memory management
};
```

---

## Summary

### Key Takeaways

1. **GGUF Loading:**
   - Memory-mapped file access for efficiency
   - Support for split files and multiple quantization formats
   - Backend-specific buffer allocation
   - Metadata-driven architecture detection

2. **Batching:**
   - Flexible batch structure supporting multiple sequences
   - Unified batches for efficient processing
   - Automatic field generation and validation
   - Multiple splitting strategies (simple, equal, sequence)

3. **KV Cache Management:**
   - Stores computed attention keys/values to avoid recomputation
   - Hybrid and recurrent memory strategies
   - Per-sequence or unified cache allocation
   - Major memory consumer - often 27+ MiB for standard models

4. **Attention Mechanism:**
   - Multi-head attention with QKV projections
   - Causal masking for autoregressive generation
   - Grouped-query attention (GQA) for memory efficiency
   - Flash Attention for O(n) memory complexity
   - Rotary position embedding (RoPE) for position encoding

5. **Parallelization:**
   - Separate thread pools for decode and prefill phases
   - Operation-level parallelism within tokens
   - Batch-level parallelism across sequences
   - Barrier synchronization between operations

6. **Backend Scheduling:**
   - Automatic operation assignment to optimal backends
   - Pipeline parallelism for multi-device setups
   - Efficient memory management and tensor transfers
   - Support for CPU, GPU, and accelerator backends

7. **Computation Graph:**
   - Layer-by-layer graph construction
   - Graph reuse for repeated inference
   - ~150-200 nodes for typical models
   - Async execution with synchronization

8. **Tokenization:**
   - Multiple algorithms (BPE, SentencePiece, WordPiece)
   - Vocabulary size typically 30k-100k tokens
   - Special token handling (BOS, EOS, control tokens)

9. **Sampling:**
   - Multi-stage pipeline (top-k, top-p, temperature, penalties)
   - Repetition penalties (frequency, presence, DRY)
   - Advanced strategies (Mirostat, typical-p, min-p)

### Performance Considerations

**Memory Optimizations:**
- **Graph Reuse:** Reduces overhead for repeated inference
- **Memory Mapping:** Efficient loading of large models
- **KV Cache Quantization:** F16 → Q8_0/Q4_0 for cache compression
- **Flash Attention:** Reduces attention memory from O(n²) to O(n)

**Computation Optimizations:**
- **Batch Splitting:** Optimizes for different sequence lengths
- **Backend Selection:** Maximizes utilization of available hardware
- **Pipeline Parallelism:** Overlaps computation across devices
- **SIMD/Vectorization:** CPU instruction-level parallelism
- **Quantization:** Reduces memory bandwidth and compute

**Threading Optimizations:**
- **Separate Pools:** Different thread counts for decode vs prefill
- **Polling Configuration:** Balance between latency and CPU usage
- **Thread Affinity:** Pin threads to CPU cores for cache locality

### Architecture-Specific Features

**Llama Architecture:**
- RoPE for position encoding
- SwiGLU activation in FFN
- RMSNorm for layer normalization
- GQA (grouped-query attention)

**Other Architectures Supported:**
- GPT (different attention patterns)
- Mamba (recurrent memory)
- Custom architectures via architecture plugins

---

## Related Files

| Component | Files | Purpose |
|-----------|-------|---------|
| **Model Loading** | `src/llama-model-loader.{h,cpp}`, `ggml/src/gguf.cpp` | GGUF file parsing, tensor loading |
| **Model Architecture** | `src/llama-model.cpp`, `src/llama-arch.cpp` | Architecture-specific implementations |
| **Batching** | `src/llama-batch.{h,cpp}`, `include/llama.h` | Batch creation, splitting, unified batches |
| **KV Cache** | `src/llama-kv-cache.cpp`, `src/llama-memory-*.cpp` | KV cache management, memory strategies |
| **Graph Building** | `src/llama-graph.{h,cpp}` | Computation graph construction |
| **Context Management** | `src/llama-context.{h,cpp}` | Runtime context, decode execution |
| **Vocabulary** | `src/llama-vocab.cpp` | Tokenization, vocabulary management |
| **Sampling** | `common/sampling.cpp`, `src/llama-sampling.cpp` | Token sampling strategies |
| **Backend Scheduling** | `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-backend-impl.h` | Backend management, operation scheduling |
| **Threading** | `ggml/src/ggml-threading.{h,cpp}`, `ggml/src/ggml-cpu/ggml-cpu.c` | Thread pools, parallel execution |
| **Attention** | `ggml/src/ggml-cpu/ops.cpp` | Flash attention, standard attention |
| **Quantization** | `ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/quants.c` | Quantization formats, dequantization |
| **GGML Core** | `ggml/src/ggml.c`, `ggml/include/ggml.h` | Tensor operations, computation graph |

---

## References

- llama.cpp GitHub: https://github.com/ggml-org/llama.cpp
- GGML Documentation: `dev_env/llama.cpp/docs/`
- GGUF Specification: `dev_env/llama.cpp/ggml/src/gguf.cpp`

