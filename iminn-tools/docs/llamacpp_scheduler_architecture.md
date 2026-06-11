# llama.cpp Scheduler and Backend Dispatch Architecture

This document explains how llama.cpp schedules and dispatches computation for a GGUF model across multiple backends (CPU, CPU_IMI, REF) and threads.

---

## Architecture Overview

llama.cpp uses a **centralized multi-backend scheduler** that:
1. Assigns operations to backends based on data locality and capabilities
2. Splits the compute graph at backend boundaries
3. Manages tensor allocation across different backends
4. Coordinates cross-backend data transfers
5. Dispatches thread-parallel execution within each backend

```
                    ┌─────────────────────────────────┐
                    │   GGUF Model (weights)          │
                    │   - Q4_0/Q8_0/F16 tensors       │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   Backend Scheduler             │
                    │   (ggml_backend_sched)          │
                    │   - Priority-based selection    │
                    │   - Graph splitting             │
                    │   - Tensor allocation           │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   Compute Graph Splits          │
                    ├─────────────┬───────────────────┤
                    │  Split 1    │  Split 2  │ Split 3│
                    │  (CPU_IMI)  │  (CPU)    │ (REF)  │
                    └──────┬──────┴─────┬─────┴────┬───┘
                           │            │          │
                  ┌────────▼────┐  ┌───▼────┐  ┌──▼────┐
                  │  CPU_IMI    │  │  CPU   │  │  REF  │
                  │  Backend    │  │Backend │  │Backend│
                  └──────┬──────┘  └───┬────┘  └───┬───┘
                         │             │           │
                  ┌──────▼──────┐ ┌────▼─────┐    │
                  │ IMI Kernels │ │ CPU Vec  │    │
                  │ (optimized) │ │ Dot      │    │
                  └─────────────┘ └──────────┘    │
                         │             │           │
                  ┌──────▼─────────────▼───────────▼──┐
                  │     Thread Pool (4 threads)       │
                  │  Thread 0 | 1 | 2 | 3             │
                  │  Each processes assigned work     │
                  └───────────────────────────────────┘
```

---

## 1. Scheduler Data Structure

### Central Scheduler: `ggml_backend_sched`
**Location**: `ggml/src/ggml-backend.cpp:678-726`

```cpp
struct ggml_backend_sched {
    bool is_reset;
    bool is_alloc;

    // Backend management
    int n_backends;  // Number of registered backends
    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];  // Priority-ordered
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;  // Graph allocator

    // Tensor-to-backend mapping (hash table)
    struct ggml_hash_set hash_set;
    int * hv_tensor_backend_ids;  // Which backend owns each tensor
    struct ggml_tensor ** hv_tensor_copies;  // Cross-backend tensor copies

    // Per-node backend assignments
    int * node_backend_ids;  // Backend for each compute node
    int * leaf_backend_ids;  // Backend for each leaf (input) tensor

    // Graph splits for multi-backend execution
    struct ggml_backend_sched_split * splits;
    int n_splits;

    // Pipeline parallelism support
    int n_copies;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];

    bool op_offload;  // Enable operation offloading to GPU
};
```

**Key Insight**: The scheduler maintains a **hash table** mapping every tensor to its assigned backend, allowing fast lookup during graph construction.

---

## 2. Backend Registration and Priority

### Backend Registration Order (Higher Priority = Lower Index)
**Location**: `ggml/src/ggml-backend-reg.cpp:188-227`

```cpp
ggml_backend_registry() {
    // Priority 0-N: GPU backends (highest priority)
    #ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
    #endif
    #ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
    #endif

    // Priority N+1: CPU backends (lower priority)
    #ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
    #endif

    // Priority N+2: REF backend (lowest priority, fallback)
    #ifdef GGML_USE_REF
        register_backend(ggml_backend_ref_reg());
    #endif
}
```

**For IMI builds (no GPU):**
- Priority 0: **CPU** backend (standard CPU operations)
- Priority 1: **REF** backend (reference implementation for validation)

**Important**: CPU and CPU_IMI are **not separate backends**. Instead, CPU_IMI is a **buffer type** within the CPU backend.

---

## 3. Backend Selection Algorithm

### Step-by-Step Backend Assignment
**Location**: `ggml/src/ggml-backend.cpp:776-831` (`ggml_backend_sched_backend_id_from_cur`)

For each tensor operation, the scheduler uses this algorithm:

```cpp
int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched,
                                           struct ggml_tensor * tensor) {
    // STEP 1: Check if tensor already has a buffer allocated
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        return cur_backend_id;  // Use buffer's backend
    }

    // STEP 2: Handle view tensors (use source tensor's backend)
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(
            sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            return cur_backend_id;
        }
    }

    // STEP 3: Graph inputs default to CPU backend (last backend)
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1;  // CPU
        return cur_backend_id;
    }

    // STEP 4: Data locality - prefer backend where weights are located
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) continue;

        // Operations with weights run on same backend as weights
        if (src->buffer != NULL &&
            src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(
                sched, src, tensor);

            // STEP 5: Check for GPU offloading opportunity
            if (sched->op_offload &&
                src_backend_id == sched->n_backends - 1 &&  // CPU backend
                ggml_backend_buffer_is_host(src->buffer)) {
                // Try to offload to higher-priority GPU backends
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) &&
                        ggml_backend_offload_op(sched->backends[b], tensor)) {
                        return b;  // Offload to GPU
                    }
                }
            }
            return src_backend_id;
        }
    }

    return -1;  // No assignment yet
}
```

**Key Principle**: **Data Locality** - Operations run on the backend where their weights are stored to minimize data movement.

---

## 4. CPU vs CPU_IMI Backend Selection

### IMI is a Buffer Type, Not a Separate Backend

**Key Concept**: CPU_IMI is implemented as an **extra buffer type** within the CPU backend, not as a separate backend.

**Location**: `ggml/src/ggml-cpu/imi/imi.cpp:961-1023`

```cpp
namespace ggml::cpu::imi {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    // Check if this operation should use IMI kernels
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        // IMI handles MUL_MAT if:
        // 1. Weight buffer is CPU_IMI type
        // 2. Input is F32 in host memory
        // 3. Optimal IMI kernel exists for weight dimensions
        if (op->op == GGML_OP_MUL_MAT &&
            op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_cpu_imi_buffer_type() &&
            ggml_imi_get_optimal_repack_type(op->src[0])) {
            return (op->src[1]->type == GGML_TYPE_F32);
        }
        return false;
    }

    // Get IMI-specific tensor traits (kernel implementation)
    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            if (op->src[0]->buffer &&
                op->src[0]->buffer->buft == ggml_backend_cpu_imi_buffer_type()) {
                // Return IMI-specific kernel stored in tensor->extra
                return (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        return nullptr;
    }
};
}
```

### When IMI Kernels Are Used

**Decision flow**:
```
Load GGUF Model
    │
    ├─> Weights allocated in CPU_IMI buffer type
    │   └─> Tensors repacked during load (set_tensor callback)
    │
    └─> During execution:
        │
        ├─> Check: Is weight buffer type CPU_IMI?  ──No──> Use standard CPU kernels
        │   └─> Yes
        │
        ├─> Check: Does optimal IMI kernel exist for shape?  ──No──> Use standard CPU
        │   └─> Yes
        │
        └─> Use IMI optimized kernel (ggml_imi_gemm_q4_0_q8_0_8x4, etc.)
```

### Optimal IMI Kernel Selection
**Location**: `ggml/src/ggml-cpu/imi/imi.cpp:865-903`

```cpp
static const ggml::cpu::tensor_traits * ggml_imi_get_optimal_repack_type(
    const struct ggml_tensor * cur) {

    // Define available IMI kernels
    static const ggml::cpu::imi::tensor_traits<block_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_q8_0_8x4;
    static const ggml::cpu::imi::tensor_traits<block_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_q8_0_4x4;
    static const ggml::cpu::imi::tensor_traits<block_q8_0, 8, 4, GGML_TYPE_Q8_0> q8_0_q8_0_8x4;
    static const ggml::cpu::imi::tensor_traits<block_q8_0, 4, 4, GGML_TYPE_Q8_0> q8_0_q8_0_4x4;

    // Select based on weight tensor shape
    if (cur->type == GGML_TYPE_Q4_0) {
        if (cur->ne[1] % 8 == 0) {
            return &q4_0_q8_0_8x4;  // Use 8-row kernel (most efficient)
        }
        if (cur->ne[1] % 4 == 0) {
            return &q4_0_q8_0_4x4;  // Use 4-row kernel
        }
    }

    if (cur->type == GGML_TYPE_Q8_0) {
        if (cur->ne[1] % 8 == 0) {
            return &q8_0_q8_0_8x4;  // Use 8-row kernel
        }
        if (cur->ne[1] % 4 == 0) {
            return &q8_0_q8_0_4x4;  // Use 4-row kernel
        }
    }

    return nullptr;  // No optimal IMI kernel, use standard CPU
}
```

**Kernel Selection Based on Matrix Dimensions**:
- **8-row kernels** (`8x4`): Used when weight matrix rows are divisible by 8 (most efficient)
- **4-row kernels** (`4x4`): Used when weight matrix rows are divisible by 4
- **Fallback**: Standard CPU kernels if shape doesn't match

---

## 5. Graph Splitting for Multi-Backend Execution

### Graph Split Algorithm
**Location**: `ggml/src/ggml-backend.cpp:912-1280` (`ggml_backend_sched_split_graph`)

The scheduler creates **subgraphs** for each backend:

```cpp
// Pass 1: Assign backends to all operations
for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    int * node_backend_id = &tensor_backend_id(node);

    if (*node_backend_id == -1) {
        *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);
    }
}

// Pass 2: Create splits when backend changes
struct ggml_backend_sched_split * cur_split = &sched->splits[0];
cur_split->backend_id = tensor_backend_id(first_node);
cur_split->i_start = 0;
cur_split->n_inputs = 0;

int cur_backend_id = cur_split->backend_id;

for (int i = 0; i < graph->n_nodes; i++) {
    struct ggml_tensor * node = graph->nodes[i];
    const int node_backend_id = tensor_backend_id(node);

    // Need new split if backend changed
    if (node_backend_id != cur_backend_id) {
        // Create new split
        cur_split = &sched->splits[++n_splits];
        cur_split->backend_id = node_backend_id;
        cur_split->i_start = i;
        cur_split->n_inputs = 0;
        cur_backend_id = node_backend_id;
    }

    // Track cross-backend dependencies
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct ggml_tensor * src = node->src[j];
        if (src && tensor_backend_id(src) != cur_backend_id) {
            cur_split->inputs[cur_split->n_inputs++] = src;  // Needs copy
        }
    }
}
```

**Example Split**:
```
Graph: [Node0-CPU_IMI] [Node1-CPU_IMI] [Node2-CPU] [Node3-CPU] [Node4-REF]

Splits:
  Split 0: backend=CPU_IMI, nodes=[0,1], inputs=[]
  Split 1: backend=CPU,     nodes=[2,3], inputs=[Node1_output]  <-- copy from CPU_IMI
  Split 2: backend=REF,     nodes=[4],   inputs=[Node3_output]  <-- copy from CPU
```

---

## 6. Thread Scheduling Within Backend

### Thread Work Distribution
**Location**: `ggml/src/ggml-cpu/ggml-cpu.c:2900-2941`

Each backend has its own **thread pool**. For CPU backend:

```cpp
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * tp = state->threadpool;

    set_numa_thread_affinity(state->ith);  // Set thread affinity

    struct ggml_compute_params params = {
        .ith       = state->ith,           // Thread ID (0, 1, 2, 3)
        .nth       = atomic_load(&tp->n_threads_cur),  // Total threads
        .wsize     = cplan->work_size,
        .wdata     = cplan->work_data,
        .threadpool= tp,
    };

    // ALL threads process ALL nodes
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        ggml_compute_forward(&params, node);  // Dispatch operation

        // Synchronize between nodes
        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(tp);
        }
    }

    return 0;
}
```

**Key Points**:
- **All threads process all nodes** sequentially
- **Within each node**, threads split the work (e.g., rows of matrix)
- **Barrier synchronization** after each node to ensure correctness

### Work Partitioning Within an Operation

**Example: Matrix Multiplication (MUL_MAT)**
**Location**: `ggml/src/ggml-cpu/ggml-cpu.c:1221-1413`

```cpp
void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params,
                                   struct ggml_tensor * dst) {
    const int ith = params->ith;  // Thread ID
    const int nth = params->nth;  // Total threads

    const int64_t nr0 = ne0;  // Output rows
    const int64_t nr1 = ne1;  // Output cols

    // Divide work into chunks
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;  // Rows per chunk
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;  // Cols per chunk

    // Each thread processes chunks
    int current_chunk = ith;  // Start with own chunk

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // Process this chunk
        ggml_compute_forward_mul_mat_one_chunk(params, dst,
            ir0_start, ir0_end, ir1_start, ir1_end);

        // Get next chunk (atomic work stealing)
        current_chunk = atomic_fetch_add(&tp->current_chunk, 1);
    }
}
```

**Thread Work Distribution**:
- Thread 0 processes chunks: 0, 4, 8, 12, ...
- Thread 1 processes chunks: 1, 5, 9, 13, ...
- Thread 2 processes chunks: 2, 6, 10, 14, ...
- Thread 3 processes chunks: 3, 7, 11, 15, ...

**Dynamic Work Stealing**: Faster threads can grab extra chunks if others are slow.

---

## 7. Operation Dispatch: IMI vs Standard CPU

### Dispatch Flow
**Location**: `ggml/src/ggml-cpu/ggml-cpu.c:1672-1820`

```cpp
static void ggml_compute_forward(struct ggml_compute_params * params,
                                  struct ggml_tensor * tensor) {
    // STEP 1: Check if extra buffer type (IMI/AMX/KleidiAI) handles this
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;  // IMI/AMX handled it
    }

    // STEP 2: Use standard CPU kernels
    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            ggml_compute_forward_mul_mat(params, tensor);
            break;
        case GGML_OP_ADD:
            ggml_compute_forward_add(params, tensor);
            break;
        // ... other operations ...
        default:
            GGML_ABORT("unknown op");
    }
}
```

### Extra Buffer Type Check (IMI/AMX/etc.)
**Location**: `ggml/src/ggml-cpu/traits.cpp:12-23`

```cpp
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params,
                                     struct ggml_tensor * op) {
    // Iterate through all extra buffer types (IMI, AMX, KleidiAI, etc.)
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

**Decision Tree**:
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

---

## 8. Complete Execution Flow

### End-to-End Path

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Model Loading (llama-model-loader.cpp)                  │
│    ├─> Load GGUF file                                       │
│    ├─> Allocate tensors in CPU_IMI buffer type             │
│    └─> Repack weights into IMI format (set_tensor)         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 2. Context Initialization (llama-context.cpp)              │
│    ├─> Register backends: [CPU, REF]                        │
│    ├─> Create backend scheduler                             │
│    └─> Reserve graph memory                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 3. Graph Construction (llama-graph.cpp)                    │
│    ├─> Build forward pass compute graph                     │
│    │   ├─> Input embedding                                  │
│    │   ├─> Attention layers (Q, K, V, O projections)        │
│    │   ├─> Feed-forward layers (up, gate, down)             │
│    │   └─> Output projection                                │
│    └─> Set input tensors (tokens, positions, etc.)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 4. Scheduling Phase (ggml-backend.cpp)                     │
│    ├─> Pass 1: Assign each operation to backend             │
│    │   ├─> Check tensor buffer locations                    │
│    │   ├─> Prefer backend where weights are                 │
│    │   └─> Result: Most ops → CPU, validation → REF         │
│    ├─> Pass 2: Split graph at backend boundaries            │
│    │   ├─> Create split for CPU ops                         │
│    │   ├─> Create split for REF ops (if any)                │
│    │   └─> Track cross-backend dependencies                 │
│    └─> Pass 3: Allocate tensors in backend buffers          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 5. Execution Phase (ggml-backend.cpp)                      │
│    └─> For each split:                                      │
│        ├─> Copy split inputs from source backend            │
│        ├─> Dispatch to backend->graph_compute()             │
│        └─> Synchronize if needed                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 6. Backend Execution (ggml-cpu.cpp)                        │
│    ├─> Create compute plan (thread allocation)              │
│    ├─> Allocate work buffer                                 │
│    └─> Compute graph with threadpool (4 threads)            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 7. Thread Dispatch (ggml-cpu.c)                            │
│    ├─> Wake/spawn worker threads                            │
│    ├─> Set thread affinity (NUMA-aware)                     │
│    └─> Each thread processes nodes sequentially             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 8. Operation Dispatch (per node)                           │
│    ├─> Check: Extra buffer type? (IMI/AMX)                  │
│    │   ├─> Yes: Use IMI kernel                              │
│    │   │   └─> ggml_imi_gemm_q4_0_q8_0_8x4(...)            │
│    │   └─> No: Use standard CPU kernel                      │
│    │       └─> ggml_vec_dot_q4_0_q8_0(...)                 │
│    └─> Barrier: Wait for all threads                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ 9. Kernel Execution (imi/opt-kernels.cpp)                  │
│    ├─> Thread 0: Processes rows 0-7                         │
│    ├─> Thread 1: Processes rows 8-15                        │
│    ├─> Thread 2: Processes rows 16-23                       │
│    └─> Thread 3: Processes rows 24-31                       │
│        └─> Use RISC-V Vector intrinsics (IMI extensions)    │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Key Insights

### Data Locality is King
- Operations run on the backend where **weights are stored**
- Minimizes cross-backend data transfers
- CPU_IMI buffer type ensures weights are in IMI-friendly format

### IMI is Not a Separate Backend
- CPU_IMI is an **extra buffer type** within the CPU backend
- Selection happens at **kernel dispatch time**, not scheduler time
- Transparent fallback to standard CPU if IMI kernel unavailable

### Thread Parallelism Strategy
- **Static work partitioning** with dynamic work stealing
- All threads process all operations (not operation-level parallelism)
- **Barrier synchronization** ensures correctness between operations

### Backend Priority System
- GPU backends have highest priority (if available)
- CPU backend is middle priority (most operations)
- REF backend is lowest priority (validation/fallback only)

### Efficient Multi-Backend Support
- Graph splits minimize cross-backend transfers
- Pipeline parallelism can overlap different splits
- Event-based synchronization between backends

---

## Summary

**Scheduler**: Centralized (`ggml_backend_sched`) managing priority-ordered backends

**Backend Selection**: Based on:
1. Tensor buffer location (data locality)
2. Weight tensor location (prefer same backend)
3. Operation support capabilities
4. Offloading hints (for GPU)

**IMI vs CPU**: IMI is a buffer type, not a backend. Selection happens via:
1. Weight buffer type check (CPU_IMI?)
2. Kernel availability check (tensor->extra)
3. Dispatch to IMI-optimized kernel or standard CPU kernel

**Threading**: Static work partitioning + dynamic work stealing across 4 threads, with barrier synchronization between operations

**Multi-Backend**: Graph split at backend boundaries, with explicit data copies for cross-backend dependencies

The design prioritizes **data locality**, **kernel extensibility** (via buffer types), and **thread-level parallelism** for maximum performance across diverse hardware.
