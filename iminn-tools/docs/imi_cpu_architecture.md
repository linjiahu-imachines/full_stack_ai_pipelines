# IMI to CPU Architectural Relationship

This document explains the architectural relationship between IMI and the CPU backend in llama.cpp from a code structure perspective.

---

## Executive Summary

**Key Finding:** IMI is **NOT a separate backend**. It is an **"extra buffer type"** that extends the CPU backend with optimized kernels for RISC-V custom instructions.

```
┌─────────────────────────────────────────────────────────────┐
│              CPU Backend (Single Backend)                   │
│                                                              │
│  ┌────────────────────┐      ┌──────────────────────────┐  │
│  │  Standard Buffer   │      │   Extra Buffer Types     │  │
│  │  Type: "CPU"       │      │  (Accelerator variants) │  │
│  └────────────────────┘      └──────────────────────────┘  │
│                               │                            │
│                               ├─► CPU_IMI (RISC-V IMI)    │
│                               ├─► CPU_KLEIDIAI (ARM)      │
│                               ├─► CPU_AMX (x86 AMX)       │
│                               ├─► CPU_REPACK (Generic)    │
│                               └─► CPU_SPACEMIT (RISC-V)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

All use the SAME backend: CPU
Different buffer types trigger different kernel selection
```

---

## Architectural Hierarchy

### 1. Backend vs Buffer Type

llama.cpp has a clear separation:

```
Backend (ggml_backend)
  ├─ Defines compute capabilities (CPU, CUDA, Metal, etc.)
  ├─ Owns thread pool
  └─ Provides compute graph execution

Buffer Type (ggml_backend_buffer_type)
  ├─ Defines memory allocation strategy
  ├─ Determines data layout/repacking
  └─ Triggers accelerator-specific kernels
```

**CPU Backend** has:
- **1 primary backend instance**: The CPU backend
- **Multiple buffer types**: Standard CPU, CPU_IMI, CPU_KLEIDIAI, CPU_AMX, etc.

### 2. Code Location Structure

```
ggml/src/ggml-cpu/
├── ggml-cpu.c/cpp          # CPU backend implementation
├── ggml-cpu.h              # CPU backend API
├── traits.h                # Extra buffer type interface
│
├── imi/                    # IMI extra buffer type
│   ├── imi.cpp             # Buffer type + optimized kernels
│   ├── imi.h               # Public API
│   ├── generic-kernels.cpp # Generic IMI implementations
│   └── opt-kernels.cpp     # Optimized IMI implementations
│
├── kleidiai/               # ARM KleidiAI buffer type
│   └── kleidiai.cpp
│
├── amx/                    # x86 AMX buffer type
│   └── amx.cpp
│
├── repack/                 # Generic repack buffer type
│   └── repack.cpp
│
└── spacemit/               # SpacemiT RISC-V buffer type
    └── ime.cpp
```

**Key Insight**: IMI code lives **inside** `ggml-cpu/` directory, not as a separate backend.

---

## How Extra Buffer Types Work

### Class Hierarchy

All extra buffer types inherit from `ggml::cpu::extra_buffer_type`:

**File**: `ggml/src/ggml-cpu/traits.h:27-32`
```cpp
class extra_buffer_type {
  public:
    virtual ~extra_buffer_type();
    virtual bool supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(const struct ggml_tensor * op) = 0;
};
```

Each accelerator implements this interface:

```cpp
// IMI implementation (ggml/src/ggml-cpu/imi/imi.cpp:961-1000)
namespace ggml::cpu::imi {
    class extra_buffer_type : ggml::cpu::extra_buffer_type {
        bool supports_op(...) override {
            // Returns true if operation can use IMI kernels
            if (op->op == GGML_OP_MUL_MAT &&
                op->src[0]->buffer->buft == ggml_backend_cpu_imi_buffer_type()) {
                return true;
            }
            return false;
        }

        tensor_traits * get_tensor_traits(...) override {
            // Returns IMI-specific kernel dispatcher
        }
    };
}

// Similar implementations for:
// - ggml::cpu::kleidiai::extra_buffer_type (ARM)
// - ggml::cpu::amx::extra_buffer_type (x86)
// - ggml::cpu::repack::extra_buffer_type (generic)
// - ggml::cpu::riscv64_spacemit::extra_buffer_type (SpacemiT)
```

---

## Registration and Discovery

### Buffer Type Registration

**File**: `ggml/src/ggml-cpu/ggml-cpu.cpp:60-84`

```cpp
std::vector<ggml_backend_buffer_type_t> ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts;

        // HBM (High Bandwidth Memory)
        #ifdef GGML_USE_CPU_HBM
            bufts.push_back(ggml_backend_cpu_hbm_buffer_type());
        #endif

        // ARM KleidiAI acceleration
        #ifdef GGML_USE_CPU_KLEIDIAI
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        #endif

        // ⭐ RISC-V IMI acceleration
        #if defined(GGML_USE_CPU_IMI) && defined(__riscv_ximimce)
            if (ggml_backend_cpu_imi_buffer_type()) {
                bufts.push_back(ggml_backend_cpu_imi_buffer_type());
            }
        #endif

        // Generic repack optimization
        #ifdef GGML_USE_CPU_REPACK
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        #endif

        return bufts;
    }();

    return bufts;
}
```

**Registration happens at compile time** based on CMake flags:
- `GGML_CPU_IMI=ON` → IMI support compiled in
- `__riscv_ximimce` → IMI instructions available at runtime

---

## IMI Buffer Type Implementation

### Buffer Type Structure

**File**: `ggml/src/ggml-cpu/imi/imi.cpp:1004-1023`

```cpp
ggml_backend_buffer_type_t ggml_backend_cpu_imi_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_imi = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cpu_imi_buffer_type_get_name,  // Returns "CPU_IMI"
            /* .alloc_buffer     = */ ggml_backend_cpu_imi_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_imi_buffer_type_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .is_host          = */ nullptr,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),  // ⭐ Uses CPU device!
        /* .context = */ new ggml::cpu::imi::extra_buffer_type(),
    };

    if (!ggml_imi_init()) {
        return nullptr;  // IMI disabled via env var
    }

    return &ggml_backend_cpu_buffer_type_imi;
}
```

**Critical**: Notice `.device = ggml_backend_cpu_reg()` - **IMI uses the CPU backend device!**

### Buffer Allocation

**File**: `ggml/src/ggml-cpu/imi/imi.cpp:930-943`

```cpp
static ggml_backend_buffer_t ggml_backend_cpu_imi_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {

    // ⭐ Allocate buffer using STANDARD CPU buffer type
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(
        ggml_backend_cpu_buffer_type(),  // Standard CPU allocation!
        size
    );

    if (buffer == nullptr) {
        return nullptr;
    }

    // Override buffer type to CPU_IMI
    buffer->buft = buft;

    // Use IMI-specific tensor initialization
    buffer->iface.init_tensor = ggml_backend_cpu_imi_buffer_init_tensor;
    buffer->iface.set_tensor   = ggml_backend_cpu_imi_buffer_set_tensor;

    return buffer;
}
```

**Key Insight**: IMI buffers use **standard CPU memory allocation**, but with IMI-specific tensor initialization.

---

## Kernel Selection Flow

### Runtime Dispatch

```
1. Model loads weights
   ↓
2. Scheduler assigns buffer type based on priority
   │  Priority order: IMI → KLEIDIAI → AMX → REPACK → Standard CPU
   ↓
3. Tensor allocated with buffer type (e.g., CPU_IMI)
   ↓
4. During compute graph build:
   │  - Check tensor->buffer->buft
   │  - If buft == CPU_IMI:
   │    └─► Query extra_buffer_type::supports_op()
   ↓
5. If supports_op() returns true:
   │  └─► Use tensor_traits->compute_forward() (IMI kernels)
   │
   Else:
   └─► Use standard CPU kernels
```

### Operation Support Check

**File**: `ggml/src/ggml-cpu/imi/imi.cpp:962-990`

```cpp
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        // Check for MUL_MAT operation
        if (op->op == GGML_OP_MUL_MAT &&
            op->src[0]->buffer &&
            (ggml_n_dims(op->src[0]) == 2) &&
            op->src[0]->buffer->buft == ggml_backend_cpu_imi_buffer_type() &&  // ⭐ Check buffer type
            ggml_imi_get_optimal_repack_type(op->src[0])  // ⭐ Check if IMI kernel exists
        ) {
            // Additional checks for input tensor
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;  // ⭐ Use IMI kernel
            }
        }

        // Check for MUL_MAT_ID (MoE operation)
        else if (op->op == GGML_OP_MUL_MAT_ID && ...) {
            return true;
        }

        return false;  // Fall back to standard CPU
    }
};
```

---

## Comparison with Other Accelerators

### All Extra Buffer Types Follow Same Pattern

| Buffer Type | Architecture | Purpose | Kernel Location |
|-------------|-------------|---------|-----------------|
| **CPU_IMI** | RISC-V | I-Machines custom instructions | `ggml-cpu/imi/*.cpp` |
| **CPU_KLEIDIAI** | ARM | KleidiAI acceleration library | `ggml-cpu/kleidiai/*.cpp` |
| **CPU_AMX** | x86 | Intel AMX instructions | `ggml-cpu/amx/*.cpp` |
| **CPU_REPACK** | Generic | Data layout optimization | `ggml-cpu/repack.cpp` |
| **CPU_SPACEMIT** | RISC-V | SpacemiT IME instructions | `ggml-cpu/spacemit/*.cpp` |

**All share**:
- Same CPU backend
- Same thread pool
- Same memory allocation (host memory)
- Same scheduler
- **Only differ in kernel implementations**

---

## Memory and Execution Architecture

### Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Memory (RAM)                         │
│                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ Standard CPU │   │   CPU_IMI    │   │ CPU_KLEIDIAI │    │
│  │   Tensors    │   │   Tensors    │   │   Tensors    │    │
│  │              │   │              │   │              │    │
│  │ (Generic     │   │ (Repacked    │   │ (Repacked    │    │
│  │  layout)     │   │  for IMI)    │   │  for ARM)    │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   CPU Backend Execution                      │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Thread Pool (4 threads)                   │    │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   │    │
│  │  │Thread 0│  │Thread 1│  │Thread 2│  │Thread 3│   │    │
│  │  └────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘   │    │
│  └───────┼──────────┼──────────┼──────────┼─────────┘    │
│          │          │          │          │                │
│  ┌───────▼──────────▼──────────▼──────────▼─────────┐    │
│  │         Kernel Dispatcher (runtime)              │    │
│  │  if (buft == CPU_IMI) → use IMI kernels          │    │
│  │  else → use standard CPU kernels                 │    │
│  └──────┬───────────────────────────────────────────┘    │
│         │                                                  │
│  ┌──────▼─────────┐       ┌──────────────┐              │
│  │  IMI Kernels   │       │ Standard CPU │              │
│  │  (RISC-V IMI)  │       │   Kernels    │              │
│  │                │       │  (Generic)   │              │
│  │ • Q4_0 GEMM    │       │ • Vec dot    │              │
│  │ • Q8_0 GEMM    │       │ • Mat mul    │              │
│  │ • MXFP8 GEMM   │       │ • Quantize   │              │
│  └────────────────┘       └──────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Execution Flow with IMI

```
Step 1: Model Loading
  ├─ Weights loaded from GGUF file
  ├─ Scheduler checks available buffer types
  └─ CPU_IMI selected for Q4_0/Q8_0 weights (if available)

Step 2: Tensor Allocation
  ├─ Allocate memory via standard CPU allocator
  ├─ Tag buffer with buft = CPU_IMI
  └─ Call ggml_backend_cpu_imi_buffer_init_tensor()
      └─ Repack weights into IMI-optimized layout

Step 3: Graph Building
  ├─ For each MUL_MAT operation:
  │   ├─ Check if weight tensor has buft == CPU_IMI
  │   ├─ Query supports_op() → returns true
  │   └─ Attach IMI tensor_traits to operation
  └─ Graph ready for execution

Step 4: Execution
  ├─ Thread pool dispatches work
  ├─ Each thread calls compute_forward()
  │   ├─ Check if tensor has traits (extra accelerator)
  │   ├─ If yes → call tensor_traits->compute_forward()
  │   │   └─ Executes IMI kernel (e.g., ggml_imi_mul_mat_q4_0)
  │   └─ If no → call standard CPU kernel
  └─ Results written back to output tensor
```

---

## Key Architectural Principles

### 1. Single Backend, Multiple Variants

IMI is **not a backend**, it's a **variant of the CPU backend**:

```
Traditional (Wrong) View:
Backend: CPU        Backend: IMI
  ↓                    ↓
CPU kernels         IMI kernels

Actual (Correct) View:
Backend: CPU
  ├─ Standard buffer type → CPU kernels
  └─ IMI buffer type → IMI kernels
```

### 2. Buffer Type = Kernel Selector

The buffer type acts as a **compile-time and runtime switch**:

```cpp
// Compile time (CMake)
if (GGML_CPU_IMI)
    compile imi/*.cpp
endif

// Runtime (kernel dispatch)
if (tensor->buffer->buft == CPU_IMI)
    use IMI kernel
else
    use standard kernel
```

### 3. Shared Infrastructure

IMI shares **all** CPU backend infrastructure:

| Component | Shared? | Notes |
|-----------|---------|-------|
| Thread pool | ✅ Yes | Same `ggml_threadpool` |
| Memory allocator | ✅ Yes | Standard `malloc`/`posix_memalign` |
| Scheduler | ✅ Yes | Same `ggml_backend_sched` |
| Graph builder | ✅ Yes | Same compute graph |
| Synchronization | ✅ Yes | Same barriers |
| Work distribution | ✅ Yes | Same partitioning algorithm |
| **Kernels** | ❌ No | IMI-specific implementations |
| **Data layout** | ❌ No | IMI-specific repacking |

### 4. Fallback Mechanism

If IMI kernel not available:
```
Operation request
  ↓
supports_op() → false
  ↓
Use standard CPU kernel
```

No errors, seamless fallback.

---

## Comparison: IMI vs Other Backends (GPU)

### True Separate Backend (CUDA/Metal)

```
Backend: CUDA
  ├─ Separate device (GPU)
  ├─ Separate memory (VRAM)
  ├─ Separate thread model (CUDA threads/blocks)
  ├─ Separate scheduler
  └─ Requires data transfers (CPU ↔ GPU)

Backend: CPU + IMI buffer type
  ├─ Same device (CPU)
  ├─ Same memory (RAM)
  ├─ Same thread model (pthread/std::thread)
  ├─ Same scheduler
  └─ No data transfers (already in host memory)
```

### Why IMI is Not a Separate Backend

**Technical reasons**:
1. **No separate execution context**: IMI kernels run in the same thread pool as CPU
2. **No separate memory space**: IMI uses host memory, not device memory
3. **No device synchronization needed**: Everything happens on CPU
4. **No cross-backend transfers**: Tensors stay in place
5. **Compile-time selectable**: IMI is optional CPU variant, not independent backend

**Architectural reasons**:
1. **Code organization**: Lives under `ggml-cpu/` directory
2. **Registration**: Registered as "extra buffer type" of CPU
3. **Device assignment**: `device = ggml_backend_cpu_reg()`
4. **Interface**: Implements `extra_buffer_type`, not `ggml_backend_i`

---

## Build System Integration

### CMake Configuration

**File**: `ggml/src/ggml-cpu/CMakeLists.txt:466-477`

```cmake
if (GGML_CPU_IMI)
    # Add _ximimce to architecture string
    string(APPEND MARCH_STR "_ximimce")

    # Define preprocessor macro
    add_compile_definitions(GGML_USE_CPU_IMI)

    # Add IMI source files to CPU backend
    list(APPEND GGML_CPU_SOURCES
        ggml-cpu/imi/imi.cpp
        ggml-cpu/imi/generic-kernels.cpp
        ggml-cpu/imi/opt-kernels.cpp
    )
endif()
```

**Note**: IMI files are added to `GGML_CPU_SOURCES`, not a separate target.

### Compilation Flags

For IMI builds:
```bash
-DGGML_CPU_IMI=ON              # Enable IMI support
-march=rv64gc_zfh_v_zvfh_ximimce  # Include IMI ISA extension
-DGGML_USE_CPU_IMI             # Preprocessor define
```

At runtime:
```cpp
#if defined(GGML_USE_CPU_IMI) && defined(__riscv_ximimce)
    // IMI code compiled AND IMI instructions available
    register_imi_buffer_type();
#endif
```

---

## Runtime Behavior

### Priority-Based Selection

When a model loads, the scheduler tries buffer types in order:

**File**: `ggml/src/ggml-cpu/ggml-cpu.cpp:60-84` (registration order = priority order)

```
1. CPU_HBM          (if GGML_USE_CPU_HBM)
2. CPU_KLEIDIAI     (if GGML_USE_CPU_KLEIDIAI)
3. CPU_IMI          (if GGML_USE_CPU_IMI && __riscv_ximimce)
4. CPU_REPACK       (if GGML_USE_CPU_REPACK)
5. Standard CPU     (always available)
```

**Selection algorithm**:
```
For each weight tensor:
  for buffer_type in extra_buffer_types:
    if buffer_type.supports_op(tensor.op):
      allocate with buffer_type
      break
  else:
    allocate with standard CPU buffer type
```

### Example: Q4_0 Weight Matrix

```
Model: stories15M-q4_0.gguf
Weight: feed_forward.w1 (Q4_0, 2D matrix)

┌─────────────────────────────────────────┐
│ Scheduler checks buffer types:          │
├─────────────────────────────────────────┤
│ 1. CPU_IMI.supports_op(MUL_MAT)?        │
│    → Check: tensor type == Q4_0? ✓      │
│    → Check: IMI kernel exists? ✓        │
│    → Result: TRUE                        │
│    → Allocate with CPU_IMI               │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ Buffer allocated:                        │
│  - Memory: Host RAM (standard malloc)   │
│  - Tag: buft = CPU_IMI                  │
│  - Init: Repack to IMI layout           │
│  - Extra: tensor_traits = IMI dispatcher│
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ During inference:                        │
│  - Operation: MUL_MAT                    │
│  - Check: tensor->extra != nullptr? ✓    │
│  - Call: tensor_traits->compute_forward()│
│  - Execute: ggml_imi_mul_mat_q4_0()     │
│  - Using: IMI custom instructions        │
└─────────────────────────────────────────┘
```

---

## Summary: IMI ↔ CPU Relationship

### Conceptual Model

```
┌────────────────────────────────────────────────────────┐
│                    CPU Backend                         │
│                                                         │
│  Base Capabilities:                                    │
│  ├─ Thread pool management                            │
│  ├─ Memory allocation                                 │
│  ├─ Scheduler integration                             │
│  ├─ Standard compute kernels                          │
│  └─ Graph execution                                   │
│                                                         │
│  Extensions (Extra Buffer Types):                      │
│  ├─ CPU_IMI (RISC-V custom instructions)              │
│  │   ├─ Optimized Q4_0/Q8_0 GEMM                      │
│  │   ├─ MXFP8 support                                 │
│  │   └─ IMI-specific data layouts                     │
│  │                                                      │
│  ├─ CPU_KLEIDIAI (ARM acceleration)                   │
│  │   └─ ARM-specific kernels                          │
│  │                                                      │
│  ├─ CPU_AMX (Intel AMX)                               │
│  │   └─ x86 AMX tile operations                       │
│  │                                                      │
│  └─ CPU_REPACK (Generic optimization)                 │
│      └─ Data layout optimizations                     │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### Key Relationships

| Aspect | Relationship |
|--------|-------------|
| **Hierarchy** | IMI is a **child component** of CPU backend |
| **Code location** | IMI lives **inside** `ggml-cpu/` directory |
| **Memory** | IMI uses **same memory** as CPU (host RAM) |
| **Execution** | IMI runs on **same threads** as CPU |
| **Scheduler** | IMI uses **same scheduler** as CPU |
| **Registration** | IMI registered as **extra buffer type** |
| **Device** | IMI points to **CPU device** (`ggml_backend_cpu_reg()`) |
| **Kernels** | IMI provides **alternative implementations** |
| **Selection** | IMI chosen at **runtime** based on tensor buffer type |
| **Fallback** | If IMI unavailable → **seamlessly use CPU kernels** |

### Analogy

Think of it like a car with different engine modes:

```
Car = CPU Backend
├─ Eco Mode = Standard CPU kernels
├─ Sport Mode = IMI kernels (turbo with custom instructions)
├─ Comfort Mode = REPACK kernels (optimized layout)
└─ Performance Mode = KLEIDIAI/AMX kernels (platform-specific)

All modes:
- Use the same chassis (thread pool)
- Use the same fuel (host memory)
- Use the same transmission (scheduler)
- Just different engine tuning (kernel implementations)
```

IMI is **NOT a different car** (separate backend), it's a **different driving mode** (buffer type variant) of the same car (CPU backend).

---

## Further Reading

- **Scheduler Architecture**: See `docs/llamacpp_scheduler_architecture.md` for how the scheduler assigns buffer types
- **Multi-threading**: See `docs/llamacpp_multithreading_changes.md` for how threads are shared across all buffer types
- **Build System**: See `docs/llamacpp_lifecycle.md` for how IMI support is compiled into the CPU backend

---

**Document Version**: 1.0
**Last Updated**: 2025-12-08
**Author**: Claude Code Analysis
