# llama.cpp Threading Configuration Analysis

## Summary

**llama.cpp in this repository is configured to run SINGLE-THREADED** at multiple levels:

1. **CMake Build Configuration**: `GGML_DEFAULT_N_THREADS=1`
2. **Test Infrastructure**: `#define SINGLE_THREADED`
3. **Runtime Arguments**: All test commands use `-t 1` or `--threads 1`
4. **OpenMP Disabled**: `GGML_OPENMP=OFF`

---

## Detailed Analysis

### 1. CMake Build-Time Configuration

**Location:** `src/iminnt/llamacpp.py:165`

```python
base_args = {
    "CMAKE_BUILD_TYPE": "RelWithDebInfo",
    "BUILD_SHARED_LIBS": "OFF",
    "GGML_OPENMP": "OFF",              # ← OpenMP disabled
    "GGML_STATIC": "ON",
    "LLAMA_CURL": "OFF",
    "GGML_LLAMAFILE": "OFF",
    "GGML_CPU_REPACK": "ON",
    "GGML_DEFAULT_N_THREADS": "1"      # ← Default threads set to 1
}
```

**Verification in Built Binaries:**

```bash
# IMI build
$ grep GGML_DEFAULT_N_THREADS dev_env/llama.cpp/linux-build-imi-bench/CMakeCache.txt
GGML_DEFAULT_N_THREADS:STRING=1

# x86 build
$ grep GGML_DEFAULT_N_THREADS dev_env/llama.cpp/linux-build-x86-bench/CMakeCache.txt
GGML_DEFAULT_N_THREADS:STRING=1

# OpenMP verification
$ grep GGML_OPENMP dev_env/llama.cpp/linux-build-imi-bench/CMakeCache.txt
GGML_OPENMP:BOOL=OFF
```

**What This Means:**
- When no thread count is specified at runtime, llama.cpp defaults to **1 thread**
- In the standard llama.cpp source, `GGML_DEFAULT_N_THREADS` is normally **4**
- This repository overrides it to **1** for simulation/emulation purposes

---

### 2. Test Infrastructure (test-backend-ops)

**Location:** `dev_env/llama.cpp/tests/test-backend-ops.cpp:45-46`

```cpp
// When emulating/simulating, need to restrict number of threads to 1 for the time being
#define SINGLE_THREADED
#define NPERF_RUNS 1
```

**Impact on Code:**

```cpp
static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    {
        // parallel initialization
        #ifdef SINGLE_THREADED
            static const size_t n_threads = 1;  // ← Forced to 1
        #else
            static const size_t n_threads = std::thread::hardware_concurrency();
        #endif

        // ... rest of initialization code
    }
}
```

**What This Means:**
- The `test-backend-ops` binary is compiled with `SINGLE_THREADED` defined
- This forces all internal parallelization to use exactly **1 thread**
- Performance runs (`NPERF_RUNS`) are limited to **1 iteration**
- This is explicitly done for **emulation/simulation compatibility**

**Comment from Source Code:**
> "When emulating/simulating, need to restrict number of threads to 1 for the time being"

This suggests that:
- Multi-threading creates issues with QEMU emulation
- The Pilos simulator may not correctly handle multi-threaded execution
- Single-threading ensures deterministic, reproducible results

---

### 3. Runtime Command-Line Arguments

**Location:** `src/iminnt/llamacpp.py` (default_runs property)

Every predefined test command explicitly sets threads to **1**:

```python
# llama-bench tests
"stories_debug_q4_0": {
    "bin": f"{self.install_dir}/bin/llama-bench",
    "args": f"--threads 1 -ngl 0 ..."  # ← --threads 1
}

"stories_decode_bench": {
    "bin": f"{self.install_dir}/bin/llama-bench",
    "args": f"--threads 1 -ngl 0 ..."  # ← --threads 1
}

# llama-cli tests
"test_q4_0_stories": {
    "bin": f"{self.install_dir}/bin/llama-cli",
    "args": f"-t 1 -ngl 0 -n 32 ..."   # ← -t 1
}

# llama-batched-bench tests
"stories_q4_0_bbench": {
    "bin": f"{self.install_dir}/bin/llama-batched-bench",
    "args": f"--threads 1 --threads-batch 1 ..."  # ← Both set to 1
}
```

**Additional Flags:**
- `-ngl 0` or `--n-gpu-layers 0`: Disables GPU offloading (CPU-only execution)
- `--no-warmup`: Skips warmup runs for consistent benchmarking

---

### 4. Original llama.cpp Default

**Location:** `dev_env/llama.cpp/ggml/include/ggml.h:228-229`

```c
#ifndef GGML_DEFAULT_N_THREADS
    #define GGML_DEFAULT_N_THREADS 4  // ← Original default is 4
#endif
```

**Location:** `dev_env/llama.cpp/ggml/CMakeLists.txt:186`

```cmake
set(GGML_DEFAULT_N_THREADS "4" CACHE STRING "ggml: default number of threads to use")
```

**What This Means:**
- The upstream llama.cpp defaults to **4 threads**
- This repository explicitly overrides it to **1 thread** via CMake

---

## Why Single-Threaded?

Based on the code comments and configuration, the reasons are:

### 1. **Emulation Compatibility**
```cpp
// When emulating/simulating, need to restrict number of threads to 1 for the time being
```
- QEMU user-mode emulation may have issues with multi-threading
- Thread scheduling in emulation can be non-deterministic
- Race conditions could produce inconsistent results

### 2. **Simulation Accuracy**
- The Pilos performance simulator models a single-threaded processor
- Multi-threaded execution would complicate cycle-accurate simulation
- ROI (Region of Interest) markers need deterministic execution

### 3. **Reproducibility**
- Single-threaded execution produces deterministic results
- Critical for regression testing and performance comparisons
- Eliminates non-determinism from thread scheduling

### 4. **Simplicity**
- Easier to debug and trace execution
- Simpler performance analysis
- Clearer mapping between operations and cycles

---

## Implications for Performance Analysis

### What Gets Measured
When running `iminnt -t llama_imi sim -d test_q4_0_stories`:

1. **Single-threaded Performance**: You're measuring performance of **1 CPU core**
2. **Sequential Execution**: No thread-level parallelism
3. **Deterministic Timing**: Results are reproducible across runs
4. **Cache Behavior**: Simpler cache modeling with single thread

### What Doesn't Get Measured
- Multi-core scalability
- Thread synchronization overhead
- Parallel memory bandwidth utilization
- SIMD parallelism within operations (this is still measured!)

**Important Note:** SIMD vectorization (RVV instructions) is **still active**. Single-threaded doesn't mean scalar - it means one thread using vector instructions.

---

## How to Change Thread Count (If Needed)

### For Testing/Development

**Option 1: Override at Runtime**
```bash
# Run with custom thread count (will override default)
iminnt -t llama_imi run -b llama-cli -a "-t 4 -m models/stories15M-q4_0.gguf -n 32"
```

**Option 2: Modify CMake Configuration**

Edit `src/iminnt/llamacpp.py:165`:
```python
base_args = {
    # ...
    "GGML_DEFAULT_N_THREADS": "4"  # Change from "1" to "4"
}
```

Then rebuild:
```bash
iminnt -t llama_imi build
```

**Option 3: Remove SINGLE_THREADED from test-backend-ops**

Edit `dev_env/llama.cpp/tests/test-backend-ops.cpp:45`:
```cpp
// Comment out these lines:
// #define SINGLE_THREADED
// #define NPERF_RUNS 1
```

Then rebuild:
```bash
iminnt -t llama_imi rebuild
```

### For Production/Native Builds

The single-threaded configuration is primarily for **RISC-V simulation**. For native x86 builds used for performance comparison, you might want multi-threading:

```python
class LlamaCppX86(LlamaCppBase):
    def __init__(self):
        super().__init__(
            "llama_x86",
            "linux-build-x86",
            "llamacpp-x86-install",
            use_qemu=False
        )

    @property
    def cmake_args(self):
        args = super().cmake_args
        args["GGML_DEFAULT_N_THREADS"] = "4"  # Override for x86
        return args
```

---

## Conclusion

**Current Configuration:**
- ✅ Single-threaded execution at all levels
- ✅ Optimized for emulation/simulation compatibility
- ✅ Deterministic, reproducible results
- ✅ Simplified performance analysis

**Trade-offs:**
- ❌ Cannot measure multi-core scalability
- ❌ May underutilize modern multi-core CPUs for native runs
- ✅ Appropriate for cycle-accurate simulation
- ✅ Matches the single-core IMI processor model

This configuration is **intentional and appropriate** for the primary use case: evaluating RISC-V performance optimizations through simulation with Pilos.
