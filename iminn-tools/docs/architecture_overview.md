# IMINN-Tools Architecture Overview

This document provides a comprehensive overview of the IMINN-Tools repository architecture, functionality, and design patterns.

## Project Overview

IMINN-Tools (`iminnt`) is a sophisticated Python-based performance evaluation and profiling framework designed for AI/ML workloads. It provides a unified CLI interface for building, running, simulating, and profiling AI/ML frameworks across different architectures (RISC-V, x86, ARM) and compilers (Clang, GCC).

---

## 1. Project Structure

The repository follows this organization:

```
/home/linhu/repo/iminn-tools/
├── src/iminnt/           # Core Python package
│   ├── cli.py            # CLI entry point
│   ├── core.py           # Base class abstractions
│   ├── registry.py       # Dynamic class registration
│   ├── constants.py      # Path definitions, environment configs
│   ├── utils.py          # Shell execution, performance metrics
│   ├── xnnpack.py        # XNNPACK framework implementations
│   ├── llamacpp.py       # llama.cpp implementations
│   ├── litert.py         # TensorFlow Lite/LiteRT
│   ├── infra.py          # Simulation infrastructure (QEMU, Spike, etc.)
│   ├── trace_gen.py      # Trace/profile generation from simulation
│   ├── ssh.py            # Remote execution support
│   └── resources/        # Config templates, kernel lists
├── dev_env/              # All built code and binaries
│   ├── XNNPACK/
│   ├── llama.cpp/
│   ├── Pilos/            # Performance simulator
│   ├── csqemu-v9/        # Custom QEMU
│   ├── IMachines_Spike/  # RISC-V ISA simulator
│   ├── Arctic/           # Carbon simulation framework
│   ├── Permafrost/       # Simulation orchestrator
│   └── riscv-env/        # RISC-V toolchain
├── results/              # Simulation output
├── toolchain.cmake       # Generic cross-compilation
├── riscv-imi.cmake       # RISC-V with I-Machines extensions
└── pyproject.toml        # Package configuration
```

---

## 2. CLI Entry Point and Command Flow

### CLI Architecture (`src/iminnt/cli.py`)

The CLI uses `argparse` with subparsers for different commands:

**Command Execution Flow:**
1. User runs: `iminnt -t <target> <command> [options]`
2. CLI parser validates target name against registry
3. Target class instantiated: `get_class_by_name(args.target)()`
4. Command dispatched: `dev_fw.exec_cmd(args.subcommand, **kwargs)`
5. Base class routes to appropriate method (`_build()`, `_run()`, `_sim()`, etc.)

**Supported Commands:**
- `init`: Clone/copy source code
- `pull`: Update to latest code
- `checkout`: Switch branches
- `sync`: Sync forked repos with upstream
- `build`: Clean build from scratch
- `rebuild`: Incremental build
- `run`: Execute via QEMU or native
- `sim`: Simulate with performance profiling
- `simpoint`: Statistical sampling simulation
- `sweep`: Benchmark sweep execution
- `defaults`: List available default run configurations

---

## 3. Core Abstractions and Class Hierarchy

### Base Class: `BaseRunner` (`src/iminnt/core.py`)

The foundation of all targets, defining the common interface:

```python
class BaseRunner(ABC):
    @property
    def target(self) -> str          # Target name
    def root(self) -> Path           # Source directory
    def env(self) -> Dict            # Environment variables
    def deps(self) -> List[str]      # Dependencies
    def get_build_cmd(**kwargs)      # Build commands
    def rebuild_cmd                  # Incremental build
    def exec_cmd(cmd, **kwargs)      # Command dispatcher
```

### Specialized Base Classes

**1. `BenchSimRunner`** - For runnable/simulatable targets (AI frameworks):
- Extends `BaseRunner` with execution capabilities
- Key properties:
  - `remote_path`: Git clone URL or local path
  - `install_dir`, `build_dir`: Build artifacts
  - `default_bin`: Default executable
  - `default_runs`: Predefined test configurations
  - `use_qemu`: Whether to use QEMU emulation

- Key methods:
  - `init()`: Clone/copy source code
  - `build()`: CMake-based compilation
  - `run()`: Execute via QEMU or remote/native
  - `simulate()`: Run through Permafrost simulation pipeline
  - `simpoint()`: Statistical sampling simulation
  - `sweep_sim()`: Benchmark sweep (XNNPACK only)

**2. `SubRunner`** - Infrastructure components with parent directory:
- Used for components within compound targets
- Has `parent_root` and `sub_root` properties

**3. `CompoundRunner`** - Meta-targets orchestrating multiple sub-targets:
- Example: `psim` = [qemu, arctic, pilos, permafrost, spike, riscv-env, simpoint]
- Iteratively executes commands across all sub-runners

**4. `MultiRunner`** - Targets with multiple build variants:
- Manages multiple related `BaseRunner` instances
- Each subtarget can have different configurations

---

## 4. Target Registration System

### Registry Pattern (`src/iminnt/registry.py`)

Simple but powerful decorator-based registration:

```python
_class_registry = {}

@register_class(name)
def decorator(cls):
    _class_registry[name] = cls
    return cls

get_class_by_name(name)  # Retrieve by string
get_class_names()        # List all registered targets
```

**Benefits:**
- Dynamic target discovery
- No need to modify CLI code when adding targets
- Clean separation of concerns

---

## 5. Key Modules and Their Purposes

### `constants.py` - Configuration Central

Defines all paths, environment variables, and architecture configurations:

- **Paths**: `DEV_ENV_ROOT`, `IMI_SDK_ROOT`, `QEMU_USER_BIN`, `PERMAFROST_BIN`
- **Architecture Specs**:
  - `IMI_RISCV_ARCH`: Full RISC-V arch string with extensions
  - `RVV_RISCV_ARCH`: Standard RISC-V Vector
- **Environment Configs**: `IMI_ENV`, `RVV_ENV`, `NEOVERSE_ENV`, `AMX_ENV`, `NATIVE_ENV`
- **Remote Info**: Azure server configurations for remote execution

### `utils.py` - Shell Execution and Metrics

- `shell()`: Execute commands with real-time output logging
- `get_cycles()`: Extract cycle counts from simulation stats
- `get_perf_in_ns()`: Convert cycles to nanoseconds

### `ssh.py` - Remote Execution

`RemoteConn` class for SSH-based remote execution:
- Copy binaries to remote servers
- Execute commands remotely
- Retrieve results
- Used for native x86/ARM benchmarking on appropriate hardware

### `trace_gen.py` - Performance Profiling

Sophisticated trace analysis from simulation retires:
- Parse retired instruction logs
- Identify function calls/returns using ELF symbols
- Generate Chrome Trace Format JSON for flamegraph visualization
- Support DWARF debug info for source-level attribution
- Handle tail calls, indirect jumps, and complex control flow

---

## 6. AI/ML Framework Integrations

### XNNPACK (`src/iminnt/xnnpack.py`)

**Targets:**
- `xnnpack_imi`: RISC-V with I-Machines extensions
- `xnnpack_x86`: Native x86
- `xnnpack_amx`: x86 with AMX (Intel)
- `xnnpack_neoverse`: ARM Neoverse N2

**Features:**
- CMake-based build with toolchain files
- Kernel-specific testing (GEMM, IGEMM variants)
- Benchmark sweep capability
- Support for different quantization schemes (Q8, Q4)
- Automatic test generation from kernel lists

**Default Runs:**
- Generated dynamically from kernel files in `resources/xnnpack/`
- Include unit tests, benchmarks, and micro-benchmarks
- Support filtering by kernel type, dimensions

### llama.cpp (`src/iminnt/llamacpp.py`)

**Targets:**
- `llama_imi`: RISC-V with IMI extensions
- `llama_rvv`: Standard RISC-V Vector
- `llama_x86`: Native x86
- `llama_amx_bench`: x86 with AMX
- `llama_oryon`: Qualcomm Oryon ARM

**Features:**
- Model downloads during `init` (TinyLLama, SmolLM, etc.)
- Backend operation testing framework
- Support for various quantization formats (Q4_0, Q8_0, MXFP4, etc.)
- Benchmark modes for prefill/decode
- REF backend for validation (optional)

**Default Runs:**
- Operation tests: MUL_MAT, ADD, CONV, etc.
- Model inference: stories15M, SmolLM, etc.
- Performance benchmarks: `llama-bench` with different configs

### LiteRT/TFLite (`src/iminnt/litert.py`)

TensorFlow Lite implementation (less mature than XNNPACK/llama.cpp)

---

## 7. Simulation Infrastructure

### Component Overview

The simulation pipeline involves multiple tools working together:

1. **QEMU** (`qemu`): User-mode emulation with custom plugins
   - Generates execution traces
   - ROI (Region of Interest) markers
   - Memory operation tracking

2. **Spike** (`spike`): RISC-V ISA functional simulator
   - Provides accurate instruction semantics
   - Executes with pk (proxy kernel)

3. **Arctic** (`arctic`): Carbon simulation framework
   - Provides simulation infrastructure APIs

4. **Pilos** (`pilos`): Performance model
   - Microarchitecture simulation
   - Pipeline modeling
   - Cache simulation
   - Vector unit modeling (VLEN=128)

5. **Permafrost** (`permafrost`): Simulation orchestrator
   - Coordinates QEMU, Spike, and Pilos
   - Configuration management
   - Result aggregation

6. **Simpoint** (`simpoint`): Statistical sampling tool
   - Identifies representative execution intervals
   - Reduces simulation time for long-running workloads

### Simulation Configuration (`PERMAFROST_SIM_CFG` in `core.py`)

Key configuration parameters:
- **Trace sources**: QEMU plugins for live trace and memory
- **Execution**: Spike for functional simulation
- **Performance**: Pilos shared library
- **Pipeline config**:
  - VLEN=128 (vector length)
  - 12 execution ports
  - Vector/int dispatch limits
  - Memory pipeline configuration
- **ROI control**: `IMI_ROI_SIM` environment variable

### Simulation Workflow

```
1. User runs: iminnt -t xnnpack_imi sim -d <test>
2. Binary compiled with debug symbols (-g)
3. Permafrost config generated (iminnt.cfg)
4. QEMU executes binary with cosim plugin
5. Spike provides functional simulation
6. Pilos models microarchitecture
7. Results written to output directory:
   - stats/pilos_combined.stats (cycle counts)
   - retires.log (instruction trace, if enabled)
   - out_config.txt (final config)
   - trace.json (flamegraph, if requested)
```

---

## 8. Cross-Compilation System

### Toolchain Files

**`toolchain.cmake`** - Generic cross-compilation:
- Supports RISC-V, ARM, x86
- Environment-driven configuration
- Required variables: `CROSS_SYSROOT`, `CROSS_ARCH`, `CROSS_TRIPLE`, `CROSS_CPU`, `CROSS_COMP`
- Handles both Clang and GCC
- Static linking support
- Debug symbol control

**`riscv-imi.cmake`** - RISC-V specific:
- Simplified version for I-Machines extensions
- Direct architecture flags for IMI CPU variants

### Environment Configuration

Each architecture has a predefined environment dict in `constants.py`:

**IMI_ENV (RISC-V with I-Machines):**
```python
{
    "CROSS_ARCH": "riscv64",
    "CROSS_COMP": "clang",
    "CROSS_TRIPLE": "riscv64-unknown-linux-gnu",
    "CROSS_CPU": "rv64gcv_ximimce_zba_zbb...",  # Full arch string
    "CROSS_TOOLCHAIN": "/path/to/riscv/sdk",
    "CROSS_SYSROOT": "/path/to/sysroot",
    "CROSS_QEMU_PATH": "/path/to/qemu-riscv64"
}
```

---

## 9. Component Interactions

### High-Level Data Flow

```
User Command (iminnt -t xnnpack_imi sim -d test)
    ↓
CLI Parser (cli.py)
    ↓
Registry Lookup (registry.py) → XNNPACKIMI class
    ↓
BenchSimRunner.simulate()
    ↓
├─ Environment Setup (constants.py: IMI_ENV)
├─ Binary Location (core.py: _get_run_cmd)
├─ Config Generation (core.py: _init_sim_cfg)
│   └─ Template from PERMAFROST_SIM_CFG
├─ Permafrost Execution (shell command via utils.py)
│   ├─ QEMU (trace generation)
│   ├─ Spike (functional simulation)
│   └─ Pilos (performance modeling)
└─ Result Processing
    ├─ Cycle extraction (utils.py: get_cycles)
    ├─ Trace generation (trace_gen.py, if requested)
    └─ Logging (log_cfg.py)
```

### Build Flow

```
User: iminnt -t xnnpack_imi build
    ↓
XNNPACKIMI.build()
    ↓
├─ Check dependencies (riscv-env must be built)
├─ Generate build commands (get_build_cmd)
│   ├─ Kernel generation scripts
│   ├─ CMake configure with toolchain.cmake
│   │   └─ Environment: IMI_ENV
│   └─ CMake build
├─ Execute commands (utils.shell)
└─ Install to xnnpack-install-imi/
```

---

## 10. Key Capabilities

### 1. Multi-Architecture Support
- Cross-compile for RISC-V, ARM, x86
- Native builds when target=host
- Remote execution for hardware unavailable locally

### 2. Performance Analysis
- Cycle-accurate simulation via Pilos
- Instruction-level tracing
- Function-level profiling with flamegraphs
- Statistical sampling with SimPoint
- Benchmark sweeps with automated result collection

### 3. Extensibility
- Easy to add new targets via `@register_class`
- Flexible build system (CMake, custom scripts)
- Pluggable simulation configurations
- Remote execution support

### 4. Developer Workflow
- Source code in `dev_env/` for easy modification
- Incremental rebuilds
- Git integration (pull, checkout, sync)
- Pre-commit hooks for code quality

### 5. AI/ML Framework Coverage
- XNNPACK (primary focus): Quantized inference kernels
- llama.cpp: LLM inference
- TFLite/LiteRT: TensorFlow Lite models
- Extensible to IREE, ONNX Runtime, etc.

---

## 11. Component Relationships

1. **CLI → Registry → Target Classes**: Dynamic dispatch based on string names
2. **Target Classes → Constants**: Environment configuration per architecture
3. **Target Classes → Utils**: Shell execution, metric extraction
4. **BenchSimRunner → Simulation Infrastructure**: Permafrost orchestrates QEMU/Spike/Pilos
5. **Simulation → Trace Generation**: Post-process retires.log into flamegraphs
6. **Cross-Compilation → Toolchain Files**: CMake integration for multi-arch builds
7. **Remote Execution → SSH Module**: Copy binaries, run on remote hardware
8. **Compound Targets → Sub-Runners**: Hierarchical dependency management

---

## 12. Adding New Targets

When adding a new target:

1. Create a class in appropriate module (or new module if needed)
2. Extend `BenchSimRunner` for runnable/simulatable targets, `SubRunner` for infrastructure
3. Decorate with `@register_class("target_name")`
4. Implement required abstract properties: `target`, `root`, `remote_path` (if git clone)
5. Implement build methods: `get_build_cmd()`, optionally `rebuild_cmd`
6. For runnable targets: implement `get_default_cmds()`, `_run()`, `_sim()` if custom behavior needed
7. Import the new class in `src/iminnt/cli.py` to register it

---

## Conclusion

This system is designed to be a comprehensive, unified interface for AI/ML performance engineering across diverse hardware architectures, providing capabilities from source-to-silicon performance analysis with minimal manual configuration.
