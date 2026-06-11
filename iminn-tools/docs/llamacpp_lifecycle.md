# llama.cpp Lifecycle in iminn-tools

This document provides a comprehensive analysis of how iminn-tools manages the complete lifecycle of llama.cpp: initialization, building, installation, execution, and simulation.

---

## Table of Contents

1. [Initialization (Clone & Setup)](#1-initialization-clone--setup)
2. [Build Process](#2-build-process)
3. [Installation](#3-installation)
4. [Runtime Execution](#4-runtime-execution)
5. [Simulation & Profiling](#5-simulation--profiling)
6. [Key Implementation Details](#6-key-implementation-details)

---

## 1. Initialization (Clone & Setup)

### Command
```bash
iminnt -t llama_imi init
```

### Entry Point Flow

**CLI Entry**: `src/iminnt/cli.py:97`
```python
dev_fw = get_class_by_name(args.target)()  # Creates LlamaCppIMI instance
dev_fw.exec_cmd(args.subcommand, **kwargs)  # Calls exec_cmd("init", ...)
```

**Command Routing**: `src/iminnt/core.py:209-233`
```python
def exec_cmd(self, cmd: str, **kwargs):
    if cmd == "init":
        self.init(**kwargs)
```

### LlamaCppIMI Class Structure

**File**: `src/iminnt/llamacpp.py:382-399`

```python
@register_class("llama_imi")
class LlamaCppIMI(LlamaCppBase):
    def __init__(self):
        super().__init__(
            "llama_imi",
            "linux-build-imi",              # Build directory
            "llamacpp-imi-install",          # Install directory
            env=IMI_ENV                      # RISC-V IMI environment vars
        )
```

### Initialization Steps

**Implementation**: `src/iminnt/core.py:321-379`

#### Step 1: Clone Repository (`core.py:345-350`)

```python
# Clone I-Machines fork of llama.cpp
shell(["git", "clone", "--progress", "--filter=blob:none",
       self.remote_path,  # https://github.com/I-Machines/llama.cpp
       f"{self.root}"])   # dev_env/llama.cpp

# Initialize submodules
shell(["git", "-C", f"{self.root}", "submodule", "update", "--init", "--progress"])
```

**Remote Path**: `src/iminnt/llamacpp.py:34-35`
```python
@property
def remote_path(self):
    return "https://github.com/I-Machines/llama.cpp"
```

#### Step 2: Post-Init Actions (`core.py:376-378`)

**Post-init commands**: `src/iminnt/llamacpp.py:42-57`

1. **Add upstream remote** (`llamacpp.py:44`):
   ```bash
   git -C {self.root} remote add upstream https://github.com/ggml-org/llama.cpp
   ```

2. **Create upstream branch** (`llamacpp.py:45-46`):
   ```bash
   git -C {self.root} checkout upstream
   git -C {self.root} checkout main
   ```

3. **Download models** (`llamacpp.py:47-55`):

   Multiple `wget` commands download pre-quantized GGUF models to `{self.root}/models/`:

   - `stories15M-q4_0.gguf` (4-bit quantization, 19MB)
   - `stories15M-q8_0.gguf` (8-bit quantization, 26MB)
   - `stories15M.gguf` (full precision, 98MB)
   - `SmolLM2-135M.F16.gguf` (FP16, 270MB)
   - `stories15M_MOE-F16.gguf` (Mixture of Experts, 73MB)
   - `lille-130m-instruct-f32.gguf` (FP32, 509MB)

### Upstream Sync Support

**Upstream Remote**: `src/iminnt/llamacpp.py:60-61`
```python
@property
def upstream_remote(self) -> Optional[str]:
    return "https://github.com/ggml-org/llama.cpp"
```

This enables syncing with official llama.cpp repository:
```bash
iminnt -t llama_imi sync
```

---

## 2. Build Process

### Command
```bash
iminnt -t llama_imi build
```

### Build Execution Flow

**Entry Point**: `src/iminnt/core.py:413-426`

```python
def build(self, threads: Optional[int] = None):
    self.check_deps(True)           # Verify dependencies (riscv-env)
    self.check_exists()             # Verify source exists
    env = dict(BASE_ENV, **self.env)  # Merge environments
    build_cmd = self.get_build_cmd(threads=threads)
    for bcmd in build_cmd:
        shell(bcmd.split(), env=env, cwd=self.root_str)
```

### Build Commands

**Implementation**: `src/iminnt/llamacpp.py:95-105`

The build process consists of 5 sequential commands:

```python
def get_build_cmd(self, threads=None) -> list:
    threads = threads if threads is not None else ""
    cmake_args = " ".join([f"-D{k}={v}" for k,v in self.cmake_args.items()])

    return [
        f"rm -rf {self.build_dir}",      # 1. Clean build directory
        f"rm -rf {self.install_dir}",    # 2. Clean install directory
        f"cmake -G Ninja -S {self.root} -B {self.build_dir} {cmake_args}",  # 3. Configure
        f"cmake --build {self.build_dir} -j{threads}",                      # 4. Build
        f"cmake --build {self.build_dir} -j{threads} -t install"            # 5. Install
    ]
```

### Directory Paths

**Implementation**: `src/iminnt/llamacpp.py:84-93`

```python
@property
def build_dir(self) -> Path:
    return self.root / self._build_dir
    # dev_env/llama.cpp/linux-build-imi

@property
def install_dir(self) -> Path:
    return self.root / self._install_dir
    # dev_env/llama.cpp/llamacpp-imi-install

@property
def root(self) -> str:
    return DEV_ENV_ROOT / "llama.cpp"
    # dev_env/llama.cpp
```

### CMake Configuration

**Implementation**: `src/iminnt/llamacpp.py:153-199`

```python
@property
def cmake_args(self):
    base_args = {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "CMAKE_INSTALL_PREFIX": f"{self.install_dir}",
        "BUILD_SHARED_LIBS": "OFF",
        "GGML_OPENMP": "OFF",
        "GGML_STATIC": "ON",
        "LLAMA_CURL": "OFF",
        "GGML_LLAMAFILE": "OFF",
        "GGML_CPU_REPACK": "ON",
        "GGML_DEFAULT_N_THREADS": "1"
    }

    # Add feature flags (none for base llama_imi)
    for f in self.features:
        base_args[f"GGML_{f}"] = "ON"

    # Cross-compilation setup for RISC-V/ARM
    if self.is_riscv or self.is_aarch64:
        base_args["CROSS_STATIC"] = "ON"
        base_args["CMAKE_TOOLCHAIN_FILE"] = f"{CROSS_TOOLCHAIN_PATH}"

    # RISC-V specific configuration
    if self.is_riscv:
        base_args["GGML_NATIVE"] = "OFF"
        base_args["GGML_RV_ZFH"] = "ON"      # Half-precision float (Zfh)
        base_args["GGML_RV_ZVFH"] = "ON"     # Vector half-precision (Zvfh)
        base_args["GGML_RV_ZICBOP"] = "ON"   # Cache-block prefetch
        base_args["GGML_RVV"] = "ON"         # RISC-V Vector extension
        base_args["GGML_RVV_VLEN"] = "128"   # Vector length = 128 bits

    # IMI-specific extensions
    if self.is_imi:
        base_args["GGML_CPU_IMI"] = "ON"     # Enable IMI custom instructions

    return base_args
```

### Cross-Compilation Environment

**Environment Variables**: `src/iminnt/constants.py:62-73`

For `llama_imi`, the `IMI_ENV` dictionary provides cross-compilation configuration:

```python
IMI_ENV = {
    "CROSS_ARCH": "riscv64",
    "CROSS_COMP": "clang",
    "CROSS_TRIPLE": "riscv64-unknown-linux-gnu",
    "CROSS_CPU": IMI_RISCV_ARCH.lower(),  # Full ISA string
    "CROSS_TOOLCHAIN": f"{IMI_SDK_ROOT}",
    "CROSS_SYSROOT": f"{IMI_SDK_ROOT / 'sysroot'}",
    "IMI_LLVM_PATH": f"{IMI_SDK_ROOT}",
    "CROSS_QEMU_PATH": f"{DEV_ENV_ROOT}/csqemu-v9/install-local/bin/qemu-riscv64",
    "RVV_VLEN": "128"
}
```

**RISC-V ISA String**: `src/iminnt/constants.py:54`

```python
IMI_RISCV_ARCH = "RV64GCV_ximimce_zba_zbb_zbs_zicntr_zihpm_zihintpause_zicbom_zicbop_zicboz_zihintntl_zicond_zcb_zfa_zawrs_zfh_zvfh_zfhmin_zvbb_zimop_zcmop_zfbfmin_zvfbfwma"
```

This extensive ISA string includes:
- **RV64GCV**: Base 64-bit RISC-V with General, Compressed, Vector
- **ximimce**: IMI custom extensions (machine learning accelerators)
- **zba/zbb/zbs**: Bit manipulation extensions
- **zfh/zvfh**: Half-precision float support (critical for ML)
- **zicbop**: Cache block prefetch operations

### Toolchain File Integration

**File**: `toolchain.cmake`

The toolchain file is triggered when `CMAKE_TOOLCHAIN_FILE` is set in cmake_args (line 172 in llamacpp.py).

**Key Sections**:

1. **Environment Variable Extraction** (lines 29-40):
   ```cmake
   get_cross_var(CROSS_SYSROOT REQUIRED)
   get_cross_var(IMI_LLVM_PATH REQUIRED)
   get_cross_var(CROSS_ARCH REQUIRED)
   get_cross_var(CROSS_TRIPLE REQUIRED)
   get_cross_var(CROSS_CPU REQUIRED)
   ```

2. **RISC-V Architecture Flags** (lines 77-86):
   ```cmake
   elseif("${CROSS_ARCH}" STREQUAL "riscv64")
       if ("${CROSS_CPU}" MATCHES "rv64")
           list(APPEND CROSS_ARCH_FLAGS_TMP "-march=${CROSS_CPU}")
           if(DEFINED RVV_VLEN)
               list(APPEND CROSS_ARCH_FLAGS_TMP "-mrvv-vector-bits=${RVV_VLEN}")
           endif()
       endif()
       list(APPEND CROSS_ARCH_FLAGS_TMP "-mabi=lp64d")
   ```

3. **Clang Compiler Setup** (lines 94-130):
   ```cmake
   if("${CROSS_COMP}" STREQUAL "clang")
       set(CMAKE_C_COMPILER "${IMI_LLVM_PATH}/bin/clang")
       set(CMAKE_CXX_COMPILER "${IMI_LLVM_PATH}/bin/clang++")
       set(CMAKE_AR "${IMI_LLVM_PATH}/bin/llvm-ar")
       set(CMAKE_RANLIB "${IMI_LLVM_PATH}/bin/llvm-ranlib")
       # ... other LLVM tools
   ```

4. **Static Linking** (lines 201-218):
   ```cmake
   if(CROSS_STATIC)
       set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
       set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
   ```

### Build Artifacts

After a successful build:

**Build directory**: `dev_env/llama.cpp/linux-build-imi/`
- Object files and intermediate build artifacts
- CMake cache and configuration files

**Install directory**: `dev_env/llama.cpp/llamacpp-imi-install/`
```
llamacpp-imi-install/
├── bin/          # 67 executable binaries
│   ├── llama-cli
│   ├── llama-bench
│   ├── test-backend-ops
│   ├── llama-batched-bench
│   └── ... (63 more binaries)
├── include/      # Header files for libllama
└── lib/          # Static/shared libraries
```

---

## 3. Installation

### Installation Process

Installation happens automatically during the build phase via CMake:

**Build Command 5**: `src/iminnt/llamacpp.py:104`
```python
f"cmake --build {self.build_dir} -j{threads} -t install"
```

This executes the CMake `install` target, which:
1. Copies built binaries to `{install_dir}/bin/`
2. Copies header files to `{install_dir}/include/`
3. Copies static libraries to `{install_dir}/lib/`

### Installation Directory

**Property**: `src/iminnt/llamacpp.py:84-85`
```python
@property
def install_dir(self) -> Path:
    return self.root / self._install_dir
```

For `llama_imi`: `dev_env/llama.cpp/llamacpp-imi-install/`

### Installed Binaries

67 binaries are installed, including:

**Primary Executables**:
- `llama-cli`: Main interactive CLI for text generation
- `llama-bench`: Benchmarking tool for performance testing
- `llama-batched-bench`: Batch processing benchmarks
- `test-backend-ops`: Backend operations testing (MUL_MAT, ADD, etc.)

**Supporting Tools**:
- `llama-embedding`: Generate embeddings from text
- `llama-imatrix`: Importance matrix generation for quantization
- `llama-quantize`: Model quantization tool
- `llama-server`: HTTP server for inference
- `llama-simple`: Minimal example
- `convert_hf_to_gguf.py`: HuggingFace to GGUF conversion

**Test Binaries** (40+ test executables):
- `test-backend-ops`: Operation performance testing
- `test-chat-template`: Chat template testing
- `test-json-schema-to-grammar`: JSON schema validation
- `test-tokenizer-*`: Various tokenizer tests
- And many more...

All binaries are statically linked RISC-V executables (due to `CROSS_STATIC=ON` and `GGML_STATIC=ON`).

---

## 4. Runtime Execution

### Command
```bash
iminnt -t llama_imi run -d <default_cmd>
# or
iminnt -t llama_imi run -b <binary_name> -a "<args>"
```

### Execution Flow

**Entry Point**: `src/iminnt/core.py:556-610`

```python
def run(self, bin_name: str = None, default_cmd: Optional[str] = None,
        bin_args: Optional[List[str]] = None, icount: bool = False):

    # Step 1: Resolve command information
    cmd_info = self._get_run_cmd(bin_name, default_cmd, bin_args)

    # Step 2: Extract command details
    run_env = cmd_info["env"]
    bin_cmd = cmd_info["cmd"]
    bin_path = cmd_info["bin_path"]

    # Step 3: Execute via QEMU (for RISC-V)
    if self.use_qemu:
        qemu_bin = f"{QEMU_USER_BIN} {QEMU_ROI_ENV_FLAG} -cpu {IMI_CPU_ALIAS}"

        # Check if binary needs dynamic linking
        if not self.is_static_bin(str(bin_path)):
            qemu_bin = f"{qemu_bin} -L {self.env['CROSS_SYSROOT']}"

        qemu_cmd = f"{qemu_bin} {bin_cmd}"
        shell(qemu_cmd.split(), env=run_env)
```

### Default Commands

**Implementation**: `src/iminnt/llamacpp.py:202-359`

The system provides 60+ pre-configured test commands. Here are key examples:

#### Operation Tests (test-backend-ops)

```python
"optest_help": {
    "bin": f"{self.install_dir}/bin/test-backend-ops",
    "args": "--help"
},

"opperf_mul_mat_bf16_prefill": {
    "bin": f"{self.install_dir}/bin/test-backend-ops",
    "args": "perf -o MUL_MAT -b CPU -p type_a=bf16,type_b=f32,m=128,n=32,k=256"
},

"opperf_mul_mat_q8_0_decode": {
    "bin": f"{self.install_dir}/bin/test-backend-ops",
    "args": "perf -o MUL_MAT -b CPU -p type_a=q8_0,type_b=f32,m=1,n=32,k=256"
},
```

**Key parameters**:
- `-o MUL_MAT`: Operation type (matrix multiplication)
- `-b CPU`: Backend (CPU, GPU, etc.)
- `-p`: Parameters (data types, dimensions)
- `m=128,n=32,k=256`: Matrix dimensions (M×K × K×N = M×N)

#### Benchmarking (llama-bench)

```python
"stories_prefill_bench": {
    "bin": f"{self.install_dir}/bin/llama-bench",
    "args": f"-v -m {self.root}/models/stories15M-q4_0.gguf "
            f"--repetitions 1 --threads 1 -ngl 0 -n 0 -p 0 -pg 8,0"
},

"stories_decode_bench": {
    "bin": f"{self.install_dir}/bin/llama-bench",
    "args": f"-v -m {self.root}/models/stories15M-q4_0.gguf "
            f"--repetitions 1 --threads 1 -ngl 0 -n 0 -p 0 -pg 0,32"
},
```

**Key parameters**:
- `-m`: Model path
- `--threads 1`: Single-threaded execution
- `-ngl 0`: No GPU layers (CPU only)
- `-pg 8,0`: Prefill 8 tokens, decode 0 (prefill benchmark)
- `-pg 0,32`: Prefill 0 tokens, decode 32 (decode benchmark)

#### Text Generation (llama-cli)

```python
"test_q8_0_stories": {
    "bin": f"{self.install_dir}/bin/llama-cli",
    "args": f"-m {self.root}/models/stories15M-q8_0.gguf "
            f"--seed 42 -t 1 -ngl 0 -n 32 "
            f"-no-cnv -st --no-warmup "
            f"--file {PROMPTS_DIR}/hello-world.txt"
},

"test_f16_smollm": {
    "bin": f"{self.install_dir}/bin/llama-cli",
    "args": f"-m {self.root}/models/SmolLM2-135M.F16.gguf "
            f"--seed 42 -t 1 -ngl 0 -n 32 "
            f"-no-cnv -st --no-warmup "
            f"--file {PROMPTS_DIR}/hello-world.txt"
},
```

**Key parameters**:
- `-m`: Model path
- `--seed 42`: Fixed random seed for reproducibility
- `-t 1`: Single thread
- `-ngl 0`: No GPU layers
- `-n 32`: Generate 32 tokens
- `-no-cnv`: No conversation mode
- `-st`: Special tokens
- `--no-warmup`: Skip warmup runs
- `--file`: Input prompt file

**Prompts Directory**: `src/iminnt/resources/prompts/`

Contains 9 prompt files:
- `hello-world.txt`: "Hello world today was a great day, and I hope for the future"
- `hello.txt`: Simple "Hello" prompt
- `chat-with-bob.txt`: Conversational prompt
- And more...

### Command Resolution

**Implementation**: `src/iminnt/core.py:427-466`

```python
def _get_run_cmd(self, bin_name: str = None, default_cmd: Optional[str] = None,
                 bin_args: Optional[List[str]] = None):

    if default_cmd:
        # Use pre-configured command
        di = self.default_runs[default_cmd]
        bin_name = di["bin"]
        bin_args = di.get("args", "")
        run_env = dict(BASE_ENV, **di.get("env", {}))

    else:
        # Resolve binary location (search order)
        if Path(bin_name).is_absolute():
            bin_path = Path(bin_name)
        elif (self.install_dir / bin_name).exists():
            bin_path = self.install_dir / bin_name
        elif (self.build_dir / bin_name).exists():
            bin_path = self.build_dir / bin_name
        elif (self.root / bin_name).exists():
            bin_path = self.root / bin_name

    bin_cmd = f"{bin_path} {bin_args}"
    return {"env": run_env, "cmd": bin_cmd, "bin_path": bin_path, ...}
```

### QEMU Integration

**Constants**: `src/iminnt/constants.py:16-30`

```python
QEMU_USER_DIR = QEMU_BASE_DIR / "install-local"  # csqemu-v9/install-local
QEMU_USER_BIN = QEMU_USER_DIR / "bin" / "qemu-riscv64"
```

**ROI (Region of Interest) Flag**: `src/iminnt/core.py:15`
```python
QEMU_ROI_ENV_FLAG = f"-E IMI_ROI_SIM=\"1\""
```

This environment variable enables ROI markers for focused profiling.

**CPU Alias**: `src/iminnt/constants.py:51`
```python
IMI_CPU_ALIAS = "imicpu-v1"
```

### Example Execution Command

For `iminnt -t llama_imi run -d test_q8_0_stories`:

```bash
/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-local/bin/qemu-riscv64 \
    -E IMI_ROI_SIM="1" \
    -cpu imicpu-v1 \
    /home/linhu/repo/iminn-tools/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
    -m /home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/stories15M-q8_0.gguf \
    --seed 42 -t 1 -ngl 0 -n 32 -no-cnv -st --no-warmup \
    --file /home/linhu/repo/iminn-tools/src/iminnt/resources/prompts/hello-world.txt
```

---

## 5. Simulation & Profiling

### Command
```bash
iminnt -t llama_imi sim -d <default_cmd> -o <output_dir>
```

### Simulation Overview

Simulation provides cycle-accurate performance analysis by running the RISC-V binary through a detailed microarchitecture model.

**Architecture**:
1. **QEMU**: Generates execution trace with memory operations
2. **Spike**: RISC-V ISA functional simulator (instruction semantics)
3. **Pilos**: Performance model (microarchitecture simulation)
4. **Permafrost**: Orchestration framework coordinating all components
5. **Arctic**: Carbon simulation framework (infrastructure)

### Simulation Flow

**Entry Point**: `src/iminnt/core.py:820-900`

```python
def simulate(self, bin_name: str = None, default_cmd: Optional[str] = None,
             bin_args: Optional[List[str]] = None, output_dir: Optional[str] = None,
             keep_retires: bool = False, custom_cfg: Optional[str] = None,
             no_roi: bool = False, fn_perf_graph: bool = False):

    # Step 1: Resolve command and create output directory
    cmd_info = self._get_run_cmd(bin_name, default_cmd, bin_args)
    out_dir = RESULTS_DIR / self.target if not output_dir else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Initialize simulation configuration
    cfg_path = self._init_sim_cfg(bin_path, bin_args, keep_retires, out_dir,
                                   custom_cfg=custom_cfg, use_roi=not no_roi)

    # Step 3: Run Permafrost simulation
    sim_cmd = [str(PERMAFROST_BIN), str(cfg_path)]
    shell(sim_cmd, cwd=str(out_dir), env=run_env)

    # Step 4: Extract performance results
    time_ns = get_cycles(out_dir)

    # Step 5: (Optional) Generate flamegraph profile
    if fn_perf_graph:
        generate_profile(out_dir, logger=logger)
```

### Simulation Configuration

**Template**: `src/iminnt/core.py:27-66`

The `PERMAFROST_SIM_CFG` dictionary defines simulation parameters:

```python
PERMAFROST_SIM_CFG = {
    # Trace generation plugins
    "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-riscv64.so",
    "memory": f"func::memory={QEMU_USER_DIR}/bin/qemu-riscv64.so",

    # ISA simulator
    "execution": f"func::execution={SPIKE_DIR}/build/libriscv.so",

    # QEMU arguments
    "argv": f"func::argv=qemu.so {QEMU_ROI_ENV_FLAG} "
            f"-one-insn-per-tb -d nochain -umode -internal-syscall "
            f"-cosim -plugin $carbon<COSIM_LIB> -cpu {IMI_CPU_ALIAS} "
            f"$carbon<SIM_BIN> $SIM_ARGS$",

    # Spike configuration
    "imi_spike": f"func::imi_spike::no_args::m_pk="
                 f"--isa={SPIKE_IMI_RISCV_ARCH} $carbon<PK_BIN> "
                 f"$carbon<SIM_BIN> $SIM_ARGS$",

    # Performance model (Pilos)
    "perf": f"perf={DEV_ENV_ROOT}/Pilos/build/libpilos.so",

    # Microarchitecture parameters
    "threads": "perf::pilos::threads=1",
    "knob_VLEN": "perf::pilos::sch::knob_VLEN=128",           # Vector length
    "fe_clk_ps": "perf::pilos::fe_clk_ps=1000",              # 1GHz clock (1ns period)
    "mc_delay": "perf::pilos::mc_delay=200",                  # Memory controller delay

    # Pipeline configuration
    "allocate_vec_limit": "perf::pilos::sch::allocate_vec_limit=8",
    "allocate_int_limit": "perf::pilos::sch::allocate_int_limit=8",
    "knob_vec_prf_entries": "perf::pilos::sch::knob_vec_prf_entries=224",
    "ports": "perf::pilos::sch::allocate_pipe::ports=12",    # 12 execution ports

    # Memory subsystem (multiple load/store ports)
    "mem_0": "perf::pilos::sch::dist_rs::mem::0 = LD_FUN | ST_FUN",
    "mem_1": "perf::pilos::sch::dist_rs::mem::1 = LD_FUN",
    # ... (more memory ports)
}
```

**Key microarchitecture parameters**:
- **VLEN=128**: Vector register length (128 bits)
- **12 execution ports**: Wide out-of-order superscalar
- **8 vector/int dispatch limit**: Per-cycle issue limits
- **224 vector PRF entries**: Physical register file size
- **1GHz clock**: Performance timing model

### Configuration Initialization

**Implementation**: `src/iminnt/core.py:673-740`

```python
def _init_sim_cfg(self, bin_path: str, bin_args: str, keep_retires: bool,
                  out_dir: Path, custom_cfg: Optional[Path] = None,
                  use_roi: bool = False) -> str:

    if custom_cfg:
        # Load and validate custom configuration
        sim_cfg = self._check_sim_cfg(custom_cfg)
    else:
        # Use default Pilos configs + PERMAFROST_SIM_CFG
        log_cfg = str(DEV_ENV_ROOT / "Pilos" / "Configs" / "disable_logs.cfg")
        aetos_cfg = str(DEV_ENV_ROOT / "Pilos" / "Configs" / "aetos_hp.cfg")

        init_cfg = [
            f"-cfg {log_cfg}",
            f"-cfg {aetos_cfg}",
            "perf_setup_delay=0"
        ] + list(PERMAFROST_SIM_CFG.values())

        # Configure ROI (Region of Interest)
        if use_roi:
            init_cfg.append("wait_for_roi=1")
            init_cfg.append("listen_to_roi=true")
        else:
            init_cfg.append("wait_for_roi=0")
            init_cfg.append("listen_to_roi=false")

    # Add binary and library paths
    init_cfg.insert(0, f"-carbon_set=SIM_BIN,{bin_path}")
    init_cfg.insert(0, f"-carbon_set=COSIM_LIB,{QEMU_USER_DIR}/plugins/libcosim.so")
    init_cfg.insert(0, f"-carbon_set=PK_BIN,{SPIKE_DIR}/pk_binary/pk")

    # Substitute template variables
    for e in init_cfg:
        e_cfg = e.replace("$SIM_ARGS$", bin_args)
        e_cfg = e_cfg.replace("$RETIRE_KEEP$", str(keep_retires).lower())
        sim_cfg.append(e_cfg)

    # Write configuration file
    cfg_path = Path(f"{out_dir}/iminnt.cfg")
    with open(str(cfg_path), "w") as f:
        f.writelines("\n".join(sim_cfg))

    return cfg_path
```

### Performance Extraction

**Implementation**: `src/iminnt/utils.py:55-71`

```python
def get_cycles(sim_dir: Path, return_perf_info = False):
    stat_file = (sim_dir / "stats" / "pilos_combined.stats")

    with open(str(stat_file), "r") as f:
        stat_lines = [l.strip() for l in f.readlines()]

    for l in stat_lines:
        s = re.search("total_cycles=(\d+)", l)
        if s:
            cycles = int(s.group(1))
            time_ns = get_perf_in_ns(float(cycles))

            if return_perf_info:
                return {0: {"cycles": cycles, "time_ns": time_ns, "freq": CPU_FREQ}}
            else:
                return {0: time_ns}
```

**Cycle to Time Conversion**: `src/iminnt/utils.py:73-74`
```python
def get_perf_in_ns(cycles: float, freq: float = CPU_FREQ):
    return (cycles / freq) * 1e9  # Convert to nanoseconds
```

**CPU Frequency**: `src/iminnt/constants.py:48`
```python
CPU_FREQ = 3.0e9  # 3 GHz
```

### Simulation Outputs

After simulation completes, the output directory contains:

```
results/{target}/
├── iminnt.cfg                    # Simulation configuration used
├── out_config.txt                # Actual config used by Permafrost
├── simbench.log                  # Simulation execution log
├── stats/
│   └── pilos_combined.stats      # Performance statistics (cycles, IPC, etc.)
├── retires.log                   # Retired instruction trace (if -k flag)
└── trace.json                    # Flamegraph profile (if -g flag)
```

**Key metrics in pilos_combined.stats**:
- `total_cycles`: Total simulated cycles
- `total_instructions`: Total instructions retired
- IPC (instructions per cycle)
- Cache hit/miss rates
- Branch prediction accuracy
- Vector unit utilization

### Trace Generation and Profiling

**Implementation**: `src/iminnt/trace_gen.py:713-723`

When `fn_perf_graph=True` (via `-g` flag):

```python
def generate_profile(res_dir: Path, fn_target: Optional[str] = None,
                     debug: bool = False, logger = None):

    # Load simulation config to find binary path
    cfg = get_sim_cfg(cfg_file)

    # Extract symbols from ELF binary
    syms = SymResolver.from_file(Path(cfg['bin_path']))

    # Parse retired instructions and build call tree
    tgen = TraceGen(res_dir, cfg, syms, fn_target=fn_target,
                    skip_fns=SKIP_FUNCS, logger=logger)
    tgen.run_trace()

    # Write Chrome Trace Format JSON
    tgen.write_trace_file()
```

The generated `trace.json` file can be visualized in:
- Chrome browser: `chrome://tracing`
- Perfetto UI: https://ui.perfetto.dev/

This provides a flamegraph showing:
- Function call hierarchy
- Time spent in each function
- Call counts and cycle attribution
- Critical path analysis

---

## 6. Key Implementation Details

### Class Hierarchy

```
BaseRunner (abstract base class)
  └── BenchSimRunner (adds simulation capabilities)
       └── LlamaCppBase (common llama.cpp logic)
            ├── LlamaCppX86           (x86 native, with REF backend)
            ├── LlamaCppX86Bench      (x86 optimized, no REF backend)
            ├── LlamaCppIMI           (RISC-V with IMI extensions, with REF)
            ├── LlamaCppIMIBench      (RISC-V IMI optimized, no REF)
            ├── LlamaCppRVV           (RISC-V standard RVV, with REF)
            ├── LlamaCppRVVBench      (RISC-V RVV optimized, no REF)
            ├── LlamaCppAMXBench      (x86 with AMX, remote execution)
            ├── LlamaCppOryonBench    (ARM Oryon, remote execution)
            └── LlamaCppOryon         (ARM Oryon with REF backend)
```

### Build Variants

The system supports multiple build variants with different configurations:

| Target | Architecture | Extensions | REF Backend | Execution |
|--------|--------------|------------|-------------|-----------|
| `llama_imi` | RISC-V | IMI custom | Yes | QEMU |
| `llama_imi_bench` | RISC-V | IMI custom | No | QEMU |
| `llama_rvv` | RISC-V | Standard RVV | Yes | QEMU |
| `llama_rvv_bench` | RISC-V | Standard RVV | No | QEMU |
| `llama_x86` | x86-64 | Native | Yes | Native |
| `llama_x86_bench` | x86-64 | Native | No | Native |
| `llama_amx_bench` | x86-64 | AMX | No | Remote |
| `llama_oryon` | ARM | Oryon | Yes | Remote |
| `llama_oryon_bench` | ARM | Oryon | No | Remote |

**REF Backend**: Reference implementation for validation. Bench variants exclude it for performance.

Each variant has:
- Unique build directory: `linux-build-{variant}`
- Unique install directory: `llamacpp-{variant}-install`
- Different environment variables
- Different CMake flags
- Different features enabled

### Registry Pattern

**File**: `src/iminnt/registry.py`

```python
_class_registry = {}

def register_class(name):
    """Decorator to register classes by name"""
    def decorator(cls):
        _class_registry[name] = cls
        return cls
    return decorator

def get_class_by_name(name: str):
    """Retrieve class by registered name"""
    return _class_registry.get(name)
```

**Usage**: Each target class is decorated:
```python
@register_class("llama_imi")
class LlamaCppIMI(LlamaCppBase):
    ...
```

**Lookup**: `src/iminnt/cli.py:97`
```python
dev_fw = get_class_by_name(args.target)()
```

This pattern enables:
- Dynamic target discovery
- No CLI code modification when adding targets
- Clean separation of concerns

### Environment Composition

Environments are merged hierarchically:

1. **Base Environment**: `src/iminnt/constants.py:50`
   ```python
   BASE_ENV = os.environ.copy()  # Current shell environment
   ```

2. **Target-specific Environment**: Provided in class constructor
   ```python
   env=IMI_ENV  # For llama_imi
   ```

3. **Merged Environment**: `src/iminnt/core.py:418`
   ```python
   env = dict(BASE_ENV, **self.env)  # Target env overrides base
   ```

4. **Command-specific Environment**: `src/iminnt/core.py:457`
   ```python
   run_env = dict(BASE_ENV, **di.get("env", {}))  # Command-specific overrides
   ```

### Remote Execution

For native execution on specialized hardware (x86 AMX, ARM Oryon):

**Implementation**: `src/iminnt/core.py:596-610`

```python
elif self.remote_info:
    bench_dir = Path(self.remote_info["bench_dir"])
    remote_bin_path = f"{bench_dir}/{Path(bin_name).stem}"

    # Create SSH connection
    rconn = RemoteConn(self.remote_info["name"],
                       self.remote_info["uname"],
                       self.remote_info["host"],
                       self.remote_info["port"])

    # Copy binary to remote server
    rconn.copy_to_remote(bin_path, remote_bin_path, 0o755)

    # Execute remotely
    remote_bin_cmd = f"{remote_bin_path} {bin_args}"
    rconn.run_cmd(remote_bin_cmd)
```

This allows running:
- x86 AMX benchmarks on Azure servers with AMX support
- ARM Oryon benchmarks on Qualcomm hardware

### Rebuild vs Build

**Rebuild**: `src/iminnt/llamacpp.py:108-109`
```python
@property
def rebuild_cmd(self) -> list:
    return [f"cmake --build {self.build_dir} -j -t install"]
```

**Usage**:
```bash
iminnt -t llama_imi rebuild
```

Rebuild skips the clean and configure steps, only rebuilding changed sources and reinstalling. This is much faster for iterative development.

### Upstream Sync Workflow

**Implementation**: `src/iminnt/core.py:270-317`

```python
def sync(self):
    # Save current branch
    cur_branch = get_current_branch(self.root)

    # 1. Checkout 'upstream' branch
    shell(["git", "-C", f"{self.root}", "checkout", "upstream"])

    # 2. Fetch from upstream remote
    shell(f"git -C {self.root} fetch upstream".split())

    # 3. Get default branch name (usually 'master' or 'main')
    default_branch_out = shell(f"git -C {self.root} remote show upstream".split())
    def_branch = extract_default_branch(default_branch_out)

    # 4. Rebase onto upstream
    shell(f"git -C {self.root} rebase upstream/{def_branch}".split())

    # 5. Force push to origin
    shell(f"git -C {self.root} push -f origin upstream".split())

    # 6. Return to original branch
    shell(["git", "-C", f"{self.root}", "checkout", cur_branch])
```

This allows I-Machines fork to stay synchronized with official llama.cpp releases.

### Static vs Dynamic Linking

**Static Linking Configuration**:
- `CROSS_STATIC=ON` in CMake args
- `GGML_STATIC=ON` for GGML library
- `-static` linker flag in toolchain.cmake

**Benefits**:
- Single binary includes all dependencies
- No runtime library search needed
- Simpler QEMU execution (no `-L` sysroot flag required)

**Trade-offs**:
- Larger binary size
- Cannot share common libraries between binaries

---

## Summary

The iminn-tools system provides a comprehensive, production-grade framework for managing the llama.cpp lifecycle across multiple architectures:

### Key Capabilities

1. **Initialization**: Clones I-Machines fork, sets up upstream tracking, downloads pre-quantized models
2. **Cross-compilation**: Builds for RISC-V (IMI/RVV), x86, ARM with architecture-specific optimizations
3. **Installation**: Installs 67 binaries with static linking for portability
4. **Execution**: Runs via QEMU user-mode emulation or remote/native execution
5. **Simulation**: Provides cycle-accurate performance analysis with Permafrost/Pilos/Spike
6. **Profiling**: Generates flamegraph visualizations of function-level performance

### Design Patterns

- **Registry pattern**: Dynamic target discovery
- **Class hierarchy**: Shared logic with variant-specific overrides
- **Environment composition**: Layered configuration management
- **Toolchain abstraction**: CMake-based cross-compilation
- **Command templating**: Pre-configured test scenarios

### Extensibility

The architecture makes it easy to:
- Add new target variants (new ISA extensions, backends)
- Support new architectures (just add environment dict and toolchain config)
- Define custom test scenarios (add to `default_runs` dict)
- Integrate new simulation tools (modify `PERMAFROST_SIM_CFG`)

This design provides a scalable, maintainable foundation for AI/ML performance engineering across diverse hardware platforms.
