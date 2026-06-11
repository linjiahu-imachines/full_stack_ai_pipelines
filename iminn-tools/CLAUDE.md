# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMINN Tools (`iminnt`) is a Python-based performance evaluation and profiling framework for AI/ML workloads. It provides a unified interface for building, running, simulating, and profiling AI/ML frameworks across different architectures (RISC-V, x86, ARM) and compilers (Clang, GCC). The project integrates simulation infrastructure (QEMU, Spike, Pilos, Permafrost, Arctic) with AI/ML frameworks (XNNPACK, llama.cpp, TFLite/LiteRT) for comprehensive performance analysis.

## Commands

### Development Setup

```bash
# Install the package in editable mode
pip3 install -e .

# Install development dependencies
pip3 install -e ".[dev]"

# Verify installation
iminnt --help
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit manually
pre-commit run --all-files
```

### Linting and Formatting

```bash
# Format code with black (line length 120)
black src/iminnt

# Lint with ruff
ruff check src/iminnt --fix
```

### Common iminnt Commands

The `iminnt` CLI is the primary interface for all operations. Format: `iminnt -t <target> <action> [options]`

**Initialization (recommended first-time setup):**
```bash
# Initialize simulation infrastructure (compound target)
iminnt -t psim init

# Build simulation infrastructure (can take very long)
iminnt -t psim build

# Initialize an AI framework
iminnt -t xnnpack_imi init
iminnt -t xnnpack_imi build
```

**Individual Actions:**
```bash
# Initialize/clone a target's source code
iminnt -t <target> init

# Build from scratch (clean build)
iminnt -t <target> build

# Rebuild (incremental build)
iminnt -t <target> rebuild

# Pull latest source code
iminnt -t <target> pull

# Checkout a branch
iminnt -t <target> checkout -b <branch_name>

# Sync with upstream (for forked targets)
iminnt -t <target> sync
```

**Execution and Profiling:**
```bash
# Run via QEMU emulation
iminnt -t xnnpack_imi run -d <default_cmd_name>

# Simulate and profile performance
iminnt -t xnnpack_imi sim -d <default_cmd_name> -o results/my_test

# Simpoint analysis
iminnt -t xnnpack_imi simpoint -d <default_cmd_name>

# List available default commands for a target
iminnt -t xnnpack_imi defaults

# Run benchmark sweep (XNNPACK only currently)
iminnt -t xnnpack_imi sweep -o results/sweep_test
```

## Architecture

### Core Design Pattern

The codebase follows an object-oriented architecture with a class hierarchy centered around runners:

1. **BaseRunner** (`src/iminnt/core.py`): Abstract base class defining the interface for all targets
2. **Specialized Base Classes**:
   - `BenchSimRunner`: For targets that can be built, run, and simulated (AI frameworks)
   - `CompoundRunner`: Meta-targets that orchestrate multiple sub-targets (e.g., `psim`, `riscv-env`)
   - `MultiRunner`: For targets with multiple build variants
   - `SubRunner`: Infrastructure components (QEMU, Spike, etc.)

3. **Framework Implementations**: Each AI/ML framework extends appropriate base class
   - `src/iminnt/xnnpack.py`: XNNPACK variants (IMI, x86, AMX, Neoverse)
   - `src/iminnt/llamacpp.py`: llama.cpp variants
   - `src/iminnt/litert.py`: TensorFlow Lite/LiteRT
   - `src/iminnt/timvx.py`: TIM-VX inference library
   - `src/iminnt/onnx_rt.py`: ONNX Runtime
   - `src/iminnt/iree.py`: IREE

4. **Infrastructure Implementations** (`src/iminnt/infra.py`): Simulation/emulation tools
   - QEMU, Spike, Arctic, Pilos, Permafrost

### Target System

Targets represent buildable/runnable components. Key target categories:

- **AI Framework Targets**: Named like `<framework>_<backend>` (e.g., `xnnpack_imi`, `llama_x86`)
- **Compound Targets**: Aliases for multiple targets (e.g., `psim` = qemu + arctic + pilos + permafrost + spike + riscv-env + simpoint)
- **Infrastructure Targets**: Simulation/emulation tools (qemu, spike, pilos, permafrost, arctic)
- **Environment Targets**: Toolchain setups (riscv-env, npu-env)

### Cross-Compilation Toolchain

The project uses CMake toolchain files for cross-compilation:

- **`riscv-imi.cmake`**: RISC-V cross-compilation with I-Machines custom extensions
- **`toolchain.cmake`**: Generic cross-compilation (RISC-V, ARM, x86)

Required environment variables for cross-compilation:
- `CROSS_SYSROOT`: Sysroot path
- `IMI_LLVM_PATH`: Path to LLVM installation
- `CROSS_ARCH`: Target architecture (riscv64, aarch64, x86_64)
- `CROSS_TRIPLE`: Target triple (e.g., riscv64-linux-gnu)
- `CROSS_COMP`: Compiler choice (clang or gcc)
- `CROSS_CPU`: CPU architecture variant

### Simulation Pipeline

The simulation workflow integrates multiple components:

1. **QEMU User Mode** (`qemu-riscv64`): Provides execution trace and memory operations
2. **Spike** (`libriscv.so`): RISC-V ISA functional simulator
3. **Pilos** (`libpilos.so`): Performance model for microarchitecture simulation
4. **Permafrost**: Orchestrates the simulation components using configuration files

Configuration is managed via `PERMAFROST_SIM_CFG` dict in `core.py:27-66`, which defines:
- Trace generation plugins
- Memory model configuration
- Performance model parameters (VLEN, pipeline config, cache behavior)
- ROI (Region of Interest) markers for focused profiling

### Registry Pattern

The `src/iminnt/registry.py` implements a class registry for dynamic target discovery:
- `@register_class(name)`: Decorator registers targets
- `get_class_by_name(name)`: Retrieves target class by string name
- Enables extensibility without modifying CLI code

### Development Environment Structure

All built artifacts and source code live under `dev_env/`:
```
dev_env/
├── XNNPACK/          # AI framework sources
├── llama.cpp/
├── Pilos/            # Performance simulator
├── csqemu-v9/        # Custom QEMU
├── IMachines_Spike/  # RISC-V ISA simulator
├── Arctic/           # Carbon simulation framework
├── Permafrost/       # Simulation orchestrator
├── riscv-env/        # RISC-V toolchain
├── simpoint/         # Simpoint analysis tool
└── playground/       # Test code snippets
```

Results are stored in `results/<target_name>/` or custom output directories.

## Important Implementation Details

### Command Execution Flow

1. CLI parsing in `src/iminnt/cli.py:main()` creates argument parser
2. Target class instantiated via registry: `get_class_by_name(args.target)()`
3. Command dispatched: `dev_fw.exec_cmd(args.subcommand, **kwargs)`
4. Base class `exec_cmd()` routes to appropriate method (`_build()`, `_run()`, `_sim()`, etc.)

### Build System Integration

- Most targets use CMake with custom toolchain files
- Build commands defined in `get_build_cmd()` method (can return list of commands for multi-stage builds)
- Rebuild commands in `rebuild_cmd` property for incremental builds
- Custom build scripts (shell scripts) specified in `custom_scripts` property

### Simulation Configuration

Simulation configs are Jinja2 templates in `src/iminnt/resources/simcfg/`. The `PERMAFROST_SIM_CFG` dictionary in `core.py` provides defaults that can be overridden:
- Custom configs via `-c/--custom-cfg` flag
- Template variables: `$carbon<COSIM_LIB>`, `$carbon<SIM_BIN>`, `$SIM_ARGS$`
- ROI markers controlled via `IMI_ROI_SIM` environment variable

### Remote Execution Support

The `src/iminnt/ssh.py` module provides `RemoteConn` class for running commands on remote servers via SSH/paramiko. This enables running native builds on appropriate architecture machines.

## Code Style and Conventions

- Python 3.10+ required (3.11.10 recommended)
- Line length: 120 characters
- Formatting: black + ruff (configured in `pyproject.toml`)
- Pre-commit hooks enforce formatting and linting
- Type hints used throughout (`from typing import Dict, Any, Optional, List`)

## Key Files to Understand

- `src/iminnt/cli.py`: Entry point, argument parsing, command routing
- `src/iminnt/core.py`: Base classes, simulation configuration, core abstractions
- `src/iminnt/registry.py`: Target registration system
- `src/iminnt/constants.py`: Path definitions, environment variables, build flags
- `src/iminnt/utils.py`: Shell execution helpers, performance metric extraction
- `src/iminnt/trace_gen.py`: Profile generation from simulation traces
- `toolchain.cmake` & `riscv-imi.cmake`: Cross-compilation configuration
- `.pre-commit-config.yaml`: Code quality automation

## Working with Targets

When adding a new target:

1. Create a class in appropriate module (or new module if needed)
2. Extend `BenchSimRunner` for runnable/simulatable targets, `SubRunner` for infrastructure
3. Decorate with `@register_class("target_name")`
4. Implement required abstract properties: `target`, `root`, `remote_path` (if git clone)
5. Implement build methods: `get_build_cmd()`, optionally `rebuild_cmd`
6. For runnable targets: implement `get_default_cmds()`, `_run()`, `_sim()` if custom behavior needed
7. Import the new class in `src/iminnt/cli.py` to register it

## Debugging and Troubleshooting

- Build logs stored in repository root (e.g., `build_12_2.log`)
- Simulation results include detailed traces, retire files, and performance stats
- Use `-p/--print-cfg` to view simulation configuration before running
- Use `-k/--keep-retires` to preserve retired instruction traces for debugging
- Check `dev_env/<target>/` for build artifacts and source code
- Logger configured in `src/iminnt/log_cfg.py`
