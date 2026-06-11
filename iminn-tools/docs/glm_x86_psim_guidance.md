# GLM Model on x86 with psim Simulator - Guidance Summary

**Date:** January 5, 2025  
**Model:** GLM-4.6V-Flash (9.4B parameters)

---

## Executive Summary

**Key Finding:** The psim simulator (Permafrost/Pilos) is **RISC-V specific** and **cannot directly simulate x86 binaries**. However, GLM-4.6V-Flash works well on native x86 for execution, and the simulation infrastructure can be used with the RISC-V version via QEMU emulation.

---

## What is psim?

**psim** is a compound target that includes the complete simulation infrastructure:

- **Permafrost**: Simulation orchestrator
- **Pilos**: Performance model (microarchitecture simulation)
- **QEMU**: Execution trace generation
- **Spike**: RISC-V ISA functional simulator
- **Arctic**: Carbon simulation framework
- **Simpoint**: Statistical sampling tool

**Architecture Limitation**: The simulation infrastructure is **RISC-V-specific**:
- Uses `qemu-riscv64.so` for trace generation
- Uses Spike RISC-V ISA simulator
- Pilos models RISC-V microarchitecture
- Cannot simulate x86-64 binaries directly

---

## Current Status

### ✅ What Works

1. **GLM on x86 (native execution)**:
   ```bash
   iminnt -t llama_x86 run -d test_glm_4_6v_text
   ```
   - Fast execution (no emulation overhead)
   - Works perfectly for functional testing
   - No simulation/profiling available

2. **GLM on IMI (with QEMU emulation)**:
   ```bash
   iminnt -t llama_imi run -d test_glm_4_6v_text
   ```
   - Slow but functional (runs for hours)
   - Model loads and generates correctly
   - Can potentially use simulation infrastructure

### ❌ What Doesn't Work Directly

**x86 with psim simulation**:
```bash
# This does NOT work as intended
iminnt -t llama_x86 sim -d test_glm_4_6v_text
```

**Why it fails:**
- `llama_x86` has `use_qemu=False` (line 432 in `llamacpp.py`)
- When `use_qemu=False`, the `simulate()` method (line 1140 in `core.py`) skips the Permafrost simulation pipeline
- Falls back to direct binary execution without performance simulation
- The simulation infrastructure requires QEMU for trace generation

---

## Architecture Analysis

### Simulation Infrastructure (`PERMAFROST_SIM_CFG`)

**Location**: `src/iminnt/core.py:30-69`

The simulation configuration is hardcoded for RISC-V:

```python
PERMAFROST_SIM_CFG = {
    "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-riscv64.so",  # RISC-V QEMU
    "execution": f"func::execution={SPIKE_DIR}/build/libriscv.so",          # RISC-V ISA
    "argv": f"... -cpu {IMI_CPU_ALIAS} ...",                               # RISC-V CPU
    "imi_spike": f"... --isa={SPIKE_IMI_RISCV_ARCH} ...",                  # RISC-V ISA
    "perf": f"perf={DEV_ENV_ROOT}/Pilos/build/libpilos.so",                # RISC-V perf model
    # ... RISC-V specific parameters (VLEN=128, vector units, etc.)
}
```

### Simulation Method Flow

**Location**: `src/iminnt/core.py:1119-1198`

```python
def simulate(self, ...):
    # ...
    if self.use_qemu:  # ← Requires QEMU (RISC-V)
        cfg_path = self._init_sim_cfg(...)  # Creates Permafrost config
        sim_cmd = [str(PERMAFROST_BIN), str(cfg_path)]
        shell(sim_cmd, ...)  # Runs Permafrost simulation
    else:  # ← x86 path (no simulation)
        # Just runs binary directly, no performance simulation
        sim_cmd = bin_cmd.split()
        shell(sim_cmd, ...)
```

---

## Options for GLM Simulation

### Option 1: Use IMI Target with Simulation (Recommended for Simulation)

Since GLM works on IMI (albeit slowly), you can potentially use simulation:

```bash
# First, ensure psim infrastructure is built
iminnt -t psim init
iminnt -t psim build

# Then run simulation with IMI target
iminnt -t llama_imi sim -d test_glm_4_6v_text
```

**Pros:**
- Full performance simulation available
- Cycle-accurate profiling
- Detailed microarchitecture analysis

**Cons:**
- Very slow (model is 9.4B parameters)
- Requires QEMU emulation overhead
- May take days to complete simulation

### Option 2: Native x86 Execution (Recommended for Fast Testing)

For functional testing and quick iteration:

```bash
iminnt -t llama_x86 run -d test_glm_4_6v_text
```

**Pros:**
- Fast execution
- No emulation overhead
- Good for development and testing

**Cons:**
- No performance simulation
- No cycle-accurate profiling
- Cannot analyze microarchitecture performance

### Option 3: Hybrid Approach

Use x86 for fast functional testing, then selectively simulate specific parts:

1. **Fast iteration on x86**: Test model behavior, prompt variations
2. **Targeted simulation on IMI**: Simulate specific kernels or operations
3. **Use smaller models**: For full simulation, consider smaller models first

---

## Technical Constraints

### Why x86 Simulation Isn't Supported

1. **QEMU Dependency**: Permafrost requires QEMU for execution trace generation, but uses `qemu-riscv64` (RISC-V specific)

2. **ISA Simulator**: Spike only supports RISC-V ISA, not x86-64

3. **Performance Model**: Pilos models RISC-V microarchitecture (vector units, RISC-V specific features like VLEN=128)

4. **Architecture Mismatch**: The simulation infrastructure is designed to model RISC-V processors, not x86

### What Would Be Needed for x86 Simulation

To simulate x86 binaries, you would need:

1. **x86 QEMU integration**: Modify Permafrost to support x86 QEMU (`qemu-x86_64`)
2. **x86 ISA simulator**: Replace or extend Spike to handle x86-64 instructions
3. **x86 performance model**: Adapt Pilos to model x86 microarchitecture (different pipeline, SIMD units, etc.)
4. **Configuration updates**: Extend `PERMAFROST_SIM_CFG` for x86 parameters

**This is a significant architectural change and is not currently supported.**

---

## Practical Recommendations

### For GLM-4.6V-Flash Specifically

Given that GLM-4.6V-Flash is a 9.4B parameter model:

1. **Use x86 for development**: Fast iteration and functional testing
   ```bash
   iminnt -t llama_x86 run -d test_glm_4_6v_text
   ```

2. **Use IMI/QEMU for simulation** (if needed): Be prepared for very long simulation times
   ```bash
   # Ensure psim is built first
   iminnt -t psim init && iminnt -t psim build
   
   # Then simulate (will take hours/days)
   iminnt -t llama_imi sim -d test_glm_4_6v_text
   ```

3. **Consider smaller models for simulation**: Start with `stories15M` (15M parameters) to validate simulation workflow before attempting GLM

4. **Use ROI markers**: If simulating, use Region-of-Interest markers to focus on specific parts of execution

---

## Code References

- **psim definition**: `src/iminnt/infra.py:430-434`
- **Simulation config**: `src/iminnt/core.py:30-69` (`PERMAFROST_SIM_CFG`)
- **Simulation method**: `src/iminnt/core.py:1119-1198` (`simulate()`)
- **x86 target**: `src/iminnt/llamacpp.py:425-433` (`LlamaCppX86`)
- **IMI target**: `src/iminnt/llamacpp.py:446-464` (`LlamaCppIMI`)

---

## Summary

**Current State:**
- ✅ GLM works on x86 (fast execution, no simulation)
- ✅ GLM works on IMI (slow execution, simulation possible)
- ❌ Direct x86 simulation not supported (psim is RISC-V specific)

**Recommendation:**
- Use **x86 for fast functional testing** of GLM
- Use **IMI for performance simulation** if cycle-accurate analysis is needed (with very long runtime)
- For simulation validation, start with smaller models (stories15M) before attempting GLM

**Key Limitation:**
The psim simulator infrastructure is fundamentally RISC-V oriented and cannot directly simulate x86 binaries. This is an architectural constraint, not a configuration issue.

---

**Last Updated:** January 5, 2025
