# psim Simulator Architecture Support

**Date:** January 5, 2025  
**Purpose:** Document which architectures and ISAs are supported by the psim simulation infrastructure

---

## Executive Summary

**psim (Permafrost/Pilos simulator) is RISC-V specific** and supports:
- ✅ **RISC-V 64-bit (RV64)** architectures
- ✅ **RISC-V Vector Extension (RVV)** - Standard vector extensions
- ✅ **IMI (I-Machines)** - Custom RISC-V extensions
- ❌ **x86-64** - Not supported
- ❌ **ARM/AArch64** - Not supported
- ❌ **Other architectures** - Not supported

---

## Supported Architectures

### 1. RISC-V 64-bit (RV64)

**Base Architecture**: RISC-V 64-bit with various extensions

#### Variant A: IMI (I-Machines Custom Extensions)

**Target**: `llama_imi`, `llama_imi_bench`

**Architecture String** (`IMI_RISCV_ARCH`):
```
RV64GCV_ximimce_zba_zbb_zbs_zicntr_zihpm_zihintpause_zicbom_zicbop_zicboz_zihintntl_zicond_zcb_zfa_zawrs_zfh_zvfh_zfhmin_zvbb_zimop_zcmop_zfbfmin_zvfbfwma
```

**Extensions**:
- **Base**: `RV64GCV` (64-bit, G (IMAFD), C (compressed), V (vector))
- **Custom**: `ximimce` (I-Machines custom extensions)
- **Standard Extensions**:
  - `zba`, `zbb`, `zbs` - Bit manipulation
  - `zicntr`, `zihpm` - Counters and performance monitors
  - `zihintpause` - Hint pause
  - `zicbom`, `zicbop`, `zicboz` - Cache-block operations
  - `zihintntl` - Non-temporal locality hints
  - `zicond` - Conditional operations
  - `zcb` - Code size reduction
  - `zfa`, `zawrs` - Floating-point and wait/reserve
  - `zfh`, `zvfh`, `zfhmin` - Half-precision floating-point
  - `zvbb` - Basic bit manipulation
  - `zimop`, `zcmop` - Matrix operations
  - `zfbfmin`, `zvfbfwma` - BF16 floating-point

**Vector Configuration**:
- VLEN: 128 bits
- Supports custom IMI vector instructions

**Simulation Support**: ✅ Fully supported

**Example Usage**:
```bash
# Build
iminnt -t llama_imi build

# Run (QEMU emulation)
iminnt -t llama_imi run -d test_q4_0_stories

# Simulate (full performance simulation)
iminnt -t llama_imi sim -d test_q4_0_stories
```

#### Variant B: RVV (RISC-V Vector Extension)

**Target**: `llama_rvv`, `llama_rvv_bench`

**Architecture String** (`RVV_RISCV_ARCH`):
```
RV64GCV_zba_zbb_zbs_zicntr_zihpm_zihintpause_zicbom_zicbop_zicboz_zihintntl_zicond_zcb_zfa_zawrs_zfh_zvfh_zvbb_zimop_zcmop_zfbfmin_zvfbfwma
```

**Extensions**:
- **Base**: `RV64GCV` (64-bit, G (IMAFD), C (compressed), V (vector))
- **No custom extensions**: Uses standard RISC-V extensions only
- **Standard Extensions**: Same as IMI but without `ximimce`

**Vector Configuration**:
- VLEN: 128 bits
- Standard RISC-V Vector Extension (RVV) instructions

**Simulation Support**: ✅ Fully supported

**Example Usage**:
```bash
# Build
iminnt -t llama_rvv build

# Run (QEMU emulation)
iminnt -t llama_rvv run -d test_q4_0_stories

# Simulate (full performance simulation)
iminnt -t llama_rvv sim -d test_q4_0_stories
```

---

## Unsupported Architectures

### ❌ x86-64

**Targets**: `llama_x86`, `llama_x86_bench`

**Status**: Not supported for simulation

**Why**:
- psim uses `qemu-riscv64.so` (RISC-V specific)
- Spike ISA simulator only supports RISC-V
- Pilos performance model is RISC-V microarchitecture specific

**What Works**:
- ✅ Native execution (fast, no simulation)
- ❌ Performance simulation via psim

**Example**:
```bash
# Works - native execution
iminnt -t llama_x86 run -d test_glm_4_6v_text

# Does NOT work - no simulation, just direct execution
iminnt -t llama_x86 sim -d test_glm_4_6v_text
# (Falls back to direct binary execution, no performance profiling)
```

**Code Reference**: `src/iminnt/core.py:1140-1189`
- When `use_qemu=False`, simulation pipeline is skipped
- Falls back to direct binary execution

### ❌ ARM/AArch64

**Targets**: `llama_oryon`, `llama_oryon_bench`

**Status**: Not supported for simulation

**Why**: Same reasons as x86 - infrastructure is RISC-V specific

**What Works**:
- ✅ Native execution on ARM hosts
- ✅ Remote execution on ARM servers
- ❌ Performance simulation via psim

**Note**: These targets use `remote_info` for execution on remote ARM servers, not local simulation.

---

## Technical Details

### Simulation Infrastructure Components

**Location**: `src/iminnt/core.py:30-69` (`PERMAFROST_SIM_CFG`)

```python
PERMAFROST_SIM_CFG = {
    # QEMU RISC-V (64-bit only)
    "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-riscv64.so",
    "memory": f"func::memory={QEMU_USER_DIR}/bin/qemu-riscv64.so",
    
    # Spike RISC-V ISA simulator
    "execution": f"func::execution={SPIKE_DIR}/build/libriscv.so",
    "imi_spike": f"func::imi_spike::no_args::m_pk=--isa={SPIKE_IMI_RISCV_ARCH} ...",
    
    # Pilos RISC-V performance model
    "perf": f"perf={DEV_ENV_ROOT}/Pilos/build/libpilos.so",
    
    # RISC-V specific microarchitecture parameters
    "knob_VLEN": "perf::pilos::sch::knob_VLEN=128",  # Vector length
    # ... more RISC-V specific parameters
}
```

### Simulation Flow

**Location**: `src/iminnt/core.py:1119-1198`

```python
def simulate(self, ...):
    if self.use_qemu:  # ← Only RISC-V targets have use_qemu=True
        # Full Permafrost simulation pipeline
        cfg_path = self._init_sim_cfg(...)
        sim_cmd = [str(PERMAFROST_BIN), str(cfg_path)]
        shell(sim_cmd, ...)
    else:  # ← x86/ARM targets (use_qemu=False)
        # No simulation - just direct execution
        sim_cmd = bin_cmd.split()
        shell(sim_cmd, ...)
```

### Architecture Detection

**RISC-V Detection**: `src/iminnt/llamacpp.py:179-198`

```python
if self.is_riscv or self.is_aarch64:
    base_args["CROSS_STATIC"] = "ON"
    base_args["CMAKE_TOOLCHAIN_FILE"] = f"{CROSS_TOOLCHAIN_PATH}"

if self.is_riscv:
    base_args["GGML_NATIVE"] = "OFF"
    base_args["GGML_RVV"] = "ON"
    base_args["GGML_RVV_VLEN"] = "128"
    # ... RISC-V specific flags
```

---

## psim Compound Target

**Location**: `src/iminnt/infra.py:430-434`

```python
@register_class("psim")
class PSim(MultiRunner):
    def __init__(self):
        super().__init__("psim", [
            RISCVEnv(),      # RISC-V toolchain
            Arctic(),        # Carbon simulation framework
            Permafrost(),    # Simulation orchestrator
            Pilos(),         # Performance model (RISC-V)
            Spike(),         # RISC-V ISA simulator
            Qemu(),          # QEMU (configured for RISC-V)
            Simpoint()       # Statistical sampling
        ])
```

**Key Point**: All components are either RISC-V specific or architecture-agnostic but used in RISC-V context.

---

## Summary Table

| Architecture | ISA | psim Simulation | Native Execution | QEMU Emulation | Notes |
|--------------|-----|----------------|------------------|----------------|-------|
| **RISC-V IMI** | RV64 + IMI extensions | ✅ Yes | ✅ Yes | ✅ Yes | Full simulation support |
| **RISC-V RVV** | RV64 + Vector | ✅ Yes | ✅ Yes | ✅ Yes | Full simulation support |
| **x86-64** | x86-64 | ❌ No | ✅ Yes | ❌ No | Native execution only |
| **ARM/AArch64** | AArch64 | ❌ No | ✅ Yes (remote) | ❌ No | Remote execution only |

---

## Requirements for Simulation

To use psim simulation, a target must:

1. **Have `use_qemu=True`**: Indicates RISC-V target
2. **Use RISC-V toolchain**: Cross-compilation for RISC-V
3. **Be compiled with debug symbols**: For trace generation (`-g` flag)
4. **Have QEMU available**: `qemu-riscv64` in `QEMU_USER_DIR`
5. **Have Spike available**: RISC-V ISA simulator in `SPIKE_DIR`
6. **Have Pilos available**: Performance model in `DEV_ENV_ROOT/Pilos`

**Example Check**:
```python
# In llamacpp.py
class LlamaCppIMI(LlamaCppBase):
    def __init__(self):
        super().__init__(
            ...
            use_qemu=True  # ← Required for simulation
        )
```

---

## Future Extensibility

To add support for other architectures (e.g., x86, ARM), would require:

1. **New QEMU targets**: `qemu-x86_64.so`, `qemu-aarch64.so`
2. **New ISA simulators**: Replace or extend Spike for x86/ARM
3. **New performance models**: Adapt Pilos for x86/ARM microarchitecture
4. **Updated configuration**: Extend `PERMAFROST_SIM_CFG` with architecture-specific parameters
5. **Architecture detection**: Modify `simulate()` method to handle multiple architectures

**This is a significant architectural change and is not currently supported.**

---

## Code References

- **psim definition**: `src/iminnt/infra.py:430-434`
- **Simulation config**: `src/iminnt/core.py:30-69` (`PERMAFROST_SIM_CFG`)
- **Simulation method**: `src/iminnt/core.py:1119-1198` (`simulate()`)
- **RISC-V architectures**: `src/iminnt/constants.py:60-61` (`IMI_RISCV_ARCH`, `RVV_RISCV_ARCH`)
- **Target definitions**: `src/iminnt/llamacpp.py:425-564`

---

**Last Updated:** January 5, 2025
