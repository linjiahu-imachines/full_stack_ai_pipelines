# Requirements for Adding x86 and ARM Support to psim

**Date:** January 5, 2025  
**Purpose:** Document what would be required to extend psim simulation to support x86-64 and ARM/AArch64 architectures

---

## Current Limitation

**psim currently only supports RISC-V** due to hardcoded architecture-specific components in the simulation pipeline.

---

## What Would Be Needed

### 1. QEMU Target Support

#### Current (RISC-V only):
```python
PERMAFROST_SIM_CFG = {
    "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-riscv64.so",
    "memory": f"func::memory={QEMU_USER_DIR}/bin/qemu-riscv64.so",
}
```

#### Required Changes:

**For x86-64:**
- Add `qemu-x86_64.so` support (or equivalent QEMU x86 user-mode emulation)
- Modify configuration to select QEMU binary based on target architecture

**For ARM/AArch64:**
- Add `qemu-aarch64.so` support (QEMU ARM user-mode emulation)
- Modify configuration to select QEMU binary based on target architecture

**Implementation Location**: `src/iminnt/core.py:30-69` (`PERMAFROST_SIM_CFG`)

**Required Code Changes**:
```python
# Would need architecture-aware configuration
def get_sim_cfg_for_arch(arch: str):
    if arch == "riscv64":
        return {
            "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-riscv64.so",
            "memory": f"func::memory={QEMU_USER_DIR}/bin/qemu-riscv64.so",
        }
    elif arch == "x86_64":
        return {
            "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-x86_64.so",  # ← Need this
            "memory": f"func::memory={QEMU_USER_DIR}/bin/qemu-x86_64.so",
        }
    elif arch == "aarch64":
        return {
            "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-aarch64.so",  # ← Need this
            "memory": f"func::memory={QEMU_USER_DIR}/bin/qemu-aarch64.so",
        }
```

**Prerequisites:**
- QEMU must be built with x86-64 and ARM support
- QEMU plugins for trace generation must support these architectures
- Check: `dev_env/csqemu-v9/` - verify if x86/ARM QEMU builds exist

---

### 2. ISA Simulator (Spike Replacement)

#### Current (RISC-V only):
```python
PERMAFROST_SIM_CFG = {
    "execution": f"func::execution={SPIKE_DIR}/build/libriscv.so",
    "imi_spike": f"func::imi_spike::no_args::m_pk=--isa={SPIKE_IMI_RISCV_ARCH} ...",
}
```

**Problem**: Spike only supports RISC-V ISA

#### Required Changes:

**For x86-64:**
- Replace or extend Spike with x86-64 ISA simulator
- Options:
  - **Gem5** (x86 support available)
  - **PTLSim** (x86 simulator)
  - **SimpleScalar** (older, x86 support)
  - **Custom x86 simulator** integrated with Permafrost/Arctic

**For ARM/AArch64:**
- Replace or extend Spike with ARM ISA simulator
- Options:
  - **Gem5** (ARM support available)
  - **ARM Fast Models** (ARM proprietary)
  - **Custom ARM simulator** integrated with Permafrost/Arctic

**Alternative Approach**: Use QEMU for functional simulation instead of Spike
- QEMU supports x86, ARM, and RISC-V
- Would require refactoring to use QEMU's functional simulation mode
- May need different trace generation approach

**Implementation Location**: `src/iminnt/core.py:34-36` and `_init_sim_cfg()` method

---

### 3. Performance Model (Pilos Adaptation)

#### Current (RISC-V specific):
```python
PERMAFROST_SIM_CFG = {
    "perf": f"perf={DEV_ENV_ROOT}/Pilos/build/libpilos.so",
    "knob_VLEN": "perf::pilos::sch::knob_VLEN=128",  # RISC-V vector length
    # ... RISC-V specific microarchitecture parameters
}
```

#### Required Changes:

**For x86-64:**
- Adapt or extend Pilos to model x86 microarchitecture
- Key differences:
  - **Pipeline**: x86 has different instruction decode, execution units
  - **SIMD**: AVX/AVX2/AVX-512 instead of RISC-V vector extensions
  - **Cache hierarchy**: Different organization
  - **Out-of-order execution**: Different scheduling policies
  - **Register renaming**: Different PRF (Physical Register File) organization

**For ARM/AArch64:**
- Adapt Pilos to model ARM microarchitecture
- Key differences:
  - **Pipeline**: ARM-specific execution units
  - **NEON SIMD**: ARM vector instructions instead of RISC-V vectors
  - **Cache coherency**: ARM-specific protocols (MOESI, etc.)
  - **Memory ordering**: ARM memory model specifics

**Options:**
1. **Extend Pilos**: Add x86/ARM microarchitecture models to existing Pilos
2. **Use alternative performance models**: 
   - **Gem5** has built-in x86/ARM performance models
   - **McPAT** for power/performance analysis
   - Architecture-specific simulators

**Implementation Location**: `src/iminnt/core.py:37-69` (performance model parameters)

---

### 4. Configuration System Refactoring

#### Current (Hardcoded RISC-V):
```python
PERMAFROST_SIM_CFG = {
    # All hardcoded to RISC-V
}
```

#### Required Changes:

**Make configuration architecture-aware**:

```python
# Architecture-specific simulation configurations
PERMAFROST_SIM_CFG_BASE = {
    # Common parameters
    "threads": "perf::pilos::threads=1",
    "fe_clk_ps": "perf::pilos::fe_clk_ps=1000",
    # ...
}

PERMAFROST_SIM_CFG_RISCV = {
    **PERMAFROST_SIM_CFG_BASE,
    "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-riscv64.so",
    "execution": f"func::execution={SPIKE_DIR}/build/libriscv.so",
    "knob_VLEN": "perf::pilos::sch::knob_VLEN=128",  # RISC-V specific
    # ... RISC-V specific parameters
}

PERMAFROST_SIM_CFG_X86 = {
    **PERMAFROST_SIM_CFG_BASE,
    "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-x86_64.so",
    "execution": f"func::execution={X86_SIMULATOR_DIR}/libx86.so",  # Need x86 simulator
    # x86 specific parameters (AVX widths, etc.)
}

PERMAFROST_SIM_CFG_ARM = {
    **PERMAFROST_SIM_CFG_BASE,
    "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-aarch64.so",
    "execution": f"func::execution={ARM_SIMULATOR_DIR}/libarm.so",  # Need ARM simulator
    # ARM specific parameters (NEON widths, etc.)
}
```

**Implementation Location**: `src/iminnt/core.py:30-69` and `_init_sim_cfg()` method

---

### 5. Simulation Method Updates

#### Current:
```python
def simulate(self, ...):
    if self.use_qemu:  # ← Only True for RISC-V
        # Full Permafrost simulation
    else:
        # Direct execution (no simulation)
```

#### Required Changes:

```python
def simulate(self, ...):
    # Determine target architecture
    target_arch = self._get_target_architecture()  # riscv64, x86_64, aarch64
    
    if target_arch in ["riscv64", "x86_64", "aarch64"]:  # ← Support multiple archs
        cfg_path = self._init_sim_cfg(bin_path, bin_args, ..., arch=target_arch)
        sim_cmd = [str(PERMAFROST_BIN), str(cfg_path)]
        shell(sim_cmd, ...)
    else:
        # Unsupported architecture
        raise NotImplementedError(f"Simulation not supported for {target_arch}")
```

**Implementation Location**: `src/iminnt/core.py:1119-1198` (`simulate()` method)

---

### 6. Target Configuration Updates

#### Current:
```python
@register_class("llama_x86")
class LlamaCppX86(LlamaCppBase):
    def __init__(self):
        super().__init__(
            ...
            use_qemu=False  # ← Prevents simulation
        )
```

#### Required Changes:

**Option A: Enable QEMU for x86 simulation**
```python
@register_class("llama_x86")
class LlamaCppX86(LlamaCppBase):
    def __init__(self):
        super().__init__(
            ...
            use_qemu=True  # ← Enable QEMU for simulation (but native exec for run)
        )
    
    @property
    def target_arch(self):
        return "x86_64"  # ← Architecture identifier
```

**Option B: Separate simulation mode**
```python
def simulate(self, ...):
    # Always use QEMU/simulator for simulation, even for x86
    # But use native execution for "run" command
```

**Implementation Location**: `src/iminnt/llamacpp.py:425-564` (target classes)

---

### 7. Build System Updates

#### Required Changes:

**QEMU Build**: Ensure QEMU is built with x86 and ARM support
- Check: `dev_env/csqemu-v9/` - verify build configuration
- May need to rebuild QEMU with `--target-list=x86_64-linux-user,aarch64-linux-user`

**ISA Simulators**: Build/install x86 and ARM simulators
- If using Gem5: Build Gem5 with x86/ARM support
- If using custom simulators: Add build targets to infrastructure

**Performance Models**: Build/install architecture-specific performance models
- If extending Pilos: Add x86/ARM microarchitecture models
- If using alternative: Integrate new performance models

**Implementation Location**: `src/iminnt/infra.py` (infrastructure build targets)

---

## Implementation Complexity

### Effort Estimate (Rough)

| Component | Complexity | Estimated Effort |
|-----------|------------|------------------|
| QEMU multi-arch support | Medium | 1-2 weeks |
| ISA simulator integration | High | 4-8 weeks |
| Performance model adaptation | Very High | 8-16 weeks |
| Configuration refactoring | Medium | 2-3 weeks |
| Testing & validation | High | 4-6 weeks |
| **Total** | **Very High** | **19-35 weeks** |

### Challenges

1. **Pilos is RISC-V specific**: Core performance model may need significant refactoring
2. **Spike is RISC-V only**: Need complete replacement or alternative approach
3. **Architecture differences**: x86 and ARM have fundamentally different microarchitectures
4. **Validation**: Need to validate simulation accuracy against real hardware
5. **Maintenance**: Supporting multiple architectures increases complexity

---

## Alternative Approaches

### Option 1: Use Gem5 Instead of Pilos

**Gem5** is a full-system simulator that already supports x86, ARM, and RISC-V:

**Pros:**
- Already supports multiple architectures
- Well-maintained and validated
- Has built-in performance models

**Cons:**
- Different simulation framework (would need to replace Permafrost integration)
- More complex setup
- Slower than Pilos for detailed microarchitecture simulation

**Required Work:**
- Replace or extend Permafrost with Gem5 integration
- Adapt iminn-tools to use Gem5 configuration system
- Significant refactoring of simulation pipeline

### Option 2: QEMU-only Functional Simulation

**Use QEMU for both functional and performance simulation**:

**Pros:**
- QEMU already supports x86, ARM, RISC-V
- No need for separate ISA simulator

**Cons:**
- QEMU's performance model is less detailed than Pilos
- May not provide cycle-accurate results
- Would need QEMU plugin modifications

### Option 3: Hybrid Approach

**Different simulators for different architectures**:
- RISC-V: Continue using current stack (QEMU + Spike + Pilos)
- x86: Use Gem5 or Intel Pin
- ARM: Use Gem5 or ARM Fast Models

**Pros:**
- Use best tool for each architecture
- Incremental implementation

**Cons:**
- Inconsistent simulation methodology across architectures
- Harder to compare results across architectures
- More complex configuration management

---

## Recommended Implementation Strategy

### Phase 1: Assessment (2-3 weeks)

1. **Evaluate existing tools**:
   - Check if QEMU in `dev_env/csqemu-v9/` supports x86/ARM
   - Evaluate Gem5 as alternative to Pilos
   - Assess Pilos extensibility for x86/ARM

2. **Prototype one architecture**:
   - Choose either x86 or ARM (x86 might be easier)
   - Create minimal prototype to validate approach

### Phase 2: Infrastructure (4-6 weeks)

1. **Refactor configuration system**:
   - Make `PERMAFROST_SIM_CFG` architecture-aware
   - Update `_init_sim_cfg()` to handle multiple architectures

2. **Add QEMU multi-arch support**:
   - Build QEMU with x86/ARM support
   - Update QEMU configuration paths

### Phase 3: ISA Simulator (6-12 weeks)

1. **Integrate ISA simulator**:
   - Choose approach (Gem5, QEMU-only, or custom)
   - Integrate with Permafrost/Arctic framework

2. **Update simulation pipeline**:
   - Modify `simulate()` method
   - Update trace generation

### Phase 4: Performance Model (8-16 weeks)

1. **Extend or replace performance model**:
   - If extending Pilos: Add x86/ARM microarchitecture models
   - If using Gem5: Integrate Gem5 performance model
   - If using alternative: Integrate new model

2. **Validate accuracy**:
   - Compare simulation results against real hardware
   - Tune parameters for accuracy

### Phase 5: Testing & Integration (4-6 weeks)

1. **End-to-end testing**:
   - Test full simulation pipeline for each architecture
   - Validate results against known benchmarks

2. **Documentation & examples**:
   - Update documentation
   - Create examples for each architecture

---

## Code Changes Required

### Files to Modify

1. **`src/iminnt/core.py`**:
   - `PERMAFROST_SIM_CFG` → Architecture-aware configuration
   - `_init_sim_cfg()` → Support multiple architectures
   - `simulate()` → Enable simulation for x86/ARM

2. **`src/iminnt/llamacpp.py`**:
   - Target classes → Add architecture detection
   - Consider enabling QEMU for x86/ARM targets

3. **`src/iminnt/infra.py`**:
   - Add x86/ARM simulator build targets
   - Update psim compound target

4. **`src/iminnt/constants.py`**:
   - Add x86/ARM architecture constants
   - Add simulator path constants

### New Files Needed

1. **Architecture detection utilities**
2. **Architecture-specific configuration templates**
3. **x86/ARM simulator integration modules**

---

## Prerequisites

### External Dependencies

1. **QEMU with x86/ARM support**:
   ```bash
   # Would need to build QEMU with:
   ./configure --target-list=x86_64-linux-user,aarch64-linux-user,riscv64-linux-user
   ```

2. **ISA Simulator**:
   - Gem5 (if chosen)
   - Or custom x86/ARM simulator

3. **Performance Model**:
   - Extended Pilos (if chosen)
   - Or Gem5 performance model
   - Or architecture-specific simulator

### Build Infrastructure

1. **CMake updates** for building multi-arch components
2. **Dependency management** for architecture-specific tools
3. **Testing infrastructure** for validating multi-arch simulation

---

## Summary

**Current State**: psim only supports RISC-V

**To Add x86/ARM Support**: Would require significant changes:
- ✅ QEMU multi-arch support (Medium effort)
- ⚠️ ISA simulator replacement (High effort)
- ⚠️ Performance model adaptation (Very High effort)
- ✅ Configuration refactoring (Medium effort)

**Estimated Total Effort**: 19-35 weeks of development

**Recommended Approach**: 
- Evaluate Gem5 as unified multi-arch solution
- Or start with QEMU-only functional simulation
- Incrementally add performance modeling

**Key Decision Points**:
1. Use Gem5 (easier, but different framework) or extend current stack (harder, but maintains consistency)
2. Start with one architecture (x86) before adding ARM
3. Functional simulation first, then add performance modeling

---

**Last Updated:** January 5, 2025
