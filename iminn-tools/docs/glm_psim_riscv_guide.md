# Running GLM-4.6V-Flash on psim with RISC-V Architecture

**Date:** January 5, 2025  
**Model:** GLM-4.6V-Flash (9.4B parameters)  
**Target Architecture:** RISC-V (IMI or RVV)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Setup Steps](#3-setup-steps)
4. [Running Simulation](#4-running-simulation)
5. [Optimization Strategies](#5-optimization-strategies)
6. [Results and Analysis](#6-results-and-analysis)
7. [Troubleshooting](#7-troubleshooting)
8. [Performance Expectations](#8-performance-expectations)

---

## 1. Overview

This guide explains how to run GLM-4.6V-Flash on psim (Permafrost/Pilos simulator) using RISC-V architecture. Since psim supports RISC-V, this is the recommended approach for cycle-accurate performance simulation of the GLM model.

**Important Note**: GLM-4.6V-Flash is a large model (9.4B parameters), so simulation will be **extremely slow** (potentially days). Consider starting with smaller models first to validate the workflow.

---

## 2. Prerequisites

### 2.1 Required Infrastructure

1. **psim infrastructure built**:
   - Permafrost (simulation orchestrator)
   - Pilos (performance model)
   - QEMU (RISC-V emulation)
   - Spike (RISC-V ISA simulator)
   - Arctic (simulation framework)

2. **llama.cpp RISC-V binary built**:
   - `llama_imi` target (IMI extensions)
   - Or `llama_rvv` target (standard RVV)

3. **Model files downloaded**:
   - `GLM-4.6V-Flash-Q4_K_M.gguf` (6.17 GB)
   - `mmproj-GLM-4.6V-Flash-Q8_0.gguf` (980 MB) - Optional for text-only

### 2.2 Verify Prerequisites

```bash
# Check if psim infrastructure exists
ls -d dev_env/Pilos dev_env/Permafrost dev_env/csqemu-v9 dev_env/IMachines_Spike

# Check if Permafrost binary exists
ls dev_env/Permafrost/build/permafrost

# Check if llama_imi binary is built
ls dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli

# Check if model files exist
ls dev_env/llama.cpp/models/GLM-4.6V-Flash-*
```

---

## 3. Setup Steps

### Step 1: Initialize psim Infrastructure (if not done)

```bash
# Initialize all simulation infrastructure components
iminnt -t psim init
```

This clones:
- `riscv-env` - RISC-V toolchain
- `Arctic` - Carbon simulation framework
- `Permafrost` - Simulation orchestrator
- `Pilos` - Performance model
- `Spike` - RISC-V ISA simulator
- `QEMU` - Emulation infrastructure
- `Simpoint` - Statistical sampling

### Step 2: Build psim Infrastructure (if not done)

**⚠️ Warning**: This can take **several hours** as each component is large and complex.

```bash
# Build all psim components (recommended)
iminnt -t psim build

# OR build individually (if you need more control)
iminnt -t riscv-env build
iminnt -t arctic build
iminnt -t permafrost build
iminnt -t pilos build
iminnt -t spike build
iminnt -t qemu build
```

### Step 3: Initialize llama.cpp for RISC-V

```bash
# Initialize llama.cpp repository
iminnt -t llama_imi init

# Build llama.cpp for RISC-V (with debug symbols for simulation)
iminnt -t llama_imi build
```

**Note**: The build should include debug symbols (`-g` flag) for proper trace generation during simulation.

### Step 4: Verify GLM Model Files

```bash
# Verify model files are present
cd /home/linhu/repo/iminn-tools/dev_env/llama.cpp/models
ls -lh GLM-4.6V-Flash-*

# Expected output:
# -rw-rw-r-- 1 linhu linhu 5.8G Jan  5 14:18 GLM-4.6V-Flash-Q4_K_M.gguf
# -rw-rw-r-- 1 linhu linhu 935M Jan  5 14:18 mmproj-GLM-4.6V-Flash-Q8_0.gguf
```

---

## 4. Running Simulation

### 4.1 Basic Simulation Command

```bash
# Run GLM simulation with IMI target
iminnt -t llama_imi sim -d test_glm_4_6v_text
```

**What this does:**
1. Loads the test configuration `test_glm_4_6v_text` from `llamacpp.py`
2. Uses RISC-V binary (`llamacpp-imi-install/bin/llama-cli`)
3. Creates Permafrost simulation configuration
4. Runs QEMU to generate execution traces
5. Uses Spike for functional simulation
6. Uses Pilos for cycle-accurate performance modeling
7. Stores results in `results/llama_imi/`

### 4.2 Command Options

```bash
# Specify output directory
iminnt -t llama_imi sim -d test_glm_4_6v_text -o results/glm_simulation_$(date +%Y%m%d)

# Print configuration before running (recommended for first run)
iminnt -t llama_imi sim -d test_glm_4_6v_text -p

# Keep retired instructions log (for detailed analysis)
iminnt -t llama_imi sim -d test_glm_4_6v_text -k

# Generate performance flamegraph
iminnt -t llama_imi sim -d test_glm_4_6v_text -g

# Disable ROI (Region of Interest) markers
iminnt -t llama_imi sim -d test_glm_4_6v_text -n

# Use custom simulation configuration
iminnt -t llama_imi sim -d test_glm_4_6v_text -c path/to/custom.cfg
```

### 4.3 Using Custom Arguments

```bash
# Run simulation with custom arguments
iminnt -t llama_imi sim \
    --bin llama-cli \
    --bin-args "-m dev_env/llama.cpp/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                --seed 42 -t 1 -ngl 0 -n 8 \
                -no-cnv -st --no-warmup \
                --file src/iminnt/resources/prompts/hello-world.txt" \
    -o results/glm_custom
```

**Note**: Reducing `-n 32` to `-n 8` or `-n 4` will significantly reduce simulation time.

### 4.4 Text-Only vs Multimodal

**Text-only mode** (no vision projection):
```bash
iminnt -t llama_imi sim -d test_glm_4_6v_text
```

**Multimodal mode** (with vision projection):
```bash
iminnt -t llama_imi sim -d test_glm_4_6v_multimodal
```

---

## 5. Optimization Strategies

Given that GLM-4.6V-Flash is a 9.4B parameter model, simulation will be very slow. Here are optimization strategies:

### 5.1 Reduce Generation Length

**Current**: `-n 32` tokens  
**Optimized**: `-n 8` or `-n 4` tokens

```python
# Add optimized test configuration
"test_glm_4_6v_text_minimal": {"bin": f"{self.install_dir}/bin/llama-cli", 
    "args": f"-m {self.root}/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                --seed 42 \
                -t 1 -ngl 0 -n 4 \
                -no-cnv -st --no-warmup \
                --file {PROMPTS_DIR}/hello-world.txt"},
```

### 5.2 Reduce Context Size

Add `-c 512` to limit context window:

```bash
iminnt -t llama_imi sim --bin llama-cli \
    --bin-args "-m dev_env/llama.cpp/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                -c 512 -n 8 ..."
```

### 5.3 Use ROI (Region of Interest) Markers

Focus simulation on specific parts of execution:

**ROI is enabled by default** when using `sim` command. The ROI markers (controlled by `IMI_ROI_SIM` environment variable) allow you to:
- Skip initialization overhead
- Focus on inference kernel execution
- Reduce simulation time

```bash
# ROI is automatically enabled for simulation
# The environment variable IMI_ROI_SIM is set automatically
iminnt -t llama_imi sim -d test_glm_4_6v_text

# To disable ROI (simulate everything):
iminnt -t llama_imi sim -d test_glm_4_6v_text -n
```

### 5.4 Use Simpoint for Statistical Sampling

For very long-running simulations, use Simpoint to identify representative execution intervals:

```bash
# First, run simpoint analysis
iminnt -t llama_imi simpoint -d test_glm_4_6v_text

# Then use simpoint results for faster simulation
# (This requires additional configuration)
```

### 5.5 Start with Smaller Models

**Strongly Recommended**: Validate the workflow with a smaller model first:

```bash
# Test with stories15M (15M parameters) first
iminnt -t llama_imi sim -d test_q4_0_stories

# If that works, then try GLM
iminnt -t llama_imi sim -d test_glm_4_6v_text
```

---

## 6. Results and Analysis

### 6.1 Output Location

Simulation results are stored in:
```
results/llama_imi/
├── iminnt.cfg                    # Simulation configuration used
├── simbench.log                  # Simulation execution log
├── stats/
│   └── pilos_combined.stats      # Cycle counts and performance metrics
├── retires.log                   # Instruction trace (if -k flag used)
└── trace.json                    # Flamegraph data (if -g flag used)
```

If custom output directory specified (`-o` flag):
```
results/<custom_dir>/
└── (same structure as above)
```

### 6.2 Extracting Results

**Cycle Count Extraction**:
```python
from src.iminnt.utils import get_cycles
from pathlib import Path

results_dir = Path("results/llama_imi")
cycles_info = get_cycles(results_dir, return_perf_info=True)
print(f"Total cycles: {cycles_info}")
```

**Viewing Stats**:
```bash
# View simulation stats
cat results/llama_imi/stats/pilos_combined.stats

# View simulation log
less results/llama_imi/simbench.log
```

**Flamegraph Visualization** (if `-g` flag used):
```bash
# Open in Chrome
# Navigate to chrome://tracing
# Load results/llama_imi/trace.json

# Or use Perfetto UI
# https://ui.perfetto.dev/
# Upload trace.json
```

### 6.3 Key Metrics

The simulation provides:
- **Total cycles**: Total simulated cycles
- **Instructions per cycle (IPC)**: Pipeline efficiency
- **Memory stall cycles**: Cache/memory subsystem performance
- **Vector unit utilization**: RISC-V vector extension usage
- **Branch prediction accuracy**: Control flow performance
- **Cache hit/miss rates**: Memory hierarchy performance

---

## 7. Troubleshooting

### Issue: psim Infrastructure Not Built

**Error**: `permafrost: command not found` or missing directories

**Solution**:
```bash
# Check if psim is initialized
ls dev_env/Permafrost dev_env/Pilos

# If missing, initialize and build
iminnt -t psim init
iminnt -t psim build
```

### Issue: Binary Not Found

**Error**: `No such binary executable at path ...`

**Solution**:
```bash
# Rebuild llama_imi target
iminnt -t llama_imi build

# Verify binary exists
ls dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli
```

### Issue: Model File Not Found

**Error**: `llama_model_load: failed to load model`

**Solution**:
```bash
# Verify model files exist
ls -lh dev_env/llama.cpp/models/GLM-4.6V-Flash-*

# Re-download if missing (see glm_4_6v_flash_usage.md)
```

### Issue: Simulation Takes Too Long

**Symptom**: Simulation appears to hang or runs for days

**Solutions**:
1. **Reduce tokens**: Use `-n 4` instead of `-n 32`
2. **Use ROI markers**: Ensure ROI is enabled (default)
3. **Test with smaller model first**: Validate with `stories15M`
4. **Consider Simpoint**: Use statistical sampling for long runs
5. **Monitor progress**: Check `simbench.log` for activity

### Issue: Out of Memory During Simulation

**Error**: Simulation crashes or runs out of memory

**Solutions**:
1. **Reduce context size**: Add `-c 256` or `-c 512`
2. **Reduce batch size**: This may require code modifications
3. **Use system with more RAM**: GLM model requires ~6-7 GB

### Issue: Simulation Configuration Errors

**Error**: Permafrost configuration errors

**Solution**:
```bash
# Print configuration to debug
iminnt -t llama_imi sim -d test_glm_4_6v_text -p

# Check configuration file
cat results/llama_imi/iminnt.cfg

# Verify paths in configuration
grep -E "SIM_BIN|PK_BIN|COSIM_LIB" results/llama_imi/iminnt.cfg
```

---

## 8. Performance Expectations

### Execution Time Estimates

**For GLM-4.6V-Flash (9.4B parameters)**:

| Configuration | Estimated Time | Notes |
|--------------|----------------|-------|
| Full run (`-n 32`) | **Days to weeks** | Not recommended for initial testing |
| Minimal (`-n 4`) | **Hours to days** | More reasonable for testing |
| With ROI enabled | **Hours** | Recommended approach |
| With Simpoint | **Minutes to hours** | Fastest, but requires setup |

**For comparison, stories15M (15M parameters)**:
- Simulation time: **Minutes to hours**
- Good for workflow validation

### Resource Requirements

- **RAM**: 8-16 GB recommended
- **Disk Space**: ~10-20 GB for simulation outputs
- **CPU**: Single-threaded simulation (uses 1 core)

### When to Use Simulation vs Emulation

**Use `run` (QEMU emulation)** when:
- Testing functionality
- Quick iteration
- Model validation

**Use `sim` (full simulation)** when:
- Need cycle-accurate performance
- Analyzing microarchitecture impact
- Optimizing kernels
- Validating RISC-V extensions

---

## 9. Step-by-Step Workflow Example

### Complete Workflow

```bash
# 1. Verify prerequisites
iminnt -t psim defaults  # Check if psim is available
iminnt -t llama_imi defaults  # Check if llama_imi is available

# 2. Ensure infrastructure is built (if needed)
iminnt -t psim build  # This takes hours - only do if needed

# 3. Test with small model first (RECOMMENDED)
iminnt -t llama_imi sim -d test_q4_0_stories -o results/test_workflow

# 4. Verify simulation worked
ls results/test_workflow/stats/pilos_combined.stats

# 5. Run GLM simulation with minimal tokens
iminnt -t llama_imi sim -d test_glm_4_6v_text -o results/glm_minimal -k -g

# 6. Monitor progress
tail -f results/glm_minimal/simbench.log

# 7. Extract results
python3 -c "from src.iminnt.utils import get_cycles; from pathlib import Path; print(get_cycles(Path('results/glm_minimal')))"

# 8. Visualize (if -g flag used)
# Open chrome://tracing and load results/glm_minimal/trace.json
```

---

## 10. Configuration Reference

### Test Configuration

**Location**: `src/iminnt/llamacpp.py:404-409`

```python
"test_glm_4_6v_text": {
    "bin": f"{self.install_dir}/bin/llama-cli", 
    "args": f"-m {self.root}/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                --seed 42 \
                -t 1 -ngl 0 -n 32 \
                -no-cnv -st --no-warmup \
                --file {PROMPTS_DIR}/hello-world.txt"
}
```

### Simulation Configuration

**Location**: `src/iminnt/core.py:30-69` (`PERMAFROST_SIM_CFG`)

Key parameters:
- **VLEN=128**: Vector length (128 bits)
- **12 execution ports**: Wide superscalar pipeline
- **1 GHz clock**: Performance timing model
- **ROI enabled**: Region of Interest markers by default

---

## 11. Alternative: Using RVV Target

If you want to use standard RISC-V Vector extension instead of IMI:

```bash
# Build with RVV target
iminnt -t llama_rvv build

# Run simulation
iminnt -t llama_rvv sim -d test_glm_4_6v_text
```

**Differences**:
- Uses standard RISC-V Vector Extension (no IMI custom instructions)
- Still uses QEMU + Spike + Pilos simulation stack
- Same workflow, just different target

---

## 12. Best Practices

1. **Start Small**: Always test with `stories15M` first
2. **Use ROI**: Keep ROI enabled for faster simulation
3. **Reduce Tokens**: Use `-n 4` or `-n 8` for initial testing
4. **Monitor Progress**: Check `simbench.log` regularly
5. **Save Outputs**: Use `-o` flag to organize results
6. **Generate Traces**: Use `-g` flag for flamegraph analysis
7. **Keep Retires**: Use `-k` flag for detailed instruction analysis
8. **Validate First**: Ensure `run` works before attempting `sim`

---

## 13. Code References

- **psim definition**: `src/iminnt/infra.py:430-434`
- **Simulation method**: `src/iminnt/core.py:1119-1198`
- **Simulation config**: `src/iminnt/core.py:30-69`
- **GLM test config**: `src/iminnt/llamacpp.py:404-416`
- **IMI target**: `src/iminnt/llamacpp.py:446-464`
- **RVV target**: `src/iminnt/llamacpp.py:493-510`

---

## Summary

**To run GLM on psim with RISC-V:**

1. ✅ Build psim infrastructure: `iminnt -t psim init && iminnt -t psim build`
2. ✅ Build llama_imi: `iminnt -t llama_imi build`
3. ✅ Verify model files: Check `dev_env/llama.cpp/models/GLM-4.6V-Flash-*`
4. ⚠️ **Start with smaller model**: Test with `test_q4_0_stories` first
5. ✅ Run simulation: `iminnt -t llama_imi sim -d test_glm_4_6v_text -o results/glm_test`
6. ⏰ **Be patient**: Simulation will take hours to days
7. 📊 Analyze results: Check `results/<dir>/stats/pilos_combined.stats`

**Key Points:**
- ✅ psim fully supports RISC-V (IMI and RVV)
- ⚠️ GLM is very large - simulation will be extremely slow
- 💡 Start with smaller models to validate workflow
- 📈 Use optimization strategies to reduce simulation time

---

**Last Updated:** January 5, 2025
