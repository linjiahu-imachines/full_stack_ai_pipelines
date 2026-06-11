# Fixing QEMU SIGSEGV During psim Simulation

**Date:** January 5, 2025  
**Issue:** QEMU SIGSEGV error during cosimulation  
**Error:** `qemu.so: QEMU internal SIGSEGV {code=MAPERR, addr=0xffffffffffffd8d0}`

---

## Problem Description

When running simulation with psim, QEMU crashes with a segmentation fault:

```
qemu.so: QEMU internal SIGSEGV {code=MAPERR, addr=0xffffffffffffd8d0}
```

This occurs during the cosimulation phase when QEMU is trying to execute the binary with the `libcosim.so` plugin.

---

## Root Cause

The SIGSEGV is likely caused by:

1. **Memory mapping issue** in QEMU's cosimulation plugin
2. **Compatibility issue** between QEMU version and the cosim plugin
3. **Binary execution path** issues during cosimulation
4. **ROI marker handling** causing memory access violations

---

## Solutions

### Solution 1: Disable ROI (Region of Interest) Markers

**Try running simulation without ROI first**:

```bash
# Use -n flag to disable ROI
iminnt -t llama_imi sim -d test_q4_0_stories -n -o results/test_no_roi
```

**What this does:**
- Removes `IMI_ROI_SIM="1"` environment variable
- Simulates the entire execution (slower, but may avoid the crash)
- Uses `wait_for_roi=0` and `listen_to_roi=false` in configuration

**If this works**, the issue is with ROI marker handling. You can then:
- Investigate ROI marker implementation
- Use simulation without ROI (slower but functional)
- Report the issue to the QEMU/cosim plugin maintainers

### Solution 2: Check QEMU and Plugin Compatibility

**Verify QEMU and cosim plugin versions match**:

```bash
# Check QEMU version
/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-local/bin/qemu-riscv64 --version

# Check cosim plugin exists and is compatible
ls -lh /home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-local/plugins/libcosim.so
ldd /home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-local/plugins/libcosim.so
```

**If versions don't match or plugin is missing dependencies:**
- Rebuild QEMU and plugins: `iminnt -t qemu build`
- Ensure cosim plugin is built with same QEMU version

### Solution 3: Use Static Binary (Already Done)

**The binary is already statically linked**, which is good:

```bash
file dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli
# Should show: statically linked
```

**If binary was dynamically linked**, you would need:
- Sysroot linking: `-L {CROSS_SYSROOT}` in QEMU command
- This is automatically handled by the code, but verify it's working

### Solution 4: Try Different QEMU Options

**Modify QEMU invocation to avoid problematic options**:

The current QEMU command includes:
- `-one-insn-per-tb`: One instruction per translation block
- `-d nochain`: Disable chaining
- `-umode`: User mode
- `-internal-syscall`: Internal syscall handling
- `-cosim`: Cosimulation mode
- `-plugin libcosim.so`: Cosimulation plugin

**Potential workaround**: Create a custom configuration that removes problematic options:

```bash
# Create custom config
cat > /tmp/custom_sim.cfg << 'EOF'
-carbon_set=SIM_BIN,/path/to/llama-cli
-carbon_set=COSIM_LIB,/path/to/libcosim.so
func::live_trace=/path/to/qemu-riscv64.so
func::memory=/path/to/qemu-riscv64.so
func::execution=/path/to/libriscv.so
func::argv=qemu.so -umode -cosim -plugin $carbon<COSIM_LIB> -cpu imicpu-v1 $carbon<SIM_BIN> $SIM_ARGS$
# ... other config ...
EOF

# Use custom config
iminnt -t llama_imi sim -d test_q4_0_stories -c /tmp/custom_sim.cfg
```

### Solution 5: Check Memory and System Resources

**Ensure sufficient system resources**:

```bash
# Check available memory
free -h

# Check if system is swapping
swapon --show

# Check QEMU process limits
ulimit -a
```

**If memory is low:**
- Close other applications
- Increase swap space
- Reduce model size or context window

### Solution 6: Debug with GDB

**Attach debugger to understand the crash**:

```bash
# Run with gdb
gdb --args /home/linhu/repo/iminn-tools/dev_env/Permafrost/build/permafrost \
    /home/linhu/repo/iminn-tools/results/results/test_workflow/iminnt.cfg

# In gdb:
(gdb) run
# When it crashes:
(gdb) bt
(gdb) info registers
(gdb) x/10i $pc
```

This will show the exact instruction causing the crash.

---

## Recommended Troubleshooting Steps

### Step 1: Try Without ROI (Easiest)

```bash
iminnt -t llama_imi sim -d test_q4_0_stories -n -o results/test_no_roi -k -g
```

**If this works**, the issue is ROI-related. Continue with ROI disabled or investigate ROI implementation.

### Step 2: Verify QEMU Build

```bash
# Rebuild QEMU to ensure everything is up to date
iminnt -t qemu build
```

### Step 3: Test with Minimal Configuration

```bash
# Use minimal test (fewer tokens)
iminnt -t llama_imi sim -d test_q4_0_stories_minimal -n -o results/test_minimal
```

### Step 4: Check Logs

```bash
# Check simulation log for more details
cat results/results/test_workflow/simbench.log

# Check if there are any warnings before the crash
grep -i "warning\|error\|fatal" results/results/test_workflow/simbench.log
```

### Step 5: Report Issue

If none of the above work, collect:
- Full error output
- Configuration file: `results/results/test_workflow/iminnt.cfg`
- QEMU version: `qemu-riscv64 --version`
- System information: `uname -a`, `free -h`
- GDB backtrace (if available)

---

## Workaround: Use `run` Instead of `sim`

**If simulation continues to fail**, you can still use QEMU emulation for functional testing:

```bash
# This works (as shown in your test)
iminnt -t llama_imi run -d test_q4_0_stories

# This provides:
# - Functional correctness testing
# - Execution time measurement (approximate)
# - No cycle-accurate performance simulation
```

**Limitations:**
- No cycle-accurate performance metrics
- No detailed microarchitecture analysis
- No Pilos performance model results

---

## Code References

- **ROI flag definition**: `src/iminnt/core.py:18`
- **Simulation config**: `src/iminnt/core.py:30-69`
- **ROI handling**: `src/iminnt/core.py:993-1003`
- **Static binary check**: `src/iminnt/core.py:1005-1017`

---

## Summary

**Quick Fix**: Try disabling ROI first:
```bash
iminnt -t llama_imi sim -d test_q4_0_stories -n -o results/test_no_roi
```

**If that doesn't work**:
1. Rebuild QEMU: `iminnt -t qemu build`
2. Check system resources
3. Debug with GDB
4. Use `run` command as workaround

**Expected Outcome**: Simulation should complete without SIGSEGV, though it may be slower without ROI optimization.

---

**Last Updated:** January 5, 2025
