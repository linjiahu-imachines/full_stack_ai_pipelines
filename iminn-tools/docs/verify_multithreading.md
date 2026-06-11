# How to Verify Multi-Threading in llama.cpp

This document explains multiple methods to verify that llama.cpp is actually using multiple threads during execution.

---

## Method 1: Check llama.cpp Output (Easiest)

### Look for Thread Count in Output

When llama.cpp runs, it prints the thread count in the output. Look for these lines:

```bash
iminnt -t llama_imi run -d test_q4_0_stories_4t
```

**Key lines to look for:**

```
main: llama threadpool init, n_threads = 4
system_info: n_threads = 4 (n_threads_batch = 4) / 28 | CPU : RISCV_V = 1 | IMI = 1
```

**What this tells you:**
- `n_threads = 4` - Decode (token generation) uses 4 threads
- `n_threads_batch = 4` - Prefill (prompt processing) uses 4 threads
- `/ 28` - Total available CPU cores (on the host machine)

**Compare single-threaded vs multi-threaded:**

```bash
# Single-threaded
iminnt -t llama_imi run -d test_q4_0_stories
# Output: main: llama threadpool init, n_threads = 1

# Multi-threaded
iminnt -t llama_imi run -d test_q4_0_stories_4t
# Output: main: llama threadpool init, n_threads = 4
```

---

## Method 2: Monitor CPU Usage (Most Visual)

### Using `htop` to See Thread Activity

While the test is running, monitor CPU usage in another terminal:

**Terminal 1 - Run the test:**
```bash
iminnt -t llama_x86 run -d test_q4_0_stories_4t
```

**Terminal 2 - Monitor CPU:**
```bash
htop
# Press 'H' to show individual threads
# Press 'F5' to switch to tree view
```

**What to look for:**
- **Single-threaded (1 thread):** You'll see 1 CPU core at ~100%, others idle
- **Multi-threaded (4 threads):** You'll see 4 CPU cores active, each at high utilization

**For RISC-V via QEMU:**

QEMU itself is multi-threaded, so you'll see the `qemu-riscv64` process with multiple threads:

```bash
# In another terminal while test is running
ps -eLf | grep qemu-riscv64
# You should see multiple thread IDs (LWP column)
```

### Using `top` (Simpler Alternative)

```bash
top -H -p $(pgrep -f llama-cli)
# -H shows threads
# Look for multiple entries with high CPU%
```

---

## Method 3: Time Comparison (Proof of Speedup)

The most definitive proof is that multi-threaded execution is faster.

### Benchmark Single-Threaded vs Multi-Threaded

```bash
# Single-threaded baseline
time iminnt -t llama_x86 run -d test_q4_0_stories

# 2 threads
time iminnt -t llama_x86 run -d test_q4_0_stories_2t

# 4 threads
time iminnt -t llama_x86 run -d test_q4_0_stories_4t
```

**Expected results:**
- 2 threads should be ~1.5-1.8x faster than 1 thread
- 4 threads should be ~2.5-3.5x faster than 1 thread (for prefill)

**Example output comparison:**

```
# 1 thread
real    0m8.500s

# 2 threads
real    0m5.200s  (1.6x speedup)

# 4 threads
real    0m3.100s  (2.7x speedup)
```

### Using llama-bench for Precise Measurements

```bash
# 1 thread baseline
iminnt -t llama_x86 run -d stories_prefill_bench
# Output: pp512: 150.23 t/s (tokens per second)

# 4 threads
iminnt -t llama_x86 run -d stories_prefill_bench_4t
# Output: pp512: 380.45 t/s (2.5x faster)
```

---

## Method 4: Use `perf` to Analyze Threading (Advanced)

### Record Performance Counters

```bash
# Run with perf (x86 only, won't work with QEMU)
perf record -e cycles,instructions -g -- \
    dev_env/llama.cpp/llamacpp-x86-install/bin/llama-cli \
    -m dev_env/llama.cpp/models/stories15M-q4_0.gguf -t 4 -n 32

# View report
perf report
```

**What to look for:**
- Multiple threads in call stacks
- Thread synchronization functions (`pthread_barrier_wait`, `ggml_barrier`)
- CPU utilization across multiple cores

---

## Method 5: Check Thread Creation in Code (Debug Mode)

### Enable Verbose Logging

Add `-v` flag to see more detailed output:

```bash
iminnt -t llama_x86 run --bin llama-cli --args "-m models/stories15M-q4_0.gguf -t 4 -n 32 -v"
```

### Use strace to See System Calls (Linux)

```bash
# For x86 native
strace -f -e clone,futex \
    dev_env/llama.cpp/llamacpp-x86-install/bin/llama-cli \
    -m dev_env/llama.cpp/models/stories15M-q4_0.gguf -t 4 -n 32 2>&1 | \
    grep clone

# You should see 'clone' syscalls for thread creation
```

**Expected output:**
```
[pid xxxxx] clone(child_stack=..., flags=CLONE_VM|CLONE_FS|...) = xxxxx
[pid xxxxx] clone(child_stack=..., flags=CLONE_VM|CLONE_FS|...) = xxxxx
[pid xxxxx] clone(child_stack=..., flags=CLONE_VM|CLONE_FS|...) = xxxxx
```

Each `clone` syscall creates a new thread.

---

## Method 6: Programmatic Verification Script

Here's a simple script to test and verify multi-threading:

```bash
#!/bin/bash
# File: verify_multithreading.sh

echo "=== Multi-Threading Verification Test ==="
echo ""

# Test configurations
THREADS=(1 2 4)
TARGET="llama_x86"  # Change to llama_imi for RISC-V testing
MODEL="stories15M-q4_0.gguf"

for T in "${THREADS[@]}"; do
    echo "Testing with $T thread(s)..."

    # Run and capture output
    OUTPUT=$(iminnt -t $TARGET run --bin llama-cli --args \
        "-m dev_env/llama.cpp/models/$MODEL -t $T -n 32 --seed 42 -ngl 0 -no-cnv -st --no-warmup" \
        2>&1)

    # Extract thread count from output
    ACTUAL_THREADS=$(echo "$OUTPUT" | grep "n_threads = " | head -1 | grep -oP 'n_threads = \K\d+')

    # Extract performance
    TOKENS_PER_SEC=$(echo "$OUTPUT" | grep "eval time" | grep -oP '\K[\d.]+(?= tokens per second)')

    # Verify
    if [ "$ACTUAL_THREADS" == "$T" ]; then
        echo "  ✓ Thread count verified: $ACTUAL_THREADS"
    else
        echo "  ✗ Thread count mismatch! Expected: $T, Got: $ACTUAL_THREADS"
    fi

    echo "  Performance: $TOKENS_PER_SEC tokens/second"
    echo ""
done

echo "=== Test Complete ==="
```

**Usage:**
```bash
chmod +x verify_multithreading.sh
./verify_multithreading.sh
```

**Expected output:**
```
=== Multi-Threading Verification Test ===

Testing with 1 thread(s)...
  ✓ Thread count verified: 1
  Performance: 8.45 tokens/second

Testing with 2 thread(s)...
  ✓ Thread count verified: 2
  Performance: 13.72 tokens/second

Testing with 4 thread(s)...
  ✓ Thread count verified: 4
  Performance: 22.18 tokens/second

=== Test Complete ===
```

---

## Quick Verification Checklist

**✓ Check llama.cpp output:**
```bash
iminnt -t llama_x86 run -d test_q4_0_stories_4t | grep "n_threads"
# Should show: n_threads = 4
```

**✓ Compare execution time:**
```bash
time iminnt -t llama_x86 run -d test_q4_0_stories    # Baseline
time iminnt -t llama_x86 run -d test_q4_0_stories_4t # Should be faster
```

**✓ Monitor CPU usage:**
```bash
# Terminal 1:
iminnt -t llama_x86 run -d test_q4_0_stories_4t

# Terminal 2:
htop  # Press 'H' to see threads, should see 4 cores busy
```

**✓ Check thread count in process:**
```bash
# While test is running in another terminal:
ps -eLf | grep llama-cli | wc -l
# Should show 5 or more lines (1 main + 4 worker threads + header)
```

---

## Expected Performance Scaling

### Prefill (Batch Processing)

Thread scaling is good for prefill because of high parallelism in GEMM:

| Threads | Expected Speedup | Example Performance |
|---------|------------------|---------------------|
| 1       | 1.0x (baseline)  | 150 tokens/sec      |
| 2       | 1.7-1.9x         | 255-285 tokens/sec  |
| 4       | 2.5-3.5x         | 375-525 tokens/sec  |
| 8       | 3.5-5.0x         | 525-750 tokens/sec  |

### Decode (Single Token Generation)

Thread scaling is more limited due to GEMV characteristics:

| Threads | Expected Speedup | Example Performance |
|---------|------------------|---------------------|
| 1       | 1.0x (baseline)  | 12 tokens/sec       |
| 2       | 1.5-1.7x         | 18-20 tokens/sec    |
| 4       | 2.0-2.5x         | 24-30 tokens/sec    |
| 8       | 2.2-2.8x         | 26-34 tokens/sec    |

**Note:** Decode has diminishing returns beyond 4 threads due to:
- Lower parallelism in GEMV operations
- Barrier synchronization overhead
- Memory bandwidth limits

---

## Troubleshooting: Not Using Multiple Threads?

### Problem: Output shows `n_threads = 1` despite using `-t 4`

**Possible causes:**

1. **Using wrong default command:**
   ```bash
   # Wrong - uses single-threaded default
   iminnt -t llama_imi run -d test_q4_0_stories

   # Correct - uses 4-threaded variant
   iminnt -t llama_imi run -d test_q4_0_stories_4t
   ```

2. **Build issue - not rebuilt after changes:**
   ```bash
   # Rebuild to pick up new configuration
   iminnt -t llama_imi build
   ```

3. **Override not working:**
   ```bash
   # Make sure to use correct syntax
   iminnt -t llama_imi run --bin llama-cli --args "-t 4 -m models/stories15M-q4_0.gguf -n 32"
   ```

### Problem: No performance improvement with more threads

**Possible causes:**

1. **Model too small:** stories15M is tiny, overhead may dominate
   - Try larger models for better scaling

2. **QEMU overhead:** For RISC-V testing via QEMU
   - QEMU user-mode adds overhead that can mask threading benefits
   - Test on x86 native first to verify scaling

3. **Memory bandwidth saturation:**
   - Check with `perf stat -e cache-misses,cache-references`

4. **Wrong benchmark:** Decode doesn't scale as well as prefill
   - Use prefill benchmarks (`stories_prefill_bench_*`) to see better scaling

---

## Recommended Verification Workflow

**Step 1:** Quick check - Look at output
```bash
iminnt -t llama_x86 run -d test_q4_0_stories_4t | grep "n_threads"
```

**Step 2:** Performance comparison
```bash
time iminnt -t llama_x86 run -d test_q4_0_stories    # 1 thread
time iminnt -t llama_x86 run -d test_q4_0_stories_4t # 4 threads
```

**Step 3:** Visual confirmation (if needed)
```bash
# Terminal 1: Run test
iminnt -t llama_x86 run -d test_q4_0_stories_4t

# Terminal 2: Monitor
htop  # Press 'H' to see threads
```

This 3-step process should give you high confidence that multi-threading is working!
