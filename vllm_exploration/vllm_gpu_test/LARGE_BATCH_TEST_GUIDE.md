# Large Batch Performance Test: Batch Size 500 (CPU & GPU)

**Test Date**: February 17, 2026
**Batch Size**: 500 (upgraded from original batch size of 5)
**Model**: facebook/opt-125m
**Hardware**: 2x Intel Xeon Gold 6242R (40 physical cores / 80 logical threads with HT), 251.5 GB RAM, 4x NVIDIA TITAN RTX (24 GB each)

---

## Overview

This test suite compares **Transformers (HuggingFace)** vs **vLLM** inference with a
production-realistic batch size of 500, and adds **CPU core utilization monitoring** to
answer how many of the 80 CPU cores are actually used during inference.

### What Changed from the Original Test (Batch Size 5)

| Aspect                | Original Test (Batch 5) | New Test (Batch 500) |
|-----------------------|-------------------------|----------------------|
| Batch Size            | 5                       | **500** (100x larger)|
| CPU Core Monitoring   | No                      | **Yes**              |
| Active Core Count     | Unknown                 | **Tracked**          |
| Per-Core Utilization  | No                      | **Yes (top 10)**     |
| Thread Count          | No                      | **Monitored**        |
| Memory Tracking       | No                      | **Per-process**      |
| Production Relevance  | Low                     | **High**             |

---

## How to Run

### Prerequisites

The test uses the existing virtual environment in `vllm_gpu_test/venv` which already
has `vllm`, `transformers`, and `torch` installed. The only additional dependency is
`psutil` for CPU monitoring.

### Step 1: Navigate to the test directory

```bash
cd /home/linhu/projects/vllm_exploration/vllm_gpu_test
```

### Step 2: Activate the virtual environment

```bash
source venv/bin/activate
```

### Step 3: Install psutil (if not already installed)

```bash
pip install psutil
```

### Step 4: Run the test

**Full test (CPU + GPU, batch 500):**

```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN python3 test_large_batch.py --batch-size 500
```

> **Important**: The `VLLM_ATTENTION_BACKEND=TRITON_ATTN` environment variable is
> required because the NVIDIA TITAN RTX (compute capability 7.5) does not support
> FlashAttention-2 (requires compute >= 8.0), and the default FlashInfer backend
> has a broken JIT compilation on this system. TRITON_ATTN is the compatible fallback.

**Other options:**

```bash
# Custom batch size
VLLM_ATTENTION_BACKEND=TRITON_ATTN python3 test_large_batch.py --batch-size 1000

# CPU tests only (no GPU, no vLLM needed)
python3 test_large_batch.py --batch-size 500 --cpu-only

# GPU tests only (skip CPU)
VLLM_ATTENTION_BACKEND=TRITON_ATTN python3 test_large_batch.py --batch-size 500 --gpu-only
```

### Alternative: Use the convenience script

```bash
./run_large_batch_test.sh
```

This script activates the venv, checks the environment, and runs the full test.

### Expected Duration

- CPU Transformers test: ~30 seconds
- vLLM GPU initialization: ~25 seconds (first-time warmup)
- vLLM GPU inference: ~1 second
- **Total**: ~80 seconds

---

## What the Test Measures

### Part 1: CPU Tests

**Transformers on CPU** (batch 500):
- Total inference time and throughput
- CPU core utilization (via `psutil`):
  - Number of active cores (>5% utilization)
  - Average CPU utilization across all cores
  - Per-core utilization (top 10 most busy cores)
  - Thread count
  - Process memory usage

**vLLM on CPU**: Skipped. vLLM is a GPU-native inference engine and does not support
CPU-only inference.

### Part 2: GPU Tests

**Transformers on GPU** (batch 500):
- Total inference time and throughput
- Peak GPU memory usage
- Processes batch in chunks of 50 to avoid OOM

**vLLM on GPU** (batch 500):
- Total inference time and throughput
- Peak GPU memory usage
- Uses PagedAttention and continuous batching natively

### Final Summary

The test prints a side-by-side comparison of all results, including:
- CPU core utilization breakdown
- GPU performance comparison (Transformers vs vLLM)
- Winner and speedup factor

---

## Actual Test Results (February 17, 2026)

### System Information

```
CPU: 2x Intel Xeon Gold 6242R @ 3.10 GHz
     40 physical cores, 80 logical threads (Hyper-Threading enabled)
RAM: 251.5 GB
GPU: NVIDIA TITAN RTX (23.6 GB)
Model: facebook/opt-125m
Batch Size: 500
```

### CPU Results: Transformers

| Metric                        | Value                |
|-------------------------------|----------------------|
| Total time                    | **31.98s**           |
| Time per prompt               | 63.97ms              |
| Throughput                    | **15.63 prompts/sec**|
| Active logical threads (>5% util) | **79 out of 80 (98.8%)**|
| Average total CPU utilization | 23.6%                |
| Max threads                   | 186                  |
| Average memory                | 2,721 MB             |

**Top 10 most utilized cores:**

| Core    | Utilization |
|---------|-------------|
| Core 0  | 44.3%       |
| Core 17 | 39.3%       |
| Core 4  | 38.3%       |
| Core 59 | 36.3%       |
| Core 55 | 36.3%       |
| Core 57 | 35.8%       |
| Core 15 | 35.8%       |
| Core 3  | 35.8%       |
| Core 51 | 35.4%       |
| Core 49 | 35.1%       |

**Key Finding**: PyTorch uses **79 of 80 logical threads** across all 40 physical cores
(98.8%), but the average per-thread utilization is only **23.6%** and the busiest thread
only reaches **44.3%**. This means PyTorch spreads the work across nearly all available
hardware threads, but each is lightly loaded -- the workload is not CPU-compute-bound
but rather limited by memory bandwidth and model computation sequencing. The machine has
40 physical cores (2x Intel Xeon Gold 6242R) with Hyper-Threading, giving 80 logical
threads total.

### GPU Results: Transformers vs vLLM

| Metric            | Transformers       | vLLM               | Winner              |
|-------------------|--------------------|---------------------|---------------------|
| Total time        | 3.11s              | **0.92s**           | vLLM (3.37x faster) |
| Time per prompt   | 6.22ms             | **1.85ms**          | vLLM                |
| Throughput        | 160.53 prompts/sec | **541.32 prompts/sec** | **vLLM**         |
| Peak GPU memory   | 0.47 GB            | **0.01 GB**         | vLLM (98% less)     |

### Cross-Platform Summary

| Platform             | Time    | Throughput          |
|----------------------|---------|---------------------|
| CPU (Transformers)   | 31.98s  | 15.63 prompts/sec   |
| GPU (Transformers)   | 3.11s   | 160.53 prompts/sec  |
| **GPU (vLLM)**       | **0.92s** | **541.32 prompts/sec** |

- GPU Transformers is **10.3x faster** than CPU Transformers
- **vLLM on GPU is 34.7x faster than CPU Transformers**
- vLLM on GPU is **3.37x faster** than Transformers on GPU

---

## Comparison with Original Test (Batch Size 5)

| Scenario              | Batch 5 (original) | Batch 500 (new)     | Scaling Factor |
|-----------------------|---------------------|---------------------|----------------|
| CPU Transformers      | 0.87s               | 31.98s              | ~37x time for 100x batch |
| GPU Transformers      | 0.27s               | 3.11s               | ~12x time for 100x batch |
| GPU vLLM              | 0.18s               | 0.92s               | ~5x time for 100x batch  |

**Key Insight**: vLLM scales much better with batch size. At batch 5, vLLM was only
1.5x faster than Transformers on GPU. At batch 500, vLLM is **3.37x faster**. This
confirms that vLLM's PagedAttention and continuous batching optimizations provide
increasing benefits as batch size grows.

---

## Troubleshooting

### vLLM fails with FlashInfer compilation error

This happens because FlashAttention-2 requires compute capability >= 8.0 (A100, H100),
but TITAN RTX is compute 7.5. The fix is to use TRITON_ATTN backend:

```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN python3 test_large_batch.py --batch-size 500
```

If FlashInfer cached data is corrupted, clear the cache:

```bash
rm -rf ~/.cache/flashinfer/
```

### CUDA out of memory

If previous test runs left stale GPU processes:

```bash
# Check what's using the GPU
nvidia-smi

# Kill stale Python processes (replace PIDs)
kill <PID1> <PID2> ...

# Then re-run the test
```

To reduce batch size:

```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN python3 test_large_batch.py --batch-size 250
```

### python: command not found

The system uses `python3`, not `python`. The test script and convenience script are
already configured to use `python3`.

---

## Files

| File                      | Description                                           |
|---------------------------|-------------------------------------------------------|
| `test_large_batch.py`     | Main test script (CPU monitoring + GPU comparison)    |
| `run_large_batch_test.sh` | Convenience shell script to run the full test         |
| `LARGE_BATCH_TEST_GUIDE.md` | This guide                                         |

---

**Created**: February 17, 2026
**Last Updated**: February 17, 2026
