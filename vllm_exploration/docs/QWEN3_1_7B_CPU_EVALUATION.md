# Qwen3-1.7B CPU-Only Evaluation

Dedicated CPU-only comparison for:

- `Qwen/Qwen3-1.7B`

Comparison targets:

1. `qwen3-1.7B without vLLM`
2. `qwen3-1.7B with vLLM`

---

## Test Setup

- Machine: Azure VM (CPU-only)
- CPU: 4 physical cores / 8 logical threads
- GPU: none
- Python env: `vllm_cpu_venv`
- Test harness:
  - `vllm_cpu_qwen3_1_7b_test/test_qwen3_cpu_compare.py`
  - `vllm_cpu_qwen3_1_7b_test/test_qwen3_vllm_cpu.py`

---

## How to Run

From project root:

```bash
cd /home/azureuser/projects/vllm_exploration
./vllm_cpu_qwen3_1_7b_test/run_qwen3_cpu_test.sh
```

Custom run:

```bash
"/home/azureuser/projects/vllm_exploration/vllm_cpu_venv/bin/python3" \
"/home/azureuser/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test/test_qwen3_cpu_compare.py" \
--single-batch-size 1 \
--large-batch-size 500 \
--max-new-tokens 30
```

Output JSON:

- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results.json`
- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results_batch500.json`
- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results_batch1000.json`
- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results_batch2000.json`

---

## Latest Results (Single + Batch-500)

Run metadata:

- Baseline timestamp (UTC): `2026-03-17T16:37:34.781617+00:00`
- Single query count: `1`
- Large input query count: `500`
- Max new tokens: `30`

## System Configuration Snapshot (This Test Run)

### CPU Configuration

- Vendor: `GenuineIntel`
- CPU model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Architecture: `x86_64`
- Sockets: `1`
- Cores per socket: `4`
- Threads per core: `2`
- Total logical CPUs: `8`
- Total physical cores: `4`
- Frequency range: `800 MHz` to `2300 MHz`

### Cache Configuration

- L1d cache: `192 KiB (4 instances)`
- L1i cache: `128 KiB (4 instances)`
- L2 cache: `8 MiB (4 instances)`
- L3 cache: `260 MiB (1 instance)`

### Memory Configuration

- Total memory: `31.34 GB`
- Available memory at test start: `29.74 GB`

---

## AMX / AI Feature and Compilation Status

### Hardware feature exposure during this experiment

The CPU flags on this machine include:

- `amx_tile`, `amx_int8`, `amx_bf16`
- `avx512f`, `avx512_bf16`, `avx512_vnni`

This confirms AMX and key AI-related x86 instruction features are exposed by CPU/OS.

### Runtime backend status observed

- PyTorch backend:
  - `torch==2.10.0+cpu`
  - `torch.backends.mkldnn.is_available() == True`
  - `torch.backends.openmp.is_available() == True`
  - `torch.backends.cpu.get_cpu_capability() == AVX512`

### Compilation/build status for this experiment

- vLLM in this run was installed from a **prebuilt CPU wheel**:
  - `vllm-0.17.0+cpu-cp38-abi3-manylinux_2_35_x86_64.whl`
- This experiment did **not** perform a local source rebuild with explicit compile flags (for example, forcing AMX-specific build args).
- Therefore:
  - We can confirm the hardware supports AMX/AVX512-BF16.
  - We can confirm the runtime stack is AVX512-capable.
  - We cannot claim every kernel path in this run was AMX-specialized unless explicitly enabled and validated by kernel-level tracing/logging.

### Important note on `VLLM_CPU_SGL_KERNEL`

- `VLLM_CPU_SGL_KERNEL=1` is optional tuning for small-batch optimized kernels.
- It is **not** required to "enable AMX"; AMX exposure is hardware/OS-level.
- In this benchmark configuration, SGL was not explicitly enabled.

---

### Single Query Performance

| Engine | Total Time | Per Prompt | Throughput |
|---|---:|---:|---:|
| qwen3-1.7B without vLLM | 2.68s | 2675.90ms | 0.37 prompts/sec |
| qwen3-1.7B with vLLM | 2.17s | 2168.45ms | 0.46 prompts/sec |

### Large-Input Performance (500 baseline + 1000/2000 follow-up runs)

| Input Query Count | without vLLM Time | with vLLM Time | without vLLM Throughput | with vLLM Throughput | Speedup (without / with) |
|---:|---:|---:|---:|---:|---:|
| 500 | 78.14s | 54.80s | 6.40 q/s | 9.12 q/s | 1.43x |
| 1000 | 155.28s | 108.83s | 6.44 q/s | 9.19 q/s | 1.43x |
| 2000 | 313.08s | 215.92s | 6.39 q/s | 9.26 q/s | 1.45x |

### Speedup Summary (with vLLM vs without vLLM)

- Single query speedup: `2.68 / 2.17 = 1.24x` (time-based, baseline run)
- Batch-500 speedup: `78.14 / 54.80 = 1.43x`
- Batch-1000 speedup: `155.28 / 108.83 = 1.43x`
- Batch-2000 speedup: `313.08 / 215.92 = 1.45x`

## `max_num_seqs` Sweep Summary (Input Query Count 500)

Additional sensitivity experiment was run at input query count `500` with different vLLM concurrency caps.

| `max_num_seqs` | without vLLM (time / throughput) | with vLLM (time / throughput) | Speedup (without / with) |
|---:|---|---|---:|
| 32 (baseline) | 78.14s / 6.40 q/s | 54.80s / 9.12 q/s | 1.43x |
| 64 | 77.58s / 6.44 q/s | 36.56s / 13.67 q/s | 2.12x |
| 128 | 78.04s / 6.41 q/s | 28.12s / 17.78 q/s | 2.77x |
| 256 | 77.64s / 6.44 q/s | 25.40s / 19.69 q/s | 3.06x |

### Key Takeaways

- `qwen3-1.7B without vLLM` stays nearly constant across these runs.
- `qwen3-1.7B with vLLM` improves substantially as `max_num_seqs` increases.
- In this report, **Input Query Count** is the total number of requests in the workload.  
  It is different from vLLM's internal processing concurrency (`max_num_seqs`), which controls how many sequences are processed per scheduler iteration.
- On this machine/workload, `max_num_seqs=32` underutilized vLLM potential for batch-500.
- `max_num_seqs=256` gave the best batch-500 throughput in this sweep (`19.69 q/s`), with diminishing but still positive gains vs `128`.
- For high-volume CPU batching, tuning `max_num_seqs` is a major performance lever.

Reference result files:

- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results_batch500.json`
- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results_batch500_seq64.json`
- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results_batch500_seq128.json`
- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results_batch500_seq256.json`

## Additional Experiments: `max_num_seqs=256` at Batch 1000 and 2000

To extend the sweep, we ran two more experiments with `max_num_seqs=256`.

| Batch Size | without vLLM Time | with vLLM Time | without vLLM Throughput | with vLLM Throughput | Speedup (without / with) |
|---:|---:|---:|---:|---:|---:|
| 1000 | 155.99s | 50.73s | 6.41 q/s | 19.71 q/s | 3.08x |
| 2000 | 312.39s | 104.36s | 6.40 q/s | 19.17 q/s | 2.99x |

### Notes

- At `max_num_seqs=256`, throughput for `qwen3-1.7B with vLLM` stays near ~19 q/s for both 1000 and 2000 query runs.
- Compared with earlier `max_num_seqs=32` runs, this is a substantial gain for high-volume batching.

Reference result files:

- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results_batch1000_seq256.json`
- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results_batch2000_seq256.json`

### Initialization and Runtime Notes

- vLLM CPU init time: `8.82s` (single) / `8.90s` (batch-500)
- `qwen3-1.7B without vLLM` used deterministic micro-batching (`micro_batch_size=25`) in batch-500 mode to avoid OOM while preserving total workload size.
- Baseline `qwen3-1.7B with vLLM` run used `max_num_seqs=32` for input query count 500.

---

## Analysis

### 1) Single query comparison

For single query (input query count `1`, `max_new_tokens=30`):

- Time ratio (`with / without vLLM`): `0.81`
- Throughput ratio (`with / without vLLM`): `1.243x`

Interpretation: `qwen3-1.7B with vLLM` is faster for this single-query run.

### 2) Batch-500 comparison

For input query count `500` (`max_new_tokens=30`):

- Time ratio (`with / without vLLM`): `0.701`
- Throughput ratio (`with / without vLLM`): `1.425x`

Interpretation: `qwen3-1.7B with vLLM` shows a larger advantage in high-volume throughput.

For larger batches, the same trend continues:

- 1000-query: about `1.43x` speedup with vLLM
- 2000-query: about `1.45x` speedup with vLLM

### 3) Cold-start tradeoff

vLLM still has a non-trivial initialization cost (`8.88s` single / `8.95s` batch-500 in baseline run).  
For one-off invocations, startup cost matters. For repeated large-batch processing, throughput gains can outweigh startup.

### 4) Practical recommendation

- **Single query**: `qwen3-1.7B with vLLM` was faster in this measurement.
- **Batch 500**: `qwen3-1.7B with vLLM` was clearly faster and more throughput-efficient.
- **Deployment**: if running as a long-lived CPU service, vLLM becomes more attractive since init cost is amortized.

---

## Related References

- vLLM CPU hardware + recommended models: [CPU - Intel Xeon - vLLM](https://docs.vllm.ai/en/stable/models/hardware_supported_models/cpu/#validated-hardware)
- vLLM CPU install/build guide: [vLLM CPU Installation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)
