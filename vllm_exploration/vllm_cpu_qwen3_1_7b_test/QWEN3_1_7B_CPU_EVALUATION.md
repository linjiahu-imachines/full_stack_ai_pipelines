# Qwen3-1.7B CPU-Only Evaluation

Dedicated CPU-only comparison for:

- `Qwen/Qwen3-1.7B`

Comparison targets:

1. `qwen3-1.7B without vLLM`
2. `qwen3-1.7B with vLLM`

For the **1 / 500 / 1000 / 2000** batch sweep description, see [EXPERIMENT_QWEN3_1_7B_BATCH_SIZES.md](EXPERIMENT_QWEN3_1_7B_BATCH_SIZES.md).

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

Custom run (four batch sizes):

```bash
"/home/azureuser/projects/vllm_exploration/vllm_cpu_venv/bin/python3" \
"/home/azureuser/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test/test_qwen3_cpu_compare.py" \
--batch-sizes "1,500,1000,2000" \
--max-new-tokens 30 \
--vllm-max-num-seqs 2048
```

Legacy two-size run:

```bash
"/home/azureuser/projects/vllm_exploration/vllm_cpu_venv/bin/python3" \
"/home/azureuser/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test/test_qwen3_cpu_compare.py" \
--single-batch-size 1 \
--large-batch-size 500 \
--max-new-tokens 30 \
--vllm-max-num-seqs 2048
```

Output JSON:

- `vllm_cpu_qwen3_1_7b_test/qwen3_1_7b_cpu_results.json`

---

## Latest Results (Single + Batch-500)

Run metadata:

- Timestamp (UTC): `2026-03-11T04:10:15.786589+00:00`
- Single query batch size: `1`
- Large batch size: `500`
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
- Available memory at test start: `29.57 GB`

---

### Single Query Performance

| Engine | Total Time | Per Prompt | Throughput |
|---|---:|---:|---:|
| qwen3-1.7B without vLLM | 2.72s | 2722.55ms | 0.37 prompts/sec |
| qwen3-1.7B with vLLM | 2.14s | 2143.78ms | 0.47 prompts/sec |

### Batch-500 Performance

| Engine | Total Time | Per Prompt | Throughput |
|---|---:|---:|---:|
| qwen3-1.7B without vLLM | 77.92s | 155.85ms | 6.42 prompts/sec |
| qwen3-1.7B with vLLM | 54.95s | 109.89ms | 9.10 prompts/sec |

### Initialization and Runtime Notes

- vLLM CPU init time: `8.82s` (single) / `8.90s` (batch-500)
- `qwen3-1.7B without vLLM` used deterministic micro-batching (`micro_batch_size=25`) in batch-500 mode to avoid OOM while preserving total workload size.
- `qwen3-1.7B with vLLM` used `max_num_seqs=32` in batch-500 mode.

---

## Analysis

### 1) Single query comparison

For single query (`batch_size=1`, `max_new_tokens=30`):

- Time ratio (`with / without vLLM`): `0.787`
- Throughput ratio (`with / without vLLM`): `1.270x`

Interpretation: `qwen3-1.7B with vLLM` is faster for this single-query run.

### 2) Batch-500 comparison

For 500-query batch (`batch_size=500`, `max_new_tokens=30`):

- Time ratio (`with / without vLLM`): `0.705`
- Throughput ratio (`with / without vLLM`): `1.417x`

Interpretation: `qwen3-1.7B with vLLM` shows a larger advantage in high-volume throughput.

### 3) Cold-start tradeoff

vLLM still has a non-trivial initialization cost (`8.85s` in this run).  
For one-off invocations, startup cost matters. For repeated large-batch processing, throughput gains can outweigh startup.

### 4) Practical recommendation

- **Single query**: `qwen3-1.7B with vLLM` was faster in this measurement.
- **Batch 500**: `qwen3-1.7B with vLLM` was clearly faster and more throughput-efficient.
- **Deployment**: if running as a long-lived CPU service, vLLM becomes more attractive since init cost is amortized.

---

## Related References

- vLLM CPU hardware + recommended models: [CPU - Intel Xeon - vLLM](https://docs.vllm.ai/en/stable/models/hardware_supported_models/cpu/#validated-hardware)
- vLLM CPU install/build guide: [vLLM CPU Installation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)
