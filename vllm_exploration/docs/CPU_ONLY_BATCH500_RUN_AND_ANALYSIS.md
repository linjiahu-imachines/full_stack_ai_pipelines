# CPU-Only Batch-500 Run Guide and Analysis

This document explains how to run the CPU-only batch-500 comparison and summarizes the latest measured results on this machine.

---

## What This Test Does

The comparison runs 2 CPU scenarios on `facebook/opt-125m` with `batch_size=500`:

1. **Transformers on CPU** (without vLLM)
2. **vLLM on CPU** (using CPU-capable vLLM environment)

GPU scenarios are present in the runner, but on CPU-only machines they are skipped automatically.

---

## Environment Used

- Machine: Azure VM
- CPU: 4 physical cores / 8 logical threads
- RAM: 31.3 GB
- GPU: Not available (`CUDA = False`)
- Python used: `vllm_cpu_venv/bin/python3`
- Model: `facebook/opt-125m`

---

## Validated CPU Prerequisites and Notes

Based on the machine check and a review of current vLLM docs:

- CPU instruction support on this host is sufficient for vLLM CPU:
  - `amx_tile`, `amx_int8`, `amx_bf16`
  - `avx512f`, `avx512_bf16`, `avx512_vnni`
- Compiler recommendation is satisfied:
  - `gcc 13.3.0` (recommended: gcc/g++ >= 12.3)
- `VLLM_TARGET_DEVICE=cpu` is a valid install/build selector.
- `setup.py install` is considered a legacy install path; prefer modern pip/uv install commands.
- `VLLM_CPU_SGL_KERNEL=1` is optional tuning for specific small-batch BF16 kernel paths, not a required AMX switch.

### Recommended install/build flow (from source)

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -v -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu pip install . --no-build-isolation
```

### Optional runtime tuning

```bash
# Optional only (experimental, x86 small-batch optimization path)
export VLLM_CPU_SGL_KERNEL=1
```

---

## How To Run

From project root:

```bash
cd /home/azureuser/projects/vllm_exploration
"/home/azureuser/projects/vllm_exploration/vllm_cpu_venv/bin/python3" "vllm_gpu_test/test_batch500_complete.py"
```

Alternative (from `vllm_gpu_test/`):

```bash
cd /home/azureuser/projects/vllm_exploration/vllm_gpu_test
"/home/azureuser/projects/vllm_exploration/vllm_cpu_venv/bin/python3" "test_batch500_complete.py"
```

Results are saved to:

- `vllm_gpu_test/batch500_results.json`

---

## Latest Run Results (2026-03-10 22:05:03)

### Raw Performance

| Scenario | Total Time | Per Prompt | Throughput |
|---|---:|---:|---:|
| Transformers CPU | 29.14s | 58.29ms | 17.16 prompts/sec |
| vLLM CPU | 64.23s | 128.46ms | 7.78 prompts/sec |

### CPU Utilization Snapshot

| Scenario | Active Threads (>5%) | Avg CPU Util | Max Proc Threads | Avg Memory |
|---|---:|---:|---:|---:|
| Transformers CPU | 8/8 | 32.6% | 13 | 2387 MB |
| vLLM CPU | 4/8 | 40.5% | 22 | 567 MB |

### Relative Comparison

- **Latency ratio**: `64.23 / 29.14 = 2.20x`
  - vLLM CPU took about **2.20x longer** than Transformers CPU.
- **Throughput ratio**: `17.16 / 7.78 = 2.21x`
  - Transformers CPU delivered about **2.21x higher throughput**.

---

## Analysis

### 1) Transformers is faster for this CPU workload

On this VM and model size, Transformers completed batch-500 significantly faster than vLLM CPU.

### 2) Threading behavior differs

- Transformers spread work across all 8 logical threads.
- vLLM CPU concentrated work on 4 threads (one per physical core), consistent with vLLM CPU auto thread-binding.

This improves predictable core affinity, but here it reduced aggregate throughput versus the Transformers baseline.

### 3) Memory tradeoff

vLLM CPU used substantially less process memory in this run, but memory savings did not translate into better speed for this test case.

### 4) Why this can happen

vLLM brings scheduling and serving-oriented infrastructure that is usually most beneficial with GPU-backed deployments and high concurrency. For this specific CPU-only setup and small model, the additional runtime overhead can outweigh gains.

---

## Practical Recommendation (This Machine)

- For CPU-only batch inference with this model/profile, use **Transformers CPU** by default.
- Keep vLLM CPU as an option for further experimentation (different models, thread binding, larger sequence lengths, or different concurrency patterns).

---

## Official vLLM CPU Reference (Intel Xeon)

The official vLLM CPU hardware/model page confirms:

- **Validated hardware families** include Intel Xeon 5 and Xeon 6 generations.
- vLLM publishes a **recommended model list** for CPU (text-only and multimodal), including:
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `meta-llama/Llama-3.2-3B-Instruct`
  - `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`
  - `zai-org/glm-4-9b-hf`
  - `google/gemma-7b`
  - multimodal examples such as `Qwen/Qwen2.5-VL-7B-Instruct` and `openai/whisper-large-v3`

For current compatibility status and updates, use:

- [CPU - Intel Xeon - vLLM](https://docs.vllm.ai/en/stable/models/hardware_supported_models/cpu/)
- [vLLM CPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)

---

## Related Files

- Runner: `vllm_gpu_test/test_batch500_complete.py`
- vLLM CPU subprocess test: `vllm_cpu_test/test2_vllm_cpu.py`
- Structured output: `vllm_gpu_test/batch500_results.json`
