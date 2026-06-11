# Qwen3-1.7B CPU inference — run summary (Transformers only)

**Machine:** `imu-thor` (NVIDIA Jetson AGX Thor)  
**Date:** 2026-04-03 (approximate, from session)  
**Mode:** Hugging Face **Transformers** on **CPU** only (`--transformers-only`; vLLM not used)

---

## Command

```bash
cd /home/linhu/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test
VLLM_CPU_PYTHON=/home/linhu/projects/vllm_exploration/vllm_gpu_test/venv/bin/python3 \
  /home/linhu/projects/vllm_exploration/vllm_gpu_test/venv/bin/python3 test_qwen3_cpu_compare.py \
  --transformers-only \
  --single-batch-size 1 \
  --large-batch-size 500 \
  --max-new-tokens 30
```

**Model:** `Qwen/Qwen3-1.7B`  
**Python env:** `vllm_gpu_test/venv` (torch + transformers)

---

## Hardware context (from run header)

| Resource | Value |
|----------|--------|
| CPU | 14 physical / 14 logical cores |
| RAM | ~122.8 GB total, ~118.3 GB available at start |
| Cache (reported) | L1d/L1i 896 KiB (14×), L2 14 MiB (14×), L3 not reported |

---

## Results

| Scenario | Wall time | Per prompt |
|----------|-----------|------------|
| **Single query** (batch size 1) | **12.23 s** | ~**12,231 ms** |
| **Batch workload** (500 prompts) | **532.06 s** (~**8.9 min**) | ~**1,064 ms** |

**Interpretation:** The **500-prompt** phase is much **lower latency per prompt** than the isolated single-query case because the benchmark **micro-batches** prompts (chunks of 25) and amortizes fixed costs across many generations. The single-query line includes a full **model load + first forward** path typical of a “one-off” request.

---

## Artifacts

- **Structured results:** `qwen3_1_7b_cpu_results_transformers_only.json` (same directory as this file)
- **Script:** `test_qwen3_cpu_compare.py` (supports `--transformers-only`)

---

## Notes

- Transformers emitted a **non-fatal** message about unused generation flags (`temperature`, `top_p`, `top_k`) when sampling is off; safe to ignore for this greedy `generate` setup.
- This summary documents **CPU Transformers** behavior only; it is **not** comparable to a GPU or vLLM run without repeating those experiments on hardware where they are supported.

---

## Related docs

- [README.md](README.md) — how to run full CPU comparison (with vLLM) or `--transformers-only`
- [QWEN3_1_7B_CPU_EVALUATION.md](QWEN3_1_7B_CPU_EVALUATION.md) — broader evaluation notes
- [../docs/MACHINE_IMU_THOR_JETSON_AGX_THOR.md](../docs/MACHINE_IMU_THOR_JETSON_AGX_THOR.md) — machine profile
