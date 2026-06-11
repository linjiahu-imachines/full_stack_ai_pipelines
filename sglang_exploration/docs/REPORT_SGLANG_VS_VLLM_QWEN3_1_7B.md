# SGLang vs vLLM Performance Comparison (Qwen3-1.7B)

## Scope

This document compares saved results from:

- **SGLang CPU**
- **SGLang GPU**
- **vLLM CPU**

All runs use the **same query set**:

- Prompt file: `/home/linhu/projects/sglang_exploration/sglang_test/data/queries_512.jsonl`
- Prompt count: `512`
- Model: `Qwen/Qwen3-1.7B`
- Batch sizes: `8, 16, 32, 64, 128, 256, 512`
- Max new tokens: `32`

## Data Sources

- SGLang results: `sglang_exploration/sglang_test/results/{cpu_bsN.json,gpu_bsN.json}`
- vLLM results: `vllm_exploration/vllm_cpu_qwen3_1_7b_test/results_same_queries/vllm_cpu_bsN.json`

## Raw Results

| Batch | SGLang CPU `time_s` | SGLang CPU `thr` | vLLM CPU `time_s` | vLLM CPU `thr` | SGLang GPU `time_s` | SGLang GPU `thr` |
|---:|---:|---:|---:|---:|---:|---:|
| 8   | 1407.680 | 0.364 | 519.92 | 0.98 | 142.423 | 3.595 |
| 16  | 1045.742 | 0.490 | 423.09 | 1.21 | 109.172 | 4.690 |
| 32  | 905.646  | 0.565 | 281.35 | 1.82 | 94.346  | 5.427 |
| 64  | 773.840  | 0.662 | 205.42 | 2.49 | 86.471  | 5.921 |
| 128 | 682.133  | 0.751 | 165.28 | 3.10 | 81.486  | 6.283 |
| 256 | 630.650  | 0.812 | 143.19 | 3.58 | 79.065  | 6.476 |
| 512 | 598.441  | 0.856 | 117.63 | 4.35 | 76.376  | 6.704 |

`thr` = `throughput_prompts_per_s`

## Comparison A: vLLM CPU vs SGLang CPU

| Batch | vLLM speedup on `time_s` (`sglang_cpu / vllm_cpu`) | vLLM speedup on throughput (`vllm_cpu / sglang_cpu`) |
|---:|---:|---:|
| 8   | 2.71x | 2.69x |
| 16  | 2.47x | 2.47x |
| 32  | 3.22x | 3.22x |
| 64  | 3.77x | 3.76x |
| 128 | 4.13x | 4.13x |
| 256 | 4.40x | 4.41x |
| 512 | 5.09x | 5.08x |

### Takeaway

- In this environment and workload, **vLLM CPU is consistently faster than SGLang CPU**.
- Advantage grows with larger batches (about **2.5x** at small-mid batches to about **5x** at batch 512).

## Comparison B: SGLang GPU vs vLLM CPU

| Batch | SGLang GPU speedup on `time_s` (`vllm_cpu / sglang_gpu`) | SGLang GPU speedup on throughput (`sglang_gpu / vllm_cpu`) |
|---:|---:|---:|
| 8   | 3.65x | 3.67x |
| 16  | 3.88x | 3.88x |
| 32  | 2.98x | 2.98x |
| 64  | 2.38x | 2.38x |
| 128 | 2.03x | 2.03x |
| 256 | 1.81x | 1.81x |
| 512 | 1.54x | 1.54x |

### Takeaway

- **SGLang GPU remains faster than vLLM CPU** across all tested batch sizes.
- The relative advantage shrinks at very large batches because vLLM CPU scales strongly with batch size in this setup.

## Overall Summary

1. **Fastest absolute path in these runs:** SGLang on GPU.
2. **Best CPU-only path in these runs:** vLLM CPU (better than SGLang CPU at every batch size tested).
3. If your deployment target is CPU-only, current measurements favor vLLM; if GPU is available, SGLang GPU still leads on this workload.

## Notes

- These comparisons are empirical for the current Jetson Thor software stack and current runtime options.
- For publication-grade conclusions, rerun each point multiple times and report mean/stddev.
