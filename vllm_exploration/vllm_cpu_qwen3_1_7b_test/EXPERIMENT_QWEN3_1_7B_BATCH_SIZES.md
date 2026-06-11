# Experiment: Qwen3-1.7B on CPU — batch size sweep (with / without vLLM)

This document matches the methodology used in earlier runs (see [QWEN3_1_7B_CPU_EVALUATION.md](QWEN3_1_7B_CPU_EVALUATION.md) and [RUN_SUMMARY_TRANSFORMERS_ONLY_IMU_THOR.md](RUN_SUMMARY_TRANSFORMERS_ONLY_IMU_THOR.md)), extended to four batch sizes.

## Experiments

| ID | Engine | Batch sizes | Metric |
|----|--------|-------------|--------|
| A | **without vLLM** (Transformers, CPU, bf16) | 1, 500, 1000, 2000 | Wall time for full workload, ms/prompt, prompts/s |
| B | **with vLLM** (CPU) | 1, 500, 1000, 2000 | Same + vLLM init time per subprocess run |

Each batch size is a separate **concurrent prompt count**: `batch_size` prompts, each with the same generation budget (`max_new_tokens`, default **30**).

- **Transformers:** The model is loaded **once**; each batch size is timed in sequence (fair throughput; avoids four cold loads).
- **vLLM:** Each batch size runs in a **fresh subprocess** (separate `LLM()` init), matching the previous file-based worker pattern in `test_qwen3_vllm_cpu.py`.

## Prompts

Same template as before:

`Question {i}: Explain CPU inference trade-offs in one paragraph.`

## Commands

Full sweep (default):

```bash
cd /path/to/vllm_exploration/vllm_cpu_qwen3_1_7b_test
./run_qwen3_cpu_test.sh
```

Or explicitly:

```bash
python3 test_qwen3_cpu_compare.py \
  --batch-sizes "1,500,1000,2000" \
  --max-new-tokens 30 \
  --vllm-max-num-seqs 2048
```

Transformers only (no vLLM):

```bash
python3 test_qwen3_cpu_compare.py --transformers-only \
  --batch-sizes "1,500,1000,2000" \
  --max-new-tokens 30
```

Legacy two-size run (same as old scripts):

```bash
python3 test_qwen3_cpu_compare.py \
  --single-batch-size 1 \
  --large-batch-size 500 \
  --max-new-tokens 30
```

## Outputs

- `qwen3_1_7b_cpu_results.json` — both engines + ratios per batch size  
- `qwen3_1_7b_cpu_results_transformers_only.json` — with `--transformers-only`

JSON shape: `by_batch_size["1" | "500" | ...]` → results under keys `qwen3-1.7B without vLLM` and `qwen3-1.7B with vLLM`.

## vLLM `max_num_seqs`

Use **`--vllm-max-num-seqs` ≥ largest batch size** (default **2048**) so vLLM can schedule full concurrency. Smaller values (e.g. 32) serialize large batches and inflate wall time.

## Environment

- `VLLM_CPU_PYTHON` — Python that has CPU vLLM installed (see [README.md](README.md)).
