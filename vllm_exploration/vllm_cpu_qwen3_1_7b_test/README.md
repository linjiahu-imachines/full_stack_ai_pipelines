# Qwen3 1.7B CPU-Only Evaluation

This folder contains a dedicated CPU-only benchmark for:

- `Qwen/Qwen3-1.7B`

It compares two named experiments:

1. `qwen3-1.7B without vLLM`
2. `qwen3-1.7B with vLLM`

**Default batch sweep:** `1`, `500`, `1000`, `2000` prompts per timed run (see [EXPERIMENT_QWEN3_1_7B_BATCH_SIZES.md](EXPERIMENT_QWEN3_1_7B_BATCH_SIZES.md)).

## Quick Run

From project root:

```bash
cd /path/to/vllm_exploration
./vllm_cpu_qwen3_1_7b_test/run_qwen3_cpu_test.sh
```

## Common Options

```bash
cd /path/to/vllm_exploration/vllm_cpu_qwen3_1_7b_test
python3 test_qwen3_cpu_compare.py \
  --batch-sizes "1,500,1000,2000" \
  --max-new-tokens 30 \
  --vllm-max-num-seqs 2048
```

Legacy (only two sizes, same as older machines):

```bash
python3 test_qwen3_cpu_compare.py \
  --single-batch-size 1 \
  --large-batch-size 500 \
  --max-new-tokens 30 \
  --vllm-max-num-seqs 2048
```

**Hugging Face only (no vLLM)** — when vLLM is unavailable or you only need to verify Transformers CPU inference:

```bash
python3 test_qwen3_cpu_compare.py --transformers-only \
  --single-batch-size 1 --large-batch-size 500 --max-new-tokens 30
```

Results are written to `qwen3_1_7b_cpu_results_transformers_only.json`.

## Output

Results are written to:

- `qwen3_1_7b_cpu_results.json` — nested under `by_batch_size` for each batch size
- `qwen3_1_7b_cpu_results_transformers_only.json` when using `--transformers-only`

## Same 512-query input as SGLang

To run vLLM CPU with the exact same prompt file used in `sglang_exploration`:

```bash
cd /home/linhu/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test
./run_vllm_same_queries_sweep.sh
```

Defaults:

- prompt file: `/home/linhu/projects/sglang_exploration/sglang_test/data/queries_512.jsonl`
- model: `Qwen/Qwen3-1.7B`
- max new tokens: `32`
- batch sizes: `8,16,32,64,128,256,512`
- outputs: `results_same_queries/vllm_cpu_bs<N>.json`

Override example:

```bash
BATCH_SIZES=16,32,64 MAX_NEW_TOKENS=32 ./run_vllm_same_queries_sweep.sh
```

Example write-up for one completed **Transformers-only** run on **imu-thor**: [RUN_SUMMARY_TRANSFORMERS_ONLY_IMU_THOR.md](RUN_SUMMARY_TRANSFORMERS_ONLY_IMU_THOR.md).

## Notes

- Use file-based execution for vLLM CPU tests (not heredoc stdin execution).
- `VLLM_CPU_SGL_KERNEL=1` is optional tuning, not required.
- This test is CPU-only by design.
- For the 500-query path, the non-vLLM side uses deterministic micro-batching to avoid OOM while preserving a 500-query total workload.
