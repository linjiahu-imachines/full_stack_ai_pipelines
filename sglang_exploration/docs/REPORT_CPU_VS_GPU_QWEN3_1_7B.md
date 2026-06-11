# CPU vs GPU Performance Analysis (SGLang)

## Scope

This report includes two experiment groups:

1. **CPU vs GPU baseline** (single prompt, batch size 1)
2. **GPU-only scaling** with real prompt set (`512` prompts) across multiple batch sizes

Common settings:

- Model: `Qwen/Qwen3-1.7B`
- Script: `sglang_test/run_offline_batch_experiment.py`
- Max new tokens: `32`
- GPU runtime profile: `attention_backend=torch_native`, `disable_cuda_graph=True`

All numbers come from successful `__RESULT_JSON__` outputs in terminal logs.

## A) CPU vs GPU Baseline (single prompt)

### Commands

```bash
source /home/linhu/projects/sglang_exploration/activate_sglang_env.sh
python /home/linhu/projects/sglang_exploration/sglang_test/run_offline_batch_experiment.py \
  --device cuda --model Qwen/Qwen3-1.7B --synthetic --num-prompts 1 --batch-size 1 --max-new-tokens 32

python /home/linhu/projects/sglang_exploration/sglang_test/run_offline_batch_experiment.py \
  --device cpu --model Qwen/Qwen3-1.7B --synthetic --num-prompts 1 --batch-size 1 --max-new-tokens 32
```

### Raw Results

| Mode | `init_time_s` | `time_s` | `per_prompt_ms` | `throughput_prompts_per_s` |
|---|---:|---:|---:|---:|
| GPU (`device=cuda`) | 14.282 | 2.583 | 2582.722 | 0.387 |
| CPU (`device=cpu`) | 20.536 | 15.075 | 15074.896 | 0.066 |

### Relative Comparison

- **Inference latency (`time_s`)**: GPU is about **5.84x faster** than CPU (`15.075 / 2.583`).
- **Throughput**: GPU is about **5.86x higher** than CPU (`0.387 / 0.066`).
- **Engine startup (`init_time_s`)**: CPU startup is ~`1.44x` longer than GPU (`20.536 / 14.282`).

## B) GPU-Only Scaling (512 real prompts)

Workload definition:

- Prompt file: `sglang_test/data/queries_512.jsonl`
- Prompt count: `512`
- Batch size = prompts per `generate()` call

### Command Pattern

```bash
source /home/linhu/projects/sglang_exploration/activate_sglang_env.sh
OUTPUT=/home/linhu/projects/sglang_exploration/sglang_test/results/gpu_bs<N>.json \
  /home/linhu/projects/sglang_exploration/sglang_test/run_experiment_gpu_batch.sh <N>
```

### Raw Results (GPU only)

| Batch size | `init_time_s` | `time_s` | `per_prompt_ms` | `throughput_prompts_per_s` |
|---:|---:|---:|---:|---:|
| 8   | 16.168 | 142.423 | 278.170 | 3.595 |
| 16  | 13.621 | 109.172 | 213.226 | 4.690 |
| 32  | 28.905 | 94.346  | 184.270 | 5.427 |
| 64  | 13.936 | 86.471  | 168.888 | 5.921 |
| 128 | 13.856 | 81.486  | 159.153 | 6.283 |
| 256 | 13.704 | 79.065  | 154.424 | 6.476 |
| 512 | 13.731 | 76.376  | 149.173 | 6.704 |

### Scaling Analysis (GPU only)

- **Throughput improves monotonically** from `3.595` (BS=8) to `6.704` (BS=512), about **1.86x** total gain.
- **Per-prompt latency decreases** from `278.170 ms` (BS=8) to `149.173 ms` (BS=512), about **46.4% lower**.
- **Largest gains are at smaller-to-mid batch sizes** (8 -> 64), then returns gradually diminish beyond 128.
- **Best observed throughput in this sweep** is at **BS=512**.
- `init_time_s` is mostly stable around ~14-16s, with one outlier at BS=32 (`28.905s`) likely due to transient warmup/system variability.

## C) CPU-Only Scaling (512 real prompts)

### Command Pattern

```bash
source /home/linhu/projects/sglang_exploration/activate_sglang_env.sh
python /home/linhu/projects/sglang_exploration/sglang_test/run_offline_batch_experiment.py \
  --device cpu --model Qwen/Qwen3-1.7B --batch-size <N> --max-new-tokens 32 \
  --output /home/linhu/projects/sglang_exploration/sglang_test/results/cpu_bs<N>.json
```

### Raw Results (CPU only)

| Batch size | `init_time_s` | `time_s` | `per_prompt_ms` | `throughput_prompts_per_s` |
|---:|---:|---:|---:|---:|
| 8   | 20.859 | 1407.680 | 2749.375 | 0.364 |
| 16  | 20.644 | 1045.742 | 2042.466 | 0.490 |
| 32  | 19.578 | 905.646  | 1768.841 | 0.565 |
| 64  | 18.158 | 773.840  | 1511.406 | 0.662 |
| 128 | 20.591 | 682.133  | 1332.291 | 0.751 |
| 256 | 18.853 | 630.650  | 1231.739 | 0.812 |
| 512 | 20.603 | 598.441  | 1168.831 | 0.856 |

### Scaling Analysis (CPU only)

- **Throughput improves monotonically** from `0.364` (BS=8) to `0.856` (BS=512), about **2.35x** total gain.
- **Per-prompt latency decreases** from `2749.375 ms` (BS=8) to `1168.831 ms` (BS=512), about **57.5% lower**.
- CPU also benefits from larger batches, but absolute throughput remains far below GPU across all tested batch sizes.
- `init_time_s` remains near ~18-21s and is consistently higher than GPU startup.

## D) CPU vs GPU Comparison (same 512-prompt workload)

| Batch size | GPU `time_s` | CPU `time_s` | CPU/GPU latency ratio | GPU `throughput` | CPU `throughput` | GPU/CPU throughput ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 8   | 142.423 | 1407.680 | 9.88x | 3.595 | 0.364 | 9.88x |
| 16  | 109.172 | 1045.742 | 9.58x | 4.690 | 0.490 | 9.57x |
| 32  | 94.346  | 905.646  | 9.60x | 5.427 | 0.565 | 9.61x |
| 64  | 86.471  | 773.840  | 8.95x | 5.921 | 0.662 | 8.94x |
| 128 | 81.486  | 682.133  | 8.37x | 6.283 | 0.751 | 8.37x |
| 256 | 79.065  | 630.650  | 7.98x | 6.476 | 0.812 | 7.98x |
| 512 | 76.376  | 598.441  | 7.84x | 6.704 | 0.856 | 7.83x |

### Comparison Takeaways

- GPU is consistently much faster than CPU for this model/workload: roughly **7.8x to 9.9x** across tested batch sizes.
- Both CPU and GPU improve with larger batch sizes, but GPU maintains a large absolute advantage.
- The speedup ratio narrows slightly at very large batch sizes because CPU benefits proportionally more from batching, but GPU still leads clearly.

## Interpretation

1. For this environment, GPU is clearly preferred over CPU for both latency and throughput.
2. For the 512-query workload, larger batch sizes help on both devices, with GPU reaching the highest throughput at BS=512.
3. A practical GPU operating range is `64`-`256` for near-peak performance with potentially safer headroom than max batch.

## Notes and Caveats

- Warnings in logs (CUDA deprecation notices, `nvidia-smi` fallback, attention backend notices) did not prevent successful completion.
- Results are specific to this Jetson Thor + software stack and current conservative compatibility settings.
- For publishable benchmarking, repeat each batch size multiple times and report mean/stddev.

## Next Suggested Step

Run a repeat sweep (e.g., 3 runs per batch size), then compute:

- Mean and stddev of `time_s`
- Mean and stddev of `throughput_prompts_per_s`
- Confidence intervals for throughput improvement vs BS=8

