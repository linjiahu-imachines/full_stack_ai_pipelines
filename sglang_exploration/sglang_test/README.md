# sglang_test — CPU and GPU offline experiments

## One-time environment (project venv)

From the repo root:

```bash
cd /home/linhu/projects/sglang_exploration
./setup_sglang_env.sh
source ./activate_sglang_env.sh
```

This creates **`../sglang_venv/`** with:

- **PyTorch** `2.9.1+cu130` (matches **SGLang 0.5.9**)
- **SGLang** and its dependencies
- **CUDA 12** user-space libraries under `site-packages/nvidia/.../lib` (so `sgl_kernel` can load `libcudart.so.12`, etc.)
- A small local **`torch_memory_saver` stub** under [`packaging/stub_torch_memory_saver`](../packaging/stub_torch_memory_saver) — the real PyPI package must compile C++/CUDA extensions and needs a full **CUDA toolkit** (`nvcc` + headers) in `/usr/local/cuda`, which this Jetson image does not ship by default. The stub satisfies imports; install the real package after `sudo apt install nvidia-cuda-toolkit` (or JetPack dev meta) if you need memory-saver features.

Always **`source ../activate_sglang_env.sh`** before runs so **`LD_LIBRARY_PATH`**, **`SGLANG_EXPLORATION_ROOT`**, and optionally **`CUDA_HOME`** are set (see `activate_sglang_env.sh`).

**RoPE / worker processes:** Debian ships **`/usr/lib/python3.12/sitecustomize.py`**, which is imported instead of a venv **`site-packages/sitecustomize.py`**. `setup_sglang_env.sh` therefore installs **`_sglang_exploration_rope.pth`** (an `import runpy; …` line) so the RoPE torch fallback runs in **SGLang scheduler workers** as well as the main process.

## Default model

[`run_offline_batch_experiment.py`](run_offline_batch_experiment.py) defaults to **`Qwen/Qwen3-1.7B`**. Override with `--model`.

## Commands

| Script | Role |
|--------|------|
| [run_offline_batch_experiment.py](run_offline_batch_experiment.py) | Main entry: `--device auto\|cpu\|cuda`, timings, `__RESULT_JSON__:` |
| [run_smoke_cpu.sh](run_smoke_cpu.sh) | Tiny CPU batch (uses `../sglang_venv/bin/python` when present) |
| [run_smoke_gpu.sh](run_smoke_gpu.sh) | Tiny GPU batch |
| [run_experiment_both.sh](run_experiment_both.sh) | CPU then GPU (synthetic prompts; `BATCH_SIZE` = prompt count) |
| [run_experiment_gpu_batch.sh](run_experiment_gpu_batch.sh) | GPU-only: `./run_experiment_gpu_batch.sh 32` (same as `--device cuda --batch-size 32`) |

Examples:

```bash
source /home/linhu/projects/sglang_exploration/activate_sglang_env.sh
# Default: 512 prompts from sglang_test/data/queries_512.jsonl; --batch-size = prompts per generate()
python /home/linhu/projects/sglang_exploration/sglang_test/run_offline_batch_experiment.py --device cuda --batch-size 16
# Tiny smoke (built-in prompts, not the 512 file)
python /home/linhu/projects/sglang_exploration/sglang_test/run_offline_batch_experiment.py --device cuda --synthetic --num-prompts 2 --batch-size 2 --max-new-tokens 16
```

Flags:

- **`--enable-cuda-graph`** — opt in to CUDA graph capture (default is off; avoids some graph-capture failures).
- **`SGLANG_FORCE_FLASHINFER_AARCH64=1`** — on **aarch64** the script forces **`attention_backend=torch_native`** by default to avoid Triton `ptxas` paths; set this env to `1` to try FlashInfer anyway.
- **`SGLANG_FORCE_TORCH_ROPE_CUDA`** — **`auto`** (default): if no **`nvcc`** is found, **`run_offline_batch_experiment.py`** patches RoPE to a pure PyTorch path so the first forward does not JIT-compile CUDA with TVM-FFI. Use **`0`** to disable, **`1`** to force PyTorch RoPE even when **`nvcc`** exists.

## Jetson Thor (compute capability **11.0**) — current limits

On **NVIDIA Thor** with this stack:

1. **Triton / `ptxas`**: kernels target **`sm_110a`**, but the **`ptxas` shipped with PyTorch/Triton** rejects `--gpu-name=sm_110a` (`ptxas fatal : Value 'sm_110a' is not defined`). CUDA graph capture hits the same issue. The experiment script therefore sets **`disable_cuda_graph=True`** by default on GPU and prefers **`torch_native`** attention on **aarch64**.
2. **JIT CUDA kernels (RoPE)**: fused RoPE may JIT-compile via TVM-FFI and needs **`nvcc`**. The offline script defaults to **PyTorch-native RoPE on CUDA when `nvcc` is missing** (`SGLANG_FORCE_TORCH_ROPE_CUDA=auto`). For best performance, install the host **CUDA toolkit** (JetPack dev / **`nvidia-cuda-toolkit`**) and point **`CUDA_HOME`** at it; avoid setting **`CUDA_HOME=/usr`** without a real **`bin/nvcc`** there.
3. **CPU `Engine`**: the prebuilt **`sgl-kernel`** wheel may not expose every CPU symbol (e.g. `init_cpu_threads_env`), so **CPU `sgl.Engine` can fail** until SGLang/sgl-kernel publish a matching CPU build.

So: the **environment is installed and importable**, but **end-to-end `Engine` + Qwen3-1.7B on this Thor image still needs host CUDA toolkit + newer toolchain support for sm_110a**, or run SGLang in an upstream **Docker** image on a GPU where `ptxas` already supports your architecture.

## Output

The Python script prints:

`__RESULT_JSON__:{...}`

Use `--output /path/to/result.json` to save a copy.
