# vLLM GPU Testing Suite

This project compares LLM inference performance with and without vLLM on GPU deployment.

**Jetson AGX Thor (aarch64):** see [JETSON_THOR_VLLM_STATUS.md](JETSON_THOR_VLLM_STATUS.md). Use `install_jetson_gpu_deps.sh` so PyTorch is **CUDA-enabled** (plain `pip install -r requirements.txt` often pulls `torch … +cpu` on aarch64).

If install succeeds but **`torch.cuda.is_available()` is still False** with **NvRm/nvmap** errors, see **[JETSON_GPU_ACCESS.md](JETSON_GPU_ACCESS.md)** (usually: add user to **`video`** group and reboot).

### Transformers only — CPU vs GPU (no vLLM)

Same prompts as the GPU suite, greedy `generate`, **no vLLM**. Runs **CPU** always; runs **GPU** if `torch.cuda.is_available()`.

```bash
./run_gpu_tests.sh transformers-cpu-gpu
# or:
venv/bin/python test_transformers_cpu_vs_gpu.py
```

Options: `--skip-gpu`, `--skip-cpu`, `--model`, `--batch-size`, `--max-new-tokens-single`, `--max-new-tokens-batch`.

## Hardware Requirements

- NVIDIA GPU with CUDA (discrete x86 server or Jetson integrated GPU)
- CUDA-capable PyTorch build (`torch.cuda.is_available()` must be **True**)
- Enough unified/VRAM for `facebook/opt-125m` (~500 MB download)

## Setup

### 1. Create a virtual environment

```bash
cd /path/to/vllm_exploration/vllm_gpu_test
sudo apt install -y python3.12-venv   # Ubuntu / Jetson if needed
python3 -m venv venv
```

### 2. Install dependencies

**Jetson (AGX Thor, Orin, etc.):** install **CUDA PyTorch first**, then the rest:

```bash
bash install_jetson_gpu_deps.sh
```

**x86_64 Linux with NVIDIA GPU:** use your distro’s CUDA PyTorch instructions, then:

```bash
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 3. Verify

```bash
./run_gpu_tests.sh verify
# or:
venv/bin/python verify_gpu_env.py
```

You should see `torch` **without** `+cpu` and `CUDA OK`.

## Running tests

```bash
./run_gpu_tests.sh              # verify env, then full comparison (default)
./run_gpu_tests.sh compare      # same
./run_gpu_tests.sh with         # vLLM only
./run_gpu_tests.sh without      # Transformers on GPU only
./run_gpu_tests.sh verify       # environment check only
```

With an activated venv you can still run:

```bash
python test_with_vllm_gpu.py
python test_without_vllm_gpu.py
python test_gpu_comparison.py
```

## Configuration (environment variables)

| Variable | Purpose |
|----------|---------|
| `GPU_TEST_MODEL` | Hugging Face model id (default `facebook/opt-125m`) |
| `VLLM_GPU_MEM_UTIL` | vLLM `gpu_memory_utilization`. vLLM V1 requires **free** memory ≥ total×this value at startup. Default **0.85** on typical GPUs; on **CC 11.0** (Jetson Thor) default **0.10** if unset. Increase (e.g. **0.5**) when vLLM runs alone with little else using the GPU. |
| `VLLM_ENFORCE_EAGER` | Set to `1` if vLLM fails during CUDA graph / custom op init |
| `VLLM_DTYPE` | Optional vLLM `dtype` (e.g. `float16`, `bfloat16`) |
| `TRANSFORMERS_DTYPE` | `float16` (default) or `bfloat16` for HF loads |
| `VLLM_ATTENTION_BACKEND` | Override auto-pick (`FLASHINFER` on sm ≥ 10, else `TRITON_ATTN`) |
| `VLLM_COMPILATION_MODE` | **Jetson Thor (CC 11.0):** unset/`auto` forces `compilation_config=0` (no `torch.compile`). Does **not** fix V1 block-table Triton kernels; see `VLLM_JETSON_BLOCK_TABLE_TRITON`. |
| `VLLM_JETSON_BLOCK_TABLE_TRITON` | Set to `1` to force stock Triton for slot mapping (fails on Thor if `ptxas` rejects `sm_110a`). Default / unset: on CC **11.0**, `gpu_env` applies a PyTorch fallback (`vllm_jetson_patch.py`) before importing vLLM. |

## Test files

- **gpu_env.py** — CUDA check, vLLM kwargs, cleanup between engines; **`apply_jetson_vllm_runtime_patches()`**
- **vllm_jetson_patch.py** — PyTorch fallback for V1 `BlockTable.compute_slot_mapping` on Thor (sm_110a)
- **verify_gpu_env.py** — Fast gate: CUDA torch + vLLM import
- **test_with_vllm_gpu.py** — vLLM single/batch/multi-GPU (multi skips if one GPU)
- **test_without_vllm_gpu.py** — Transformers on GPU
- **test_gpu_comparison.py** — Full comparison with cleanup between phases
- **test_batch500_complete.py** / **test_large_batch.py** — Heavy batch tests (attention backend auto-selected by GPU generation)

## Expected results

On many discrete GPUs, vLLM wins on throughput vs naive Transformers `generate`. **opt-125m** is tiny; gains are often small. Larger models (&gt;1B) usually show clearer vLLM benefits.

## Troubleshooting

### `verify_gpu_env.py` fails with `+cpu`

Re-run **`bash install_jetson_gpu_deps.sh`** (Jetson) or install a **CUDA** `torch` wheel matching your driver/JetPack, then `pip install -r requirements.txt` again.

### `fatal error: Python.h: No such file or directory` (during vLLM / Triton / torch.compile)

vLLM uses **Triton** and **torch.compile**; building small helpers needs **Python development headers** and a compiler:

```bash
sudo apt-get install -y python3.12-dev build-essential
```

Then re-run `./run_gpu_tests.sh compare`.

### `ImportError: libcudart.so.12: cannot open shared object file`

PyTorch **cu130** wheels ship **CUDA 13** (`libcudart.so.13`). The **vLLM** native module (`vllm._C`) is built against **CUDA 12** (`libcudart.so.12`). Install the pip runtime package:

```bash
pip install 'nvidia-cuda-runtime-cu12>=12.4,<13'
```

This is included in **`requirements.txt`** and **`install_jetson_gpu_deps.sh`**; re-run install if you still see the error.

### `ptxas-blackwell fatal : Value 'sm_110a' is not defined for option 'gpu-name'`

Jetson Thor uses **sm_110a** (CUDA capability **11.0**). Triton’s `ptxas-blackwell` may reject that **gpu-name**. Failures can come from:

1. **`torch.compile` / Inductor** — mitigated with **`compilation_config=0`** (automatic on 11.0).
2. **vLLM V1 `BlockTable.compute_slot_mapping`** — still uses a **Triton JIT** kernel unless patched.

This repo calls **`apply_jetson_vllm_runtime_patches()`** before importing vLLM (see `test_with_vllm_gpu.py`, `verify_gpu_env.py`, etc.), which replaces (2) with a **PyTorch** implementation on CC **11.0**. To force the original Triton path (will likely fail until NVIDIA updates the toolchain):

```bash
export VLLM_JETSON_BLOCK_TABLE_TRITON=1
```

To only adjust `torch.compile`:

```bash
export VLLM_COMPILATION_MODE=default   # try vLLM default compilation
export VLLM_COMPILATION_MODE=0         # force eager (no Inductor)
```

### vLLM kernel / attention errors on Jetson

Try:

```bash
export VLLM_ENFORCE_EAGER=1
export VLLM_ATTENTION_BACKEND=FLASHINFER   # or TRITON_ATTN as a fallback
./run_gpu_tests.sh with
```

### `Free memory on device ... is less than desired GPU memory utilization`

vLLM V1 checks that **free** GPU memory is at least **total device memory × `gpu_memory_utilization`** before allocating the KV cache. A high value (e.g. **0.85**) requires almost the entire device to be free. After another framework has used the GPU, or on a busy Jetson desktop, lower it:

```bash
export VLLM_GPU_MEM_UTIL=0.10
```

On Jetson Thor (CC 11.0), the suite defaults to **0.10** when `VLLM_GPU_MEM_UTIL` is unset.

### CUDA OOM (unified memory)

Lower utilization:

```bash
export VLLM_GPU_MEM_UTIL=0.5
```

## Comparison with CPU results

Sibling project: `vllm_test/` (CPU). There, Transformers often beats vLLM; on GPU this suite expects the opposite trend for larger workloads.
