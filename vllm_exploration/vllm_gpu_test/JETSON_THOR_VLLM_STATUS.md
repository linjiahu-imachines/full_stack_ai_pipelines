# vLLM on Jetson AGX Thor — status and setup

**Machine:** Jetson AGX Thor Developer Kit, **aarch64**, Blackwell iGPU (sm **11.0**).  
**Folder:** `vllm_gpu_test/`

---

## What was fixed in this repo

1. **`install_jetson_gpu_deps.sh`** — Removes CPU-only `torch`, then installs **`torch` / `torchvision` / `torchaudio`** from **Jetson AI Lab** PyPI indexes (`jp7/cu130`, then `jp6/cu128`, `jp6/cu126`), then **`pip install -r requirements.txt`**.
2. **`verify_gpu_env.py`** — Fails fast if `torch.cuda.is_available()` is false or `vllm` cannot import.
3. **`run_gpu_tests.sh`** — Runs `verify_gpu_env.py` before any GPU test; clear errors if CUDA stack is wrong.
4. **`gpu_env.py`** / **`vllm_jetson_patch.py`** — `require_cuda()`; **`compilation_config=0`** on CC **11.0**; **`apply_jetson_vllm_runtime_patches()`** (block-table Triton workaround). **`gpu_memory_utilization`**: default **0.10** on CC **11.0** if `VLLM_GPU_MEM_UTIL` unset — vLLM V1 requires **free** memory ≥ total×util at engine start (see `request_memory` in vLLM). **`VLLM_ATTENTION_BACKEND`**, **`cuda_cleanup()`**.
5. **Tests** — vLLM `LLM` instances use `try`/`finally` + cleanup; Transformers loads use **fp16/bf16** on GPU; **`test_batch500_complete.py`** / **`test_large_batch.py`** no longer force Triton on Blackwell.

---

## One-command setup (on the Thor)

```bash
cd /path/to/vllm_exploration/vllm_gpu_test
python3 -m venv venv    # sudo apt install python3.12-venv  if needed
bash install_jetson_gpu_deps.sh
./run_gpu_tests.sh compare
```

If **`pypi.jetson-ai-lab.dev`** does not resolve, install PyTorch using NVIDIA’s guide for your JetPack, then:

```bash
venv/bin/pip install -r requirements.txt
venv/bin/python verify_gpu_env.py
```

---

## Environment hints

| Issue | Try |
|--------|-----|
| **`ptxas-blackwell` / `sm_110a`** during vLLM init or first token | Use **`compilation_config=0`** (auto on 11.0) **and** the **block-table PyTorch patch** (`apply_jetson_vllm_runtime_patches`, automatic on 11.0). Inductor-off alone is **not** enough — V1 still JITs Triton in `block_table.py`. `VLLM_JETSON_BLOCK_TABLE_TRITON=1` forces Triton (likely fails). |
| `Free memory ... less than desired GPU memory utilization` | vLLM requires **free** ≥ total×util. Lower util: `export VLLM_GPU_MEM_UTIL=0.08`–`0.15`, or close other GPU apps. Default **0.10** on Thor if unset. |
| OOM / desktop uses GPU RAM | `export VLLM_GPU_MEM_UTIL=0.5` (only if enough free memory for that fraction) |
| vLLM CUDA graph / custom op errors | `export VLLM_ENFORCE_EAGER=1` |
| Attention backend errors | `export VLLM_ATTENTION_BACKEND=FLASHINFER` or `TRITON_ATTN` |

---

## Full README

See [README.md](README.md) for all options and file descriptions.

---

## `torch 2.10.0+cu130` but `cuda.is_available()` is False

See **[JETSON_GPU_ACCESS.md](JETSON_GPU_ACCESS.md)** (NvRm / nvmap permission denied, `video` group, reboot).
