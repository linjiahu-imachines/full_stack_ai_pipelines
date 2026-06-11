# Transformers CPU vs GPU benchmark — run summary (imu-thor)

**Machine:** `imu-thor` (Jetson AGX Thor)  
**Suite:** `vllm_gpu_test` — **no vLLM** (Hugging Face Transformers only)  
**Mode:** `./run_gpu_tests.sh transformers-cpu-gpu` → `test_transformers_cpu_vs_gpu.py`

---

## Command

```bash
cd /home/linhu/projects/vllm_exploration/vllm_gpu_test
./run_gpu_tests.sh transformers-cpu-gpu
```

**Model:** `facebook/opt-125m`  
**PyTorch:** `2.10.0+cu130` (CUDA build present, but runtime could not use the GPU from this user/session.)

---

## GPU / CUDA

| Check | Result |
|--------|--------|
| `torch.cuda.is_available()` | **False** |
| GPU section | **Skipped** (script only runs GPU when CUDA is available) |
| Console | **NvRmMemInitNvmap** / **Permission denied** style messages — same class of issue as before: process/user cannot use the Jetson GPU device nodes (typically needs **`video`** / **`render`** group + re-login; see [JETSON_GPU_ACCESS.md](JETSON_GPU_ACCESS.md)). |

So this run is a **CPU-only** completion of the benchmark script; **no GPU timings** were collected.

---

## CPU results (Transformers)

Greedy generation (`do_sample=False`), `dtype=float32` on CPU.

| Phase | Wall time | Per prompt |
|--------|-----------|------------|
| **Single** (3 prompts, `max_new_tokens=50`) | **13.15 s** | ~**4,384 ms** |
| **Batch** (5 prompts, `max_new_tokens=30`) | **2.37 s** | ~**473 ms** |

Batch is faster per prompt than the single-query loop because work is batched in one `generate` call instead of three separate forward paths.

---

## Related

- Script: [test_transformers_cpu_vs_gpu.py](test_transformers_cpu_vs_gpu.py)  
- Machine profile: [../docs/MACHINE_IMU_THOR_JETSON_AGX_THOR.md](../docs/MACHINE_IMU_THOR_JETSON_AGX_THOR.md)  
- GPU access (no sudo): [JETSON_GPU_ACCESS.md](JETSON_GPU_ACCESS.md)

After CUDA works for your user, re-run the same command to obtain **GPU** numbers and an automatic **GPU/CPU time ratio** in the log.
