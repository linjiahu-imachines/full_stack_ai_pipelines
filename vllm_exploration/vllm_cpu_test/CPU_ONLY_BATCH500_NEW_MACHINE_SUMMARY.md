# CPU-Only Batch-500 Summary on New Machine

**Date**: 2026-03-10  
**Model**: `facebook/opt-125m`  
**Workload**: 500 prompts, `max_new_tokens=30`  
**Machine**: Azure VM (CPU-only, no NVIDIA GPU attached)

---

## Machine Configuration

| Item | Value |
|---|---|
| CPU | Intel Xeon Platinum 8573C |
| Logical CPUs | 8 |
| Physical cores | 4 |
| Threads per core | 2 |
| Sockets | 1 |
| NUMA nodes | 1 |
| RAM | 31 GiB |
| GPU | None (`torch.cuda.is_available() = False`) |
| Disk (`/`) | 2.0T total, ~1.9T free |
| Notable CPU flags | `avx2`, `avx512f`, `avx512_vnni`, `avx512_bf16`, `amx_tile`, `amx_int8`, `amx_bf16` |

---

## Test Setup

- **Without vLLM**: HuggingFace Transformers on CPU (`vllm_gpu_test/test_batch500_complete.py`, Test 1)
- **With vLLM**: vLLM CPU backend (`vllm_cpu_test/test2_vllm_cpu.py`, invoked from `vllm_gpu_test/test_batch500_complete.py` Test 2)
- **Batch size**: 500
- **Environment**: CPU-only (GPU tests skipped/unsupported on this node)

---

## Performance Results (Batch 500)

| Scenario | Total Time | Per Prompt | Throughput | Notes |
|---|---:|---:|---:|---|
| Transformers on CPU (without vLLM) | **28.17s** | **56.34ms** | **17.75 prompts/sec** | Uses all 8 logical threads |
| vLLM on CPU | 62.75s | 125.50ms | 7.97 prompts/sec | Auto-binds to 4 physical-core worker threads |

---

## CPU Utilization Snapshot

### Without vLLM (Transformers CPU)
- Active threads (>5%): **8/8**
- Average CPU utilization: **31.4%**
- Max process threads: **13**
- Average memory: **3294 MB**

### With vLLM (CPU backend)
- Active threads (>5%): **4/8**
- Average CPU utilization: **40.2%**
- Max process threads: **22**
- Average memory: **579 MB**

### Why only 4/8 threads were active in vLLM CPU

This is expected for this VM shape (4 physical cores, 8 logical threads with Hyper-Threading).
vLLM CPU auto thread-binding selected one worker thread per physical core, as shown in the
runtime log:

- `auto thread-binding list ... [(1,0), (3,1), (5,2), (7,3)]`

So vLLM intentionally used 4 worker threads (one per physical core), not all 8 logical threads.
This default strategy usually avoids sibling-SMT contention and can improve stability/cache
behavior on CPU inference workloads.

If needed, thread policy can be overridden for experimentation:

- `VLLM_CPU_OMP_THREADS_BIND=0-7` (bind to all 8 logical CPUs), or
- `VLLM_CPU_OMP_THREADS_BIND=nobind` with `OMP_NUM_THREADS=8`

Note: on small VMs, forcing all logical threads does not always improve throughput.

---

## Comparison and Key Findings

1. **vLLM CPU is slower for this workload on this machine**
   - Time ratio: `62.75 / 28.17 = 2.23x`
   - Throughput ratio: `17.75 / 7.97 = 2.23x`
   - Result: vLLM CPU is about **2.23x slower** than Transformers CPU for batch-500 on this 4-core VM.

2. **Transformers spreads work across all logical CPUs**
   - 8/8 active logical threads with better throughput.

3. **vLLM CPU focuses on physical-core worker binding**
   - 4/8 active threads (matching physical-core pattern), higher per-thread load, lower end-to-end throughput here.

4. **Memory behavior differs**
   - vLLM CPU used much less process memory in this run (about 579 MB vs 3294 MB), but that did not translate into better speed.

---

## Conclusion

On this new CPU-only VM (4 physical cores / 8 logical), for `facebook/opt-125m` and batch size 500:

- **Best performance**: Transformers on CPU (without vLLM)
- **vLLM CPU**: functional and stable, but slower in this specific setup/workload

Recommended default on this machine for offline batch inference: **Transformers CPU**.
