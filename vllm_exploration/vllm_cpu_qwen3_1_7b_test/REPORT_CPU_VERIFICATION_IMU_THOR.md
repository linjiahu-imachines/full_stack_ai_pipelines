# vLLM verification on imu-thor — CPU vs GPU (`Qwen/Qwen3-1.7B`)

This document records **pre-flight verification** (steps 1–4), completed **vLLM batch-500 `max_num_seqs` sweeps** on **CPU** and **CUDA/GPU**, and a **comparison** of the two vLLM backends on the same host. vLLM supports **both**: the **`+cpu`** wheel (`vllm_cpu_venv`) and **CUDA** vLLM (`vllm_gpu_test/venv`); they are separate installs and are **not** interchangeable in one process.

The **optional** full Transformers-vs-vLLM multi-batch CPU run (`run_qwen3_cpu_test.sh`) is listed at the end.

---

## Environment

### Summary

| Item | Value |
|------|--------|
| Host | `imu-thor` |
| OS / kernel | Linux **`6.8.12-tegra`** (Jetson stack) |
| **CPU** | **14×** Arm **Neoverse-V3AE-class** cores (`aarch64`), **~2.6 GHz** max |
| **GPU** | **NVIDIA Thor** ( **Blackwell** ), **20 SMs**, **2560** CUDA cores (Jetson **T5000**–class), CC **11.0** (`sm_110a`) |
| System RAM | ~**122.8 GiB** total (typical during runs) |
| **vLLM CPU** | `vllm_cpu_venv` → **`vllm==0.19.0+cpu`** |
| **vLLM GPU** | `vllm_gpu_test/venv` → **vLLM CUDA v0.19.0**, PyTorch **`2.10.0+cu130`** |
| Model (all sweeps) | `Qwen/Qwen3-1.7B` |

**CPU runs:** `device_config=cpu`, `cpu_worker`; logs may still say “GPU KV cache size” (generic wording).

**GPU runs:** `device_config=cuda`; Jetson **`gpu_env.py`** + **`vllm_jetson_patch.py`** (block-table + top-k/top-p PyTorch fallbacks for **`sm_110a`** / `ptxas-blackwell`). **`gpu_memory_utilization`** default **0.10** on CC 11.0 unless **`VLLM_GPU_MEM_UTIL`** is set.

---

### CPU — Arm cluster (details)

This machine exposes a **homogeneous** Arm application cluster (no big.LITTLE split in `lscpu` for this socket).

| Topic | Detail |
|--------|--------|
| **ISA** | **AArch64** (`aarch64`), **little-endian**, **Armv8-A** (`CPU architecture: 8` in `/proc/cpuinfo`) |
| **Vendor / core ID** | **Arm Limited** (`CPU implementer: 0x41`). **MIDR part number `0xd83`** — in toolchains such as LLVM and util-linux’s `lscpu` tables this maps to **Neoverse-V3AE** (an **automotive / embedded** profile related to the **Neoverse V3** line). `lscpu` often shows **`Model name: -`** on Jetson; identification is from **`/proc/cpuinfo`** fields above. |
| **Topology** | **14** logical CPUs = **14** cores, **1** thread per core (no SMT). **1** cluster, **1** NUMA node (CPUs **0–13** all on `node0`). |
| **Frequency** | From `lscpu`: **~54 MHz** min · **~2601 MHz** max (DVFS; actual clocks depend on governor and load). |
| **Caches (per `lscpu`)** | **L1d** 896 KiB (14 instances), **L1i** 896 KiB (14), **L2** 14 MiB (14) — consistent with **per-core private L2** on the order of **1 MiB/core**. |
| **ISA features (selection)** | **NEON** (`asimd`), **SVE2** (`sve2`, `sveaes`, …), **bfloat16** (`bf16`, `svebf16`), **Int8 dot-product** (`i8mm`, `svei8mm`), **SHA3/SM3/SM4**, **PAC/BTI** (`paca`, `pacg`, `bti`), and other Armv8.x extensions listed under **`Flags:`** in `lscpu`. These matter for math libraries (BLAS, attention kernels) and for **CPU** inference paths that can exploit wide SIMD. |

**Relevance to this report:** vLLM’s **CPU wheel** runs inference on this cluster using PyTorch/CPU kernels compiled for **aarch64**; throughput is bounded by **core count**, **memory bandwidth**, and **single-threaded vs batched** scheduling — not by GPU.

---

### GPU — NVIDIA Thor (details)

**Product line:** This host matches the **Jetson Thor** / **Jetson T5000** class ( **14** Neoverse-V3AE CPU cores + **~128 GB** unified memory). Figures below combine **runtime queries** (PyTorch / `nvidia-smi`) with **NVIDIA Jetson Thor / T5000** module documentation where the driver does not print core counts.

| Topic | Detail |
|--------|--------|
| **Product** | **NVIDIA Thor** (`nvidia-smi` **Product Name**), **Blackwell** **Product Architecture** (not to be confused with the CPU vendor “Arm”). |
| **Compute capability** | **11.0** — PyTorch reports `torch.cuda.get_device_capability(0) == (11, 0)`; vLLM/Triton builds target **`sm_110a`** on this platform. |
| **Streaming Multiprocessors (SMs)** | **20** — `torch.cuda.get_device_properties(0).multi_processor_count` on this system. |
| **CUDA cores (FP32)** | **2560** total (NVIDIA **Jetson T5000** spec) → **128** CUDA cores per SM (2560 ÷ 20). |
| **Tensor Cores** | **5th-generation** Blackwell Tensor Cores (NVIDIA docs). Some **T5000** datasheet revisions list **96** Tensor Cores total; newer datasheet revisions may omit the explicit count in tables. **`nvidia-smi` / PyTorch do not report a separate “tensor core count”** — use NVIDIA’s module datasheet for marketing block counts. |
| **GPC / TPC topology** | **3 GPC**, **10 TPC** (Jetson **T5000** / full Thor GPU in NVIDIA docs). |
| **Execution properties (PyTorch)** | **Warp size** **32**; **max threads per SM** **1536**; **registers per SM** **65536**; **max threads per block** **1024** (`torch.cuda.get_device_properties(0)`). |
| **Memory size (unified)** | **~122.8 GiB** reported as `total_memory` on GPU 0 (matches **128 GB** LPDDR5X class modules; exact bytes vary). |
| **PCI / bus** | **PCIe** device **`00000000:01:00.0`**, **NVIDIA device ID `0x2B0010DE`**, board **`GPU Part Number 2B00--A1`** (`nvidia-smi -q`). **Note:** `nvidia-smi` may report a **low** current link speed (**Gen 1**, **1×**) when the link is idle or constrained by power state; **max** advertised capabilities can differ — treat **snapshot** values as **non-final** for benchmarking unless you pin workload and recheck. |
| **Driver / CUDA user stack** | **Driver `580.00`**, **`CUDA Version` 13.0** (driver-reported). PyTorch in `vllm_gpu_test/venv` is built with **`torch.version.cuda == 13.0`** (**`+cu130`**). |
| **Memory model** | Jetson-class integration: framebuffer memory may show **`[N/A]`** or **“Not Supported”** in some `nvidia-smi` fields — the SoC uses **unified memory** semantics for application use; vLLM’s **`gpu_memory_utilization`** and Jetson **`gpu_env.py`** defaults reflect that (fraction of a **pool**, not a classic discrete **VRAM** bar). |
| **Display / MIG** | No display attached in sample query; **MIG** disabled; **ECC** not applicable / N/A as reported. |
| **Relevance to this report** | Weights and **GEMM/attention** run on this GPU; the **Arm CPU** runs Python, the **vLLM scheduler**, tokenizer, and IPC. **Triton** codegen for **Blackwell** can hit **`ptxas-blackwell` / `sm_110a`** issues — mitigated in-repo by **`vllm_jetson_patch.py`** (PyTorch fallbacks where noted). |

**Smaller SKU (not this machine):** **Jetson T4000** is advertised as **1536** CUDA cores, **6 TPC**, **2 GPC** — use only for comparison; this report’s **20 SM** / **~128 GB** configuration aligns with **T5000**-class specs.

---

## Step 1 — Confirm CPU vLLM package

**Purpose:** Ensure the interpreter exposes a **CPU** vLLM build (`importlib.metadata.version('vllm')` contains `cpu`).

**Command:**

```bash
/home/linhu/projects/vllm_exploration/vllm_cpu_venv/bin/python3 -c \
  "import importlib.metadata as m; assert 'cpu' in m.version('vllm').lower(); print('vLLM OK:', m.version('vllm'))"
```

**Result:**

```text
vLLM OK: 0.19.0+cpu
```

**Status:** Passed.

---

## Step 2 — vLLM CPU smoke (tiny model)

**Purpose:** One generation on CPU without loading Qwen — validates vLLM engine init, tokenizer, and a single request (`facebook/opt-125m`, batch 1, 8 new tokens).

**Command:**

```bash
/home/linhu/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test/smoke_vllm_cpu_one_query.sh
```

**Observed highlights:**

- `device_config=cpu`, `cpu_worker.py` in logs.
- Example timing line from stdout: `init_time_s` ≈ **9.98**, generation `time_s` ≈ **1.35**, `per_prompt_ms` ≈ **1354** (values vary slightly run-to-run).

**Result line (example):**

```text
__RESULT_JSON__:{"engine": "qwen3-1.7B with vLLM", "model": "facebook/opt-125m", "batch_size": 1, "max_new_tokens": 8, "max_num_seqs": 1, "init_time_s": 9.98, "time_s": 1.35, "per_prompt_ms": 1354.31, "throughput_prompts_per_s": 0.74}
```

**Status:** Passed.

---

## Step 3 — Transformers-only path (Qwen on CPU)

**Purpose:** Verify Hugging Face load + CPU `generate` for the real model, **without** vLLM.

**Command:**

```bash
cd /home/linhu/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test
/home/linhu/projects/vllm_exploration/vllm_cpu_venv/bin/python3 test_qwen3_cpu_compare.py \
  --transformers-only \
  --batch-sizes 1 \
  --max-new-tokens 8
```

**Results:**

| Metric | Value |
|--------|--------|
| Batch size | 1 |
| Max new tokens | 8 |
| Wall time (without vLLM) | **4.48 s** |
| Per prompt | **4480.13 ms** |

**Output file:** `qwen3_1_7b_cpu_results_transformers_only.json` (under `vllm_cpu_qwen3_1_7b_test/`).

**Notes:** Transformers may print that some generation flags are not valid (`temperature`, `top_p`, `top_k`); generation still completed.

**Status:** Passed.

---

## Step 4 — End-to-end dry run (Transformers + vLLM CPU, batch 1)

**Purpose:** Same harness as the full study, but **one** batch size and short generation — validates both arms and JSON output.

**Command:**

```bash
cd /home/linhu/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test
/home/linhu/projects/vllm_exploration/vllm_cpu_venv/bin/python3 test_qwen3_cpu_compare.py \
  --batch-sizes 1 \
  --max-new-tokens 8 \
  --vllm-max-num-seqs 32
```

**Results (batch size 1):**

| Arm | Result |
|-----|--------|
| Without vLLM (Transformers) | **3.20 s**; per prompt **3202.66 ms** |
| With vLLM (CPU) | **0.97 s** gen (**18.96 s** init); per prompt **968.66 ms** |

**Ratios printed (batch 1 only):**

```json
{
  "1": {
    "time_ratio_vllm_over_transformers": 0.3031,
    "throughput_ratio_vllm_over_transformers": 3.3226
  }
}
```

**Output file:** `qwen3_1_7b_cpu_results.json` (same directory; **overwrites** prior default-named runs unless you pass `--output`).

**Interpretation:** For batch **1** on imu-thor, vLLM CPU’s **steady-state** generation time was lower than Transformers after a **one-time engine init**.

**Status:** Passed.

---

## vLLM CPU only — batch 500, `max_num_seqs` sweep (completed)

**Purpose:** **500** prompts, measure how **vLLM CPU** scales with **`max_num_seqs`**. **Transformers not run.**

**Command:**

```bash
/home/linhu/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test/run_vllm_cpu_batch500_seq_sweep.sh
# Optional: include 32 and rerun 64 only
/home/linhu/projects/vllm_exploration/vllm_cpu_venv/bin/python3 \
  /home/linhu/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test/run_vllm_cpu_batch500_seq_sweep.py \
  --max-num-seqs-list 32,64 --tag imu_thor
```

**Settings:** `Qwen/Qwen3-1.7B`, **500** prompts, **`max_new_tokens` = 30**, `enforce_eager=True`, vLLM **0.19.0+cpu**.

### Results (generation time; `init_time_s` per new process)

| `max_num_seqs` | Gen time `time_s` | ms / prompt | Throughput (prompts/s) | Init `init_time_s` |
|----------------|-------------------|-------------|-------------------------|---------------------|
| **32** | **163.65** | 327.30 | **3.06** | 11.79 |
| **64** | **119.41** | 238.82 | **4.19** | 12.25 |
| **128** | **186.26** | 372.53 | **2.68** | 13.13 |
| **256** | **135.97** | 271.94 | **3.68** | 12.93 |

### Analysis (CPU)

1. **32 → 64** improves throughput (**3.06 → 4.19** prompts/s) — more scheduler concurrency helps, consistent with the qualitative pattern on this host.
2. **64 vs 128 / 256:** The **2026-04-08** **64** rerun is **faster** than the older **128** row (**119 s** vs **186 s** gen). That means the **128** and **256** entries from **2026-04-07** are **not** directly comparable as a monotonic sweep with the new **64** point — mix of **different run days** and system state. Treat **32 / 64** (**2026-04-08**) as one pair; use **128 / 256** as indicative of higher **`max_num_seqs`** on an earlier sweep, or re-run **128 / 256** on the same day if you need a strict ordering table.
3. **Init ~12–13 s** per subprocess (each sweep step is a **new** engine).
4. **`VLLM_CPU_KVCACHE_SPACE`** was unset in logs (large default KV warning); optional tuning.

**Artifacts:** `qwen3_1_7b_cpu_results_batch500_seq{32,64,128,256}_imu_thor.json`  
**Runs:** **32 / 64** — **2026-04-08**; **128 / 256** — **2026-04-07**.

**Status:** Completed.

---

## vLLM GPU (CUDA) — batch 500, `max_num_seqs` sweep (completed)

**Clarification:** This is **vLLM’s CUDA backend** (compute on **GPU**). It uses **`vllm_gpu_test/venv`**, not the **`+cpu`** wheel.

**Purpose:** Same workload as the CPU sweep — **`Qwen/Qwen3-1.7B`**, **500** prompts, **`max_new_tokens` = 30**, **`max_num_seqs`** sweep (includes **32**, **64**, **128**, **256**; use `--max-num-seqs-list` to customize).

**Command:**

```bash
export GPU_VLLM_PYTHON=/home/linhu/projects/vllm_exploration/vllm_gpu_test/venv/bin/python3
/home/linhu/projects/vllm_exploration/vllm_gpu_test/run_vllm_gpu_batch500_seq_sweep.sh
```

**Settings:** `gpu_env.vllm_llm_kwargs` ( **`gpu_memory_utilization` = 0.10** default on CC 11.0), **`compilation_config` = eager (0)** on Thor, Jetson **Triton workarounds** in `vllm_jetson_patch.py` (block table + **`apply_top_k_top_p` → PyTorch**) so **`sm_110a`** does not hit broken **`ptxas-blackwell`** paths.

### Results (generation time)

| `max_num_seqs` | Gen time `time_s` | ms / prompt | Throughput (prompts/s) | Init `init_time_s` |
|----------------|-------------------|-------------|-------------------------|---------------------|
| **32** | **17.35** | 34.69 | **28.82** | 14.32 |
| **64** | **10.34** | 20.67 | **48.37** | 14.48 |
| **128** | **8.21** | 16.42 | **60.91** | 15.56 |
| **256** | **6.14** | 12.28 | **81.43** | 16.08 |

### Analysis (GPU)

1. **Concurrency vs batch size:** With **500** prompts, **`max_num_seqs`** caps how many sequences the scheduler can run **in parallel**. Throughput increases monotonically from **32 → 64 → 128 → 256** on these runs — **32** is the most **under-scheduled** (longest **gen** time); **256** is the fastest.
2. **64 vs 128:** Still a clear gain (**~10.3 s → ~8.2 s** gen; **~48 → ~61** prompts/s on these runs), but **not** the orders-of-magnitude gap seen in the old **~128 s** **64** measurement. Treat that older **64** result as **untrusted** unless reproduced; the **2026-04-08** **64** rerun is the reference for **seq 64** going forward.
3. **128 → 256**: smaller improvement (**~8.2 s → ~6.1 s**), typical diminishing returns once the GPU is well fed.
4. Logs may show **`Unknown vLLM environment variable: VLLM_ATTENTION_BACKEND`** (set in `test_qwen3_vllm_gpu_batch.py`); harmless. **`spawn`** worker method is expected after CUDA init in the parent.

**Artifacts:** `qwen3_1_7b_gpu_results_batch500_seq{32,64,128,256}_imu_thor.json`  
**Runs:** **128 / 256** — **2026-04-07**; **32 / 64** (rerun) — **2026-04-08**.

**Status:** Completed.

---

## vLLM CPU vs GPU — side-by-side (same model, batch 500, `max_num_seqs` sweep)

Both backends are **vLLM**; only the **platform** differs (**CPU wheel** vs **CUDA**). Same prompt list and **`max_new_tokens` = 30**.

### Throughput (prompts/s)

**CPU @ 32 / 64** — **2026-04-08**; **CPU @ 128 / 256** — **2026-04-07**. **GPU** timings are as in the GPU section (mixed dates; **32 / 64** GPU **2026-04-08**).

| `max_num_seqs` | vLLM **CPU** | vLLM **GPU** | GPU / CPU ratio |
|----------------|-------------|-------------|-----------------|
| **32** | **3.06** | **28.82** | **9.4×** |
| **64** | **4.19** | **48.37** | **11.5×** |
| **128** | 2.68 | 60.91 | **22.7×** |
| **256** | 3.68 | 81.43 | **22.1×** |

### Wall time for 500 prompts (generation only, seconds)

| `max_num_seqs` | vLLM **CPU** | vLLM **GPU** | CPU time / GPU time |
|----------------|-------------|-------------|----------------------|
| **32** | **163.65** | **17.35** | **9.4×** slower on CPU |
| **64** | **119.41** | **10.34** | **11.5×** slower on CPU |
| **128** | 186.26 | 8.21 | **22.7×** slower on CPU |
| **256** | 135.97 | 6.14 | **22.1×** slower on CPU |

### Interpretation

- **vLLM** is one framework with **two distinct builds** here: **`+cpu`** (Arm) vs **CUDA** (Jetson Thor). Pick the venv that matches the target device.
- For **32 / 64** on **2026-04-08**, GPU is **~9–12×** faster than vLLM CPU on these runs — lower than the **~22×** at **128 / 256** partly because **CPU rows for 128 / 256** are from a **different day** and do not line up monotonically with the new **64** CPU measurement (see CPU analysis). For a clean apples-to-apples matrix, re-run **all** **`max_num_seqs`** values on one day.
- **GPU @ 32** is slower than **GPU @ 64+** because **500** prompts cannot be fully overlapped with only **32** concurrent sequence slots; raise **`max_num_seqs`** when the workload allows (memory permitting).

---

## Step 5 — Full multi-batch CPU experiment (optional)

**Purpose:** **Transformers vs vLLM (CPU)** for batch sizes **1, 500, 1000, 2000** in one `test_qwen3_cpu_compare.py` invocation.

**Command:**

```bash
/home/linhu/projects/vllm_exploration/vllm_cpu_qwen3_1_7b_test/run_qwen3_cpu_test.sh
```

**Expected artifact:** `qwen3_1_7b_cpu_results.json` (unless `--output` is used).

**Status:** Optional (can be very long on CPU).

---

## Optional tuning (not required for correctness)

- **`VLLM_CPU_KVCACHE_SPACE`:** cap KV reservation on long CPU jobs.
- **`CPU_TEST_PYTHON` / `GPU_VLLM_PYTHON`:** pick interpreters without stray `VLLM_*` names if vLLM warns about unknown env vars.
- **GPU:** **`VLLM_GPU_MEM_UTIL`**, **`vllm_gpu_test/README.md`**, **`setup_jetson_gpu_user_after_sudo.sh`** for device access.

---

## Checklist

- [x] Step 1 — CPU vLLM version string contains `cpu`
- [x] Step 2 — Smoke script completes with `device_config=cpu`
- [x] Step 3 — Transformers-only JSON written
- [x] Step 4 — Combined JSON with both arms for batch 1
- [x] vLLM **CPU** batch **500** + `max_num_seqs` **32 / 64 / 128 / 256** — `qwen3_1_7b_cpu_results_batch500_seq*_imu_thor.json` (**32 / 64** **2026-04-08**; **128 / 256** **2026-04-07**)
- [x] vLLM **GPU** batch **500** + `max_num_seqs` **32 / 64 / 128 / 256** — `qwen3_1_7b_gpu_results_batch500_seq*_imu_thor.json` (**32 / 64** **2026-04-08**)
- [ ] Step 5 — Full `run_qwen3_cpu_test.sh` (batches 1, 500, 1000, 2000) — optional

---

*Report: vLLM CPU + GPU batch-500 sweeps on imu-thor (CPU **32 / 64** and GPU **32 / 64** updated **2026-04-08**).*
