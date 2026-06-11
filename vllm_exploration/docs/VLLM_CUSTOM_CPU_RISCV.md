# Running vLLM on a Custom or RISC-V CPU (CPU-Only)

**Scope**: This note covers **CPU-only** inference on **non-x86** Linux targets, with emphasis on **RISC-V (`riscv64`)** and chips that expose **custom or recent ISA extensions** (for example **RVV** vector, vendor matrix, or other ratified profiles). It does **not** describe running NVIDIA CUDA GPU kernels on RISC-V; production vLLM throughput still assumes accelerators elsewhere.

**Audience**: Integrators bringing up **new silicon** or **embedded/datacenter RISC-V** who want vLLM’s **scheduling, batching, and APIs** on top of PyTorch CPU execution.

**PyTorch on RISC-V Linux (current practice)**: **Yes — PyTorch models can run on `riscv64` Linux today** for many workloads, using **vendor or community PyTorch builds**, **distribution packages**, or **source builds**, plus upstream **RISC-V CI and roadmap** work. What still differs by platform is **how you obtain `torch`** (not every generic `pip install` matrix is tier‑1 yet), **full operator coverage** for exotic models, and **performance** relative to x86 or GPU—these are integration and tuning problems, not “PyTorch cannot run at all.”

---

## Table of contents

1. [Reality check](#reality-check)
2. [Architecture you are actually porting](#architecture-you-are-actually-porting)
3. [Prerequisites](#prerequisites)
4. [Step 1: PyTorch on `riscv64`](#step-1-pytorch-on-riscv64)
5. [From working PyTorch to vLLM and your model](#from-working-pytorch-to-vllm-and-your-model)
6. [Step 2: vLLM CPU backend](#step-2-vllm-cpu-backend)
7. [Using new instruction extensions (RVV, etc.)](#using-new-instruction-extensions-rvv-etc)
8. [Smoke tests](#smoke-tests)
9. [Common failure modes](#common-failure-modes)
10. [When not to use vLLM on CPU](#when-not-to-use-vllm-on-cpu)
11. [References](#references)

---

## Reality check

| Expectation | Fact |
|-------------|------|
| “Same as x86 CUDA vLLM” | **No.** CUDA paths do not apply to a generic RISC-V CPU. |
| “`pip install torch` everywhere” | **Sometimes** on supported boards/distros; otherwise **vendor wheels**, **distro packages**, or **PyTorch/vLLM built from source** for your exact glibc/toolchain/ISA. |
| “PagedAttention CUDA on CPU” | PagedAttention’s **GPU** kernels are irrelevant; the **CPU** backend uses different execution and memory paths. |
| Throughput | Expect **orders of magnitude** below GPU serving for the same model size. |

vLLM on CPU is **valid** for bring-up, correctness checks, and **small** models or **low** concurrency—not for datacenter-scale LLM serving without an accelerator.

---

## Architecture you are actually porting

Rough dependency chain:

```text
Your application
    → vLLM (Python: engine, scheduler, HTTP API)
        → PyTorch (CPU ATen ops, autograd where used)
            → Eigen / BLAS / oneDNN-style paths (version-dependent)
                → libc + your CPU ISA
```

**Leveraging a new RISC-V extension** therefore lands mostly in **PyTorch and native math libraries**, not in a single vLLM configuration flag. vLLM benefits indirectly when `torch` matrix and attention ops run faster on your CPU.

---

## Prerequisites

- **OS**: Linux on `riscv64` is the realistic target (glibc-based distributions are most tested in open source).
- **Python**: Use the range supported by your vLLM version (commonly **3.10–3.13**; check upstream release notes).
- **Toolchain**: Recent **GCC** or **Clang** that supports your chip’s **`-march=`** / **`-mcpu=`** string for the extensions you need (for example vector).
- **RAM**: Plan for **model weights + KV cache + framework overhead**; CPU-only setups often use **quantized** or **small** models first.
- **Disk**: Source builds of PyTorch and vLLM need **tens of GB** for build trees and ccache is strongly recommended.

---

## Step 1: PyTorch on `riscv64`

This is the **main gate**: if `import torch` fails or core ops crash, vLLM cannot run. Functional **PyTorch on RISC-V Linux** is now a **routine bring-up goal** for many silicon and distro teams; confirm your chosen build against the **model ops** you need (matmul-heavy models are the easiest path).

**Options (pick what matches your product timeline):**

1. **Vendor or community PyTorch builds** for your board (fastest if ABI and Python versions match).
2. **Build PyTorch from source** on the RISC-V machine (or a reproducible cross-build flow your distro supports), with:
   - CPU-only configuration (no CUDA expectation on pure RISC-V bring-up).
   - **`-march=` / `-mcpu=`** aligned with your silicon so generated code can use **RVV** or other enabled extensions where the PyTorch build supports them.
3. **Track upstream RISC-V work**: PyTorch has active discussion and roadmap items for **RISC-V CI and performance** (micro-kernels, vector/matrix extensions, `torch.compile` over time). See the References section.

**Sanity check before touching vLLM:**

```bash
python3 -c "import torch; x=torch.randn(512,512); y=x@x; print(y.sum().item())"
```

---

## From working PyTorch to vLLM and your model

If **PyTorch already compiles and runs** on your RISC-V CPU, the **next layer** is not “recompile the model again for vLLM” in a special format: vLLM loads the same **weights** (for example Hugging Face / Safetensors) and drives them through **`torch` on CPU**. Your job is to add the **vLLM CPU package** (and its Python/native deps) so scheduling, batching, and optional OpenAI-style serving sit on top of that PyTorch.

**Ordered checklist:**

1. **Pin versions**  
   Open [vLLM `requirements/cpu.txt`](https://github.com/vllm-project/vllm/blob/main/requirements/cpu.txt) for the **vLLM tag** you want. Check the line that applies to **`riscv64`** (or the generic non–`x86_64` / non-`s390x` case). Your installed **`torch`** must satisfy that pin (same major/minor as upstream expects for CPU vLLM on that arch).

2. **Use a clean virtualenv**  
   `python3 -m venv vllm-riscv && source vllm-riscv/bin/activate`  
   Install **your working PyTorch first** (or keep the system/site-packages path you already validated—just avoid mixing two `torch` installs).

3. **Install vLLM’s CPU-only stack**  
   Follow [CPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/) for that tag: either a **`vllm-cpu`** wheel if one exists for your ABI, or a **from-source** build. For source builds, set **`VLLM_TARGET_DEVICE=cpu`** (and any other flags the docs list for CPU) so CMake does not assume CUDA.

4. **Let vLLM use your existing PyTorch**  
   Do not install a second **`torch`** from a different channel unless `cpu.txt` requires it. Mismatched `torch` + vLLM is a common source of import or ABI errors.

5. **Model loader**  
   If the model comes from Hugging Face, install **`transformers`**, **`tokenizers`**, and any extras the model card lists, in versions compatible with that vLLM release.

6. **First run: tiny and local**  
   Use a **small** model, low **`max_model_len`**, and **batch size 1** (see [Smoke tests](#smoke-tests)). Prefer `LLM(...)` in a short script or `vllm serve` with minimal concurrency until memory and latency look sane.

7. **If the vLLM build fails on native code**  
   Capture the **CMake / C++ error**. Some optional accelerators (x86-only intrinsics, CUDA-only paths) must be **disabled** on RISC-V; upstream or your vendor may carry small patches. The **model weights** are usually not the problem once `torch` runs the same graph outside vLLM.

**Important distinction:** “PyTorch model compiles” might mean **TorchScript / `torch.compile` / export** on your machine. vLLM’s default path for many models is still **eager PyTorch + vLLM modules**, not a separate “vLLM IR.” You do **not** need a second compilation step for the model **unless** you choose a backend that explicitly requires one (check vLLM docs for your model family).

---

## Step 2: vLLM CPU backend

Upstream documents a **CPU** installation path (package names and pins evolve; always read the version you deploy).

**General approach:**

1. Read the **CPU installation** section of the vLLM docs for your tag (for example [CPU — vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)).
2. Align **PyTorch** with what **`requirements/cpu.txt`** specifies for **non-x86** arches (upstream splits dependencies by architecture; **`riscv64`** may be listed alongside other CPU-only tiers).
3. Install **CPU-only vLLM** (`vllm-cpu` or a **from-source** build with **`VLLM_TARGET_DEVICE=cpu`**, per current docs). Building from source is common on exotic CPUs when wheels are unavailable.

**Environment hints (exact names change by release):**

- Prefer **`VLLM_TARGET_DEVICE=cpu`** when building or running if your docs say so.
- Avoid mixing a **CUDA** PyTorch wheel with a **CPU** vLLM build on a machine that has no usable CUDA stack.

After install:

```bash
python3 -c "import vllm; print(vllm.__version__)"
```

---

## Using new instruction extensions (RVV, etc.)

vLLM does not expose a per-extension toggle like `USE_RVV=1` for the whole stack. Instead:

| What you control | Effect |
|------------------|--------|
| **Compiler flags** for PyTorch, BLAS, and vLLM native extensions | Enables legal use of RVV (and friends) in compiled kernels and autovectorized loops. |
| **PyTorch build options** | Determines which CPU backend paths exist (e.g. oneDNN / vectorized ATen); this dominates matmul and many attention building blocks on CPU. |
| **Math library** (OpenBLIS, BLIS, vendor BLAS) | If built for your ISA, heavy linear algebra speeds up for all frameworks using it. |
| **vLLM optional native code** | Must **compile for `riscv64`**. If any extension assumes **x86-only intrinsics** (AVX, etc.), you need **portable fallbacks** or to disable that feature on RISC-V until patched upstream. |

**Practical order of work:**

1. Tune **PyTorch + BLAS** for your ISA (biggest win).
2. Build **vLLM** against that PyTorch.
3. Profile a tiny model; if specific ops are missing or slow, open upstream issues or add **RISC-V-specific** implementations in the **library that owns the op** (often PyTorch, not vLLM’s Python layer).

---

## Smoke tests

1. **Imports**: `torch`, `vllm`, `transformers` (if you load HF models).
2. **Tiny generation**: a **very small** model or minimal config to limit RAM and compile time.
3. **Single request then batch**: confirm scheduler and memory paths work before scaling batch size.

Use the smallest **public** weight shard you can (or a local stub) so failures isolate **port** vs **model** issues.

---

## Common failure modes

- **Illegal instruction**: Binary built for a **baseline** `riscv64` profile but your CPU requires **Zicsr/Zifencei** or you enabled **RVV** in `-march=` while the core does not implement it—or the opposite (no RVV in build while a dependency assumes it). Align **`-march=`** with the **minimum** ISA your fleet guarantees.
- **Missing symbols / wrong glibc**: Wheel built on another distro; prefer **build on target** or a reproducible rootfs.
- **x86-only dependency**: Optional features (certain codecs, AVX-only helpers) pulled in; disable that feature or patch.
- **OOM**: CPU path still allocates KV cache; reduce **`max_model_len`**, batch size, or use quantization.

---

## When not to use vLLM on CPU

- You need **high tokens/sec** on large models without a GPU or NPU: use an **accelerator stack** or a **lighter** inference engine (for example **llama.cpp** with RVV-tuned paths is a common research path on RISC-V).
- Your accelerator is **non-NVIDIA**: you need a **supported backend** in PyTorch and vLLM for that device; that is a separate integration from “RISC-V CPU only.”

---

## References

- vLLM CPU installation (current): [https://docs.vllm.ai/en/latest/getting_started/installation/cpu/](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)
- vLLM CPU requirements (pins by arch): [https://github.com/vllm-project/vllm/blob/main/requirements/cpu.txt](https://github.com/vllm-project/vllm/blob/main/requirements/cpu.txt)
- PyTorch RISC-V roadmap / discussion (upstream): [https://github.com/pytorch/pytorch/issues/171659](https://github.com/pytorch/pytorch/issues/171659) (RFC and related issues; titles and scope may evolve)

---

**Document**: companion to `VLLM_DEEP_DIVE.md`  
**Last updated**: May 4, 2026  
**Location**: `/home/linhu/projects/vllm_exploration/docs/VLLM_CUSTOM_CPU_RISCV.md`
