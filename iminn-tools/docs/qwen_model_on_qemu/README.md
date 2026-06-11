# Run Qwen3-Coder-30B-A3B (MXFP4) on RISC-V via QEMU

**Date:** June 9, 2026  
**Scope:** Run `Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf` through the `llama_imi` target using QEMU user mode on iminn-tools.

---

## 1. Overview

This guide documents how to run a large Qwen3 MoE model in GGUF format on the **I-Machines RISC-V CPU path** emulated by custom QEMU.

| Item | Value |
|------|-------|
| Model | `Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf` |
| Model path (local) | `dev_env/llama.cpp/models/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf` |
| Model path (NFS) | `/projects3/workloads/ai/models/mxfp4/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf` |
| Model size | ~16 GB on disk |
| Quantization | MXFP4 |
| Architecture | Qwen3 MoE (30B total, ~3B active) |
| iminn-tools target | **`llama_imi_bench`** (use this for inference; see §8) |
| Execution | QEMU user mode (`qemu-riscv64 -cpu imicpu-v1`) |
| Binary | `dev_env/llama.cpp/llamacpp-imi-bench-install/bin/llama-cli` |

### Execution flow

```text
Host (aarch64/x86_64)
  └─ qemu-riscv64 -cpu imicpu-v1
       └─ llama-cli  (RISC-V binary)
            └─ Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf
```

QEMU user mode uses the **host filesystem** directly. Copy the model to local disk for faster mmap/I/O:

```bash
cp /projects3/workloads/ai/models/mxfp4/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf \
   dev_env/llama.cpp/models/
```

---

## 2. Prerequisites

### 2.1 iminn-tools CLI

Install into a virtual environment (required on Ubuntu/Debian due to PEP 668):

```bash
cd /home/linhu/projects/iminn-tools
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
iminnt --help
```

Activate the venv in every new terminal:

```bash
source /home/linhu/projects/iminn-tools/.venv/bin/activate
```

### 2.2 Built artifacts

These must exist under `dev_env/`:

| Component | Path |
|-----------|------|
| RISC-V llama-cli | `dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli` |
| QEMU user binary | `dev_env/csqemu-v9/install-local/bin/qemu-riscv64` |

Quick check:

```bash
file dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli
# Expected: UCB RISC-V

file dev_env/csqemu-v9/install-local/bin/qemu-riscv64
# Expected: ELF for YOUR host arch (aarch64 on imu-thor, x86-64 on imu008)
```

**Important (ARM hosts):** If `dev_env/` was copied from an x86 machine, `qemu-riscv64` will be the wrong architecture and runs will fail with:

```text
qemu-riscv64: 1: Syntax error: "(" unexpected
```

Rebuild QEMU on the local machine:

```bash
source .venv/bin/activate
iminnt -t qemu build
```

The RISC-V `llama-cli` binary does **not** need to be rebuilt when only the host architecture changes.

### 2.3 Model file

```bash
ls -lh /projects3/workloads/ai/models/mxfp4/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf
```

Expected: ~16 GB file on NFS (`/projects3`).

### 2.4 Memory

| Resource | Requirement |
|----------|-------------|
| Disk | ~16 GB (model file) |
| RAM | ~20–30 GB+ (model mmap + KV cache + compute buffers) |
| Recommended free RAM | ≥ 32 GB before starting |

Check host memory:

```bash
free -h
```

### 2.5 llama.cpp support

The I-Machines llama.cpp fork in `dev_env/llama.cpp/` includes:

- MXFP4 quantization kernels (`GGML_TYPE_MXFP4`)
- Qwen3 / MoE support
- IMI CPU backend (`GGML_CPU_IMI=ON` for `llama_imi` builds)

No extra build flags are needed beyond a standard `llama_imi` build.

---

## 3. Run via iminnt (recommended)

Run from the repo root. Use **`llama_imi_bench`** and the **local model copy**.

**Model path variable** (used below):

```bash
MODEL=dev_env/llama.cpp/models/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf
```

### 3.1 Recommended inference command (local model)

Minimal generation settings for QEMU (`-n 1 -c 256`). Model load still takes several minutes.

```bash
cd /home/linhu/projects/iminn-tools
source .venv/bin/activate

iminnt -t llama_imi_bench run \
  -b llamacpp-imi-bench-install/bin/llama-cli \
  -a -m dev_env/llama.cpp/models/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf \
     --cpu-moe \
     --file src/iminnt/resources/prompts/code-hello-world.txt \
     -n 1 -t 1 -ngl 0 -c 256 \
     --seed 42 -no-cnv -st --no-warmup
```

Prompt input: `src/iminnt/resources/prompts/code-hello-world.txt` → `say hello world with python`

### 3.2 Longer generation (slow under QEMU)

Only use after §3.1 succeeds. `-n 64 -c 8192` can take **hours** per token under QEMU.

```bash
iminnt -t llama_imi_bench run \
  -b llamacpp-imi-bench-install/bin/llama-cli \
  -a -m dev_env/llama.cpp/models/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf \
     --cpu-moe \
     --file src/iminnt/resources/prompts/code-hello-world.txt \
     -n 64 -t 1 -ngl 0 -c 8192 \
     --seed 42 -no-cnv -st --no-warmup
```

### 3.3 Inline prompt (quote multi-word text)

```bash
iminnt -t llama_imi_bench run \
  -b llamacpp-imi-bench-install/bin/llama-cli \
  -a -m dev_env/llama.cpp/models/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf \
     --cpu-moe \
     -p "Write a hello world function in Python." \
     -n 1 -t 1 -ngl 0 -c 256 \
     --seed 42 -no-cnv -st --no-warmup
```

### 3.4 Confirm QEMU is used

Look for this line at the start of the log:

```text
Running bash command: .../qemu-riscv64 -E IMI_ROI_SIM="1" -cpu imicpu-v1 .../llama-cli -m /projects3/...
```

Inside model output, confirm RISC-V / IMI backend:

```text
system_info: ... | CPU : RISCV_V = 1 | IMI = 1 | ...
```

---

## 4. Direct QEMU command (without iminnt)

```bash
/home/linhu/projects/iminn-tools/dev_env/csqemu-v9/install-local/bin/qemu-riscv64 \
  -E IMI_ROI_SIM="1" -cpu imicpu-v1 \
  /home/linhu/projects/iminn-tools/dev_env/llama.cpp/llamacpp-imi-bench-install/bin/llama-cli \
  -m /home/linhu/projects/iminn-tools/dev_env/llama.cpp/models/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf \
  --cpu-moe \
  --file /home/linhu/projects/iminn-tools/src/iminnt/resources/prompts/code-hello-world.txt \
  -n 1 -t 1 -ngl 0 -c 256 \
  --seed 42 -no-cnv -st --no-warmup
```

---

## 5. Flag reference

| Flag | Value | Purpose |
|------|-------|---------|
| `-m` | Model GGUF path | Load the Qwen3 weights |
| `--cpu-moe` | (flag) | Route MoE expert ops to CPU (required for this MoE model) |
| `-p` | Text prompt | Input for generation |
| `--file` | Prompt file | Alternative to `-p` |
| `-n` | e.g. `8`, `64` | Max tokens to generate |
| `-c` | e.g. `2048`, `8192` | Context window size |
| `-t` | `1` (start here) | CPU threads inside llama.cpp |
| `-ngl` | `0` | GPU layers (0 = CPU only) |
| `--seed` | e.g. `42` | Reproducible sampling |
| `-st` | (flag) | Simple text mode (matches iminn-tools presets) |
| `--no-warmup` | (flag) | Skip warmup pass |

### MoE note

`Qwen3-Coder-30B-A3B` is a **Mixture-of-Experts** model. Always include `--cpu-moe` for CPU-only inference. Without it, MoE routing may fail or fall back incorrectly.

### Instruct / chat mode

For chat-style formatting, omit `-st` and let llama.cpp apply the model's chat template:

```bash
iminnt -t llama_imi run \
  -b llamacpp-imi-install/bin/llama-cli \
  -a -m /projects3/workloads/ai/models/mxfp4/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf \
     --cpu-moe \
     -p "Write a hello world function in Python." \
     -n 64 -t 1 -ngl 0 -c 8192 --seed 42
```

---

## 6. Comparison with the default smoke test

The built-in preset `test_q4_0_stories` is a tiny model for fast validation:

```bash
iminnt -t llama_imi run -d test_q4_0_stories
```

| | `test_q4_0_stories` | Qwen3-Coder MXFP4 |
|--|---------------------|-------------------|
| Model | `stories15M-q4_0.gguf` (~19 MB) | `Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf` (~16 GB) |
| Params | ~24M | 30B MoE (~3B active) |
| Quant | Q4_0 | MXFP4 |
| Load time (QEMU) | ~1–2 min | Several minutes |
| Invocation | `-d test_q4_0_stories` | Custom `-b` / `-a` (see above) |

Use `test_q4_0_stories` to verify QEMU + iminn-tools before attempting the large Qwen model.

---

## 7. Performance expectations

QEMU user mode emulates every RISC-V instruction on the host CPU. Expect:

- **Slow model load** (reading and mapping 16 GB through emulation)
- **Low tokens/sec** compared to native x86 or hardware RISC-V
- **Long wall-clock time** for even small `-n` values

This path is intended for **correctness validation**, **IMI kernel testing**, and **simulation prep** — not production inference speed.

For faster functional testing of the same GGUF on native hardware, use `llama_x86` on an x86 host or a native aarch64 llama.cpp build on ARM (see `docs/qemu_all_in_one_guide.md`).

---

## 8. Troubleshooting

### `ggml-ref/ops.cpp: fatal error` / ptrace message

**Cause:** `llama_imi` builds with `GGML_REF=ON`. The REF backend aborts on unsupported MXFP4/MoE ops during first token generation.

**Fix:** Use **`llama_imi_bench`** and `llamacpp-imi-bench-install/bin/llama-cli` (REF disabled). Confirm log shows `IMI = 1` without `REF`.

### No tokens after 30+ minutes

**Cause:** 30B MoE under QEMU is extremely slow; `-n 64 -c 8192` is impractical. Process may still be at 100% CPU doing prefill.

**Fix:**

- Use `-n 1 -c 256`
- Use local model copy (not NFS)
- Monitor with: `ps -p <pid> -o etime,pcpu,rss`

### `error: invalid argument: a` (multi-word prompt)

**Cause:** When using `-a` with `-p`, a prompt containing spaces (e.g. `Write a hello world...`) was previously split into separate tokens. The word `a` was passed to `llama-cli` as a stray argument.

**Fix (code):** Recent `core.py` keeps `-a` arguments as a list instead of re-splitting on whitespace. Re-run with quotes (recommended):

```bash
iminnt -t llama_imi run \
  -b llamacpp-imi-install/bin/llama-cli \
  -a -m /projects3/workloads/ai/models/mxfp4/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf \
     --cpu-moe \
     -p "Write a hello world function in Python." \
     -n 64 -t 1 -ngl 0 -c 8192 \
     --seed 42 -st --no-warmup
```

Use quotes around multi-word `-p` values when passing via `-a`.

**Alternative:** Use `--file` instead of `-p` (no quoting issues):

```bash
iminnt -t llama_imi run \
  -b llamacpp-imi-install/bin/llama-cli \
  -a -m /projects3/workloads/ai/models/mxfp4/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf \
     --cpu-moe \
     --file src/iminnt/resources/prompts/code-hello-world.txt \
     -n 64 -t 1 -ngl 0 -c 8192 \
     --seed 42 -st --no-warmup
```

### `qemu-riscv64: Syntax error: "(" unexpected`

**Cause:** QEMU binary built for wrong host architecture (e.g. x86 binary on aarch64).

**Fix:**

```bash
iminnt -t qemu build
file dev_env/csqemu-v9/install-local/bin/qemu-riscv64
```

### `cannot execute binary file: Exec format error` on llama-cli

**Cause:** Trying to run the RISC-V binary directly on the host without QEMU.

**Fix:** Always use `iminnt -t llama_imi run` or wrap with `qemu-riscv64`.

### Out of memory / killed

**Cause:** 16 GB model + KV cache exceeds available RAM.

**Fix:**

- Reduce context: `-c 2048` or `-c 4096`
- Reduce generation: `-n 8`
- Ensure no other large jobs are running: `free -h`

### Model load errors (unsupported tensor / quant type)

**Cause:** llama.cpp build too old for MXFP4 or Qwen3.

**Fix:**

```bash
iminnt -t llama_imi pull
iminnt -t llama_imi build
```

### MoE-related failures

**Cause:** Missing `--cpu-moe` for MoE models.

**Fix:** Add `--cpu-moe` to the `-a` arguments.

### `iminnt: command not found`

**Cause:** Virtual environment not activated.

**Fix:**

```bash
source /home/linhu/projects/iminn-tools/.venv/bin/activate
```

---

## 9. Optional: add a named preset

To avoid typing the full command each time, add an entry to `default_runs` in `src/iminnt/llamacpp.py`:

```python
"test_qwen3_coder_mxfp4": {"bin": f"{self.install_dir}/bin/llama-cli",
    "args": f"-m /projects3/workloads/ai/models/mxfp4/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf \
                --cpu-moe \
                -p \"Write a hello world function in Python.\" \
                --seed 42 \
                -t 1 -ngl 0 -n 64 -c 8192 \
                -st --no-warmup"},
```

Then run:

```bash
iminnt -t llama_imi run -d test_qwen3_coder_mxfp4
```

Re-install after editing (editable install picks up changes automatically):

```bash
# No reinstall needed with pip install -e .
```

---

## 10. Related docs

| Document | Topic |
|----------|-------|
| `docs/qemu_all_in_one_guide.md` | QEMU user vs system mode, multi-core |
| `docs/glm_qemu_system_mode_1cpu.md` | System-mode example for GLM model |
| `docs/llama_cpp_model_to_cpu_inference.md` | General llama.cpp CPU inference |
| `README.md` (repo root) | iminn-tools targets and commands |
| `CLAUDE.md` | Developer reference for iminn-tools |

---

## 11. Quick reference

```bash
cd /home/linhu/projects/iminn-tools
source .venv/bin/activate

# Verify QEMU with tiny model (~1-2 min)
iminnt -t llama_imi run -d test_q4_0_stories

# Qwen3-Coder MXFP4 on RISC-V via QEMU (local model, minimal gen)
iminnt -t llama_imi_bench run \
  -b llamacpp-imi-bench-install/bin/llama-cli \
  -a -m dev_env/llama.cpp/models/Qwen3-Coder-30B-A3B-Instruct-mxfp4_only.gguf \
     --cpu-moe \
     --file src/iminnt/resources/prompts/code-hello-world.txt \
     -n 1 -t 1 -ngl 0 -c 256 \
     --seed 42 -no-cnv -st --no-warmup
```
