# Run GLM-4.6V-Flash in QEMU **System Mode** (9p share) on **1 Core**

This note documents the **recommended** system-mode workflow (Approach 4: **9p shared filesystem**) adapted for:

- **Model**: GLM-4.6V-Flash (`GGUF`)
- **QEMU mode**: `qemu-system-riscv64` (system mode)
- **CPU cores**: **1 vCPU** (`-smp cpus=1`)
- **Threads**: **1 thread** (`-t 1`) to match single-core and avoid multi-core psim limitations

---

## 1) Prerequisites (host)

These paths are expected to exist (they are the same ones used in `docs/qemu_all_in_one_guide.md`):

- **QEMU system binary**: `/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64`
- **Kernel**: `/projects2/linhu/VSI/linux-kernels/Image-6.12`
- **Rootfs (disk image)**: `/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2`
- **Shared repo path**: `/home/linhu/repo/iminn-tools` (shared into guest via 9p)
- **RISC-V llama.cpp binary**: `/home/linhu/repo/iminn-tools/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli`
- **GLM model file**: `/home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/GLM-4.6V-Flash-Q4_K_M.gguf`

---

## 2) Host: boot QEMU system-mode with 9p (**single core**)

Use the helper script:

```bash
/home/linhu/repo/iminn-tools/scripts/run_glm_system_9p_1cpu.sh
```

What it does:

- Boots QEMU system mode with:
  - `-machine virt`
  - `-cpu imicpu-v1`
  - `-smp cpus=1`
  - `-m 4G`
  - disk rootfs + kernel
  - 9p share mount tag: `shared`
  - `-nographic` (console in your terminal)

Wait until the guest boots to a login prompt.

---

## 3) Guest: mount the shared folder

Inside the guest OS:

```bash
mkdir -p /mnt/shared
mount -t 9p -o trans=virtio,version=9p2000.L shared /mnt/shared
ls /mnt/shared
```

Sanity checks (inside guest):

```bash
ls /mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli
ls /mnt/shared/dev_env/llama.cpp/models/GLM-4.6V-Flash-Q4_K_M.gguf
ls /mnt/shared/src/iminnt/resources/prompts/hello-world.txt
```

---

## 4) Guest: run GLM (text-only, single thread)

This runs a **minimal** GLM invocation (fastest smoke test):

- `-n 4` generate 4 tokens
- `-c 256` context size 256
- `-t 1` single thread

```bash
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p /mnt/shared/results

/mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /mnt/shared/dev_env/llama.cpp/models/GLM-4.6V-Flash-Q4_K_M.gguf \
  --seed 42 \
  -t 1 -ngl 0 -n 4 -c 256 \
  -no-cnv -st --no-warmup \
  --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt \
  >"/mnt/shared/results/glm_system_mode_${TS}.txt" 2>&1
```

---

## 4.1) About **simpoint/psim** in system mode (important limitation)

**You cannot run `simpoint` (psim simpointing) *inside* the QEMU system-mode guest.**

In this repo, **simpoint/psim is a host-side pipeline** driven by `iminnt` that invokes **Permafrost + Pilos + QEMU user-mode plugins** (e.g., `qemu-riscv64.so`, `libcosim.so`, Spike ISA sim). Those components run on the **host**, not inside the guest OS.

So in **system mode guest**, you can run **regular inference** (`llama-cli`) only — not the Permafrost/Pilos/simpoint pipeline.

### What to do instead (recommended)

- **If you want system-mode guest execution**: use §4/§5 (run `llama-cli` inside the guest).
- **If you want simpointing/psim**: run `iminnt ... simpoint` on the **host** (QEMU user-mode based), not in the guest.

---

## 4.2) Host: run GLM with **simpoint** (psim simpointing)

Run this on the **host** (not in the guest):

```bash
cd /home/linhu/repo/iminn-tools
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p results/glm_psim_outputs

iminnt -t llama_imi simpoint -d test_glm_4_6v_text_minimal \
  -o results/glm_simpoint_${TS} \
  2>&1 | tee results/glm_psim_outputs/glm_simpoint_${TS}.log
```

Notes:
- The correct subcommand is **`simpoint`** (not `simpointing`).
- `simpoint` **does not support** `-n/--no-roi` (that flag exists for `sim`).
- Results directory: `results/glm_simpoint_${TS}/`
- Terminal log: `results/glm_psim_outputs/glm_simpoint_${TS}.log`

---

## 4.3) System-mode QEMU + **SimPoint plugin** (BB profiling) (closest “simpoint in system mode”)

If what you want is **SimPoint basic-block vector generation from a `qemu-system-riscv64` run** (as your colleague described), you can do that by launching system-mode QEMU with the **`libsimpbbprof.so`** plugin enabled.

This is **not** the Permafrost/Pilos `iminnt simpoint` pipeline. It is:

1. `qemu-system-riscv64` + `libsimpbbprof.so` → generates `BB.log` + `qemu_simpoint.bb` (on the host)
2. Run `dev_env/simpoint/bin/simpoint` on the generated `.bb` file (on the host)

### Step A: Host boot (system mode + plugin) — copy/paste

Run the system-mode QEMU launcher that enables the BB profiler plugin:

```bash
cd /home/linhu/repo/iminn-tools
./scripts/run_glm_system_9p_1cpu_simpoint.sh
```

It will boot the guest and create a host output directory like:

- `results/qemu_sys_simpoint_<TS>/BB.log`
- `results/qemu_sys_simpoint_<TS>/qemu_simpoint.bb`

Leave QEMU running while you do Step B inside the guest.

### Step B: Guest run GLM (same as §4) — copy/paste

**Important**: in system-mode, you do **NOT** run `iminnt sim` or `iminnt simpoint` inside the guest.
You run the workload normally (below). The **SimPoint BB profiling** is happening in the **host QEMU command** (Step A) via `-plugin ...libsimpbbprof.so...`.

Inside the guest (Linux shell), mount the host repo via 9p and run a minimal GLM command:

```bash
# mount host repo as /mnt/shared
mkdir -p /mnt/shared
mount -t 9p -o trans=virtio,version=9p2000.L shared /mnt/shared

# run GLM (minimal, single thread) and also save output to the shared host folder
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p /mnt/shared/results

/mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /mnt/shared/dev_env/llama.cpp/models/GLM-4.6V-Flash-Q4_K_M.gguf \
  --seed 42 \
  -t 1 -ngl 0 -n 4 -c 256 \
  -no-cnv -st --no-warmup \
  --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt \
  >"/mnt/shared/results/glm_system_mode_${TS}.txt" 2>&1
```

### Step C: Host post-process SimPoints — copy/paste

This step is where you run the **SimPoint clustering** tool on the generated `qemu_simpoint.bb`.

Stop QEMU (Ctrl+C, or QEMU monitor `Ctrl+A` then `C`, `quit`). Then on the host, run SimPoint clustering on the latest generated `.bb` file:

```bash
cd /home/linhu/repo/iminn-tools
OUT_DIR="$(ls -dt results/qemu_sys_simpoint_* | head -1)"
echo "Using OUT_DIR=${OUT_DIR}"

./dev_env/simpoint/bin/simpoint -maxK 30 \
  -saveSimpoints "${OUT_DIR}/simpoints" \
  -saveSimpointWeights "${OUT_DIR}/weights" \
  -loadFVFile "${OUT_DIR}/qemu_simpoint.bb"
```

⚠️ Note: This BB vector may include **kernel/boot activity**. To reduce noise, make your rootfs auto-run the workload immediately after boot (Approach 1/3 in `qemu_all_in_one_guide.md`) or accept that the first interval(s) may be dominated by boot.

### Where is the output?

Because `/mnt/shared` maps to your host repo, the output file is also visible on the host:

```bash
/home/linhu/repo/iminn-tools/results/glm_system_mode_${TS}.txt
```

To watch output live (in guest):

```bash
tail -f /mnt/shared/results/glm_system_mode_${TS}.txt
```

---

## 5) Optional: run in background in the guest

```bash
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p /mnt/shared/results

nohup /mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /mnt/shared/dev_env/llama.cpp/models/GLM-4.6V-Flash-Q4_K_M.gguf \
  --seed 42 \
  -t 1 -ngl 0 -n 4 -c 256 \
  -no-cnv -st --no-warmup \
  --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt \
  >"/mnt/shared/results/glm_system_mode_${TS}.txt" 2>&1 &
```

---

## 6) Exiting system-mode QEMU

In the terminal where QEMU is running:

- **Preferred**: `Ctrl+C` (may take a moment)
- **QEMU monitor**: `Ctrl+A` then `C`, then type:

```text
quit
```

You can also shut down from inside the guest:

```bash
poweroff
```

---

## 7) Common mistakes / troubleshooting

- **9p not mounted**: you must run the `mount -t 9p ...` command inside the guest before trying `/mnt/shared/...`.
- **Wrong binary**: inside the guest you must run the **RISC-V** `llama-cli` under `/mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli` (not the x86 build).
- **No output file on host**: ensure you redirect output to `/mnt/shared/results/...` (not to `/results` inside the guest rootfs).
- **Single core requirement**: keep `-smp cpus=1` and run `llama-cli -t 1` for consistent single-thread behavior.

