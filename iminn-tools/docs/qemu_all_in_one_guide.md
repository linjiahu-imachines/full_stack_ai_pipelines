## QEMU All‑In‑One Guide for iminn-tools

**Date:** December 18, 2025  
**Scope:** Fundamentals, configuration, multi‑core, and running `llama.cpp` tests in QEMU user *and* system mode (including 9p and initrd approaches).

---

## 1. Big Picture: User Mode vs System Mode

- **User mode (`qemu-riscv64`)**
  - Emulates a **single RISC‑V process**.
  - Intercepts RISC‑V **system calls** and translates them to host Linux syscalls.
  - Uses the **host filesystem** directly.
  - **No guest OS**, no boot process, **no multi‑core** (single emulated CPU), but supports multi‑threading (threads are scheduled by the host OS on the single emulated core).
  - Fast startup, lower overhead → ideal for most `iminnt run` use cases.
  - In our case, it runs the **RISC‑V‑compiled `llama-cli` binary directly on the host**, with QEMU translating its RISC‑V instructions and syscalls to x86_64 + host Linux.

- **System mode (`qemu-system-riscv64`)**
  - Emulates a **full RISC‑V machine** (CPU(s), RAM, devices, interrupt controller).
  - Boots a **guest Linux kernel** and runs processes on top of it.
  - Root filesystem is provided by a **disk image** (ext2/ext4) or **initrd (cpio ramfs)**.
  - Supports **multi‑core (SMP)** via `-smp`, full OS behavior, isolation, and device emulation.
  - Slower and more complex, but required for true multi‑core experiments and OS‑level testing.
  - In our case, it runs the **same RISC‑V‑compiled `llama-cli` binary inside a RISC‑V Linux guest**, exactly as it would run on real RISC‑V hardware, with QEMU emulating the CPUs and devices underneath.

### 1.1 Conceptual Model

- **User mode**:
  - RISC‑V `llama-cli` (and other apps) are started **from the host shell** via `qemu-riscv64`.
  - QEMU loads the RISC‑V ELF, translates its instructions, and forwards its syscalls to the **host** kernel.
  - Guest app → QEMU translates instructions + syscalls → host kernel + host filesystem.
  - Only one emulated CPU; `-smp` is **not** available.

- **System mode**:
  - QEMU first boots a **RISC‑V Linux kernel and rootfs**; `llama-cli` is then started **inside the guest shell**.
  - From `llama-cli`’s point of view, it is running on a real multi‑core RISC‑V machine.
  - Guest `llama-cli` → guest Linux kernel handles syscalls → QEMU emulates hardware (CPUs, disk, 9p) → host kernel + host files/devices.
  - Multiple emulated CPUs available via `-smp N`.

---

## 2. QEMU in iminn-tools

### 2.1 Binaries and Paths

- **User mode binary** (used today by `iminnt run`):

```text
dev_env/csqemu-v9/install-local/bin/qemu-riscv64
```

- **System mode binary** (already built and working):

```text
dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64
```

The correct constants (if/when wired into code) should be:

```python
QEMU_SYS_DIR = QEMU_BASE_DIR / "install-sys-local"
QEMU_SYS_BIN = QEMU_SYS_DIR / "bin" / "qemu-system-riscv64"
```

### 2.2 Current Effective Setup

- **Kernels & rootfs you actually use:**
  - Kernel: `/projects2/linhu/VSI/linux-kernels/Image-6.12`
  - Rootfs disk: `/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2`
- **Additional kernel with initrd support:**
  - Kernel (with `CONFIG_BLK_DEV_INITRD` etc.):  
    `/projects2/linhu/VSI/linux-kernels-v2/Image-6.12`

Summary:

- For **disk‑image based system mode** (no initrd), use `Image-6.12` + `rootfs-6.12.ext2`.
- For **initrd‑based experiments** (Approach 2), use the **v2** kernel with initrd support.

---

## 3. System Mode Building Blocks

### 3.1 QEMU System Binary

Verify:

```bash
/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64 -version
/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64 -cpu help | grep -i imi
```

You should see:

- Version: around **QEMU 9.0.2**
- CPU model `imicpu-v1` listed.

### 3.2 Kernel and Root Filesystem

Two main ways to provide a root filesystem:

- **Disk image (ext2/ext4)** – persistent, recommended for the working system:
  - `-drive file=...rootfs-6.12.ext2,format=raw,id=hd0`
  - `-device virtio-blk-device,drive=hd0`
  - Kernel cmdline: `root=/dev/vda ro console=ttyS0`

- **Initrd / initramfs (cpio)** – RAM‑only, great for scripted tests and no‑sudo workflows:
  - `-initrd /path/to/initrd.cpio`
  - Kernel cmdline: typically `console=ttyS0` (optionally `rdinit=/init` or `init=/init`).
  - Root filesystem is the cpio content, lives in RAM and is lost on poweroff.

For your existing `Image-6.12` from `/projects2/linhu/VSI/linux-kernels`, **initrd is not supported** (no `CONFIG_BLK_DEV_INITRD`), so you must boot from a disk image. The `linux-kernels-v2` `Image-6.12` **does** support initrd and is what Approach 2 uses.

### 3.3 9p Shared Filesystem (virtio‑9p-pci)

#### What is 9p?

**9p** (also called **Plan 9 Filesystem Protocol** or **virtfs**) is a network filesystem protocol that allows a guest OS to access host directories as if they were local filesystems. It's similar to:
- **VMware/VirtualBox "shared folders"** – host directories accessible from the guest
- **NFS (Network File System)** – but designed for virtualization, not network sharing
- **Samba/CIFS** – but simpler and optimized for QEMU

#### Why do we need it?

**The Problem:**
- In QEMU system mode, the guest OS has its own isolated filesystem (the rootfs disk image).
- Files on the host are **not automatically visible** in the guest.
- To run `llama-cli` in the guest, you would normally need to:
  1. Copy the binary, model, and prompts into the rootfs image (requires `sudo` and is slow)
  2. Or rebuild the entire rootfs every time you change files (very slow)

**The Solution (9p):**
- 9p lets you **share a host directory directly** with the guest.
- No copying needed – files are accessed in real-time.
- Changes in the guest are immediately visible on the host (and vice versa).
- No `sudo` required – QEMU handles the sharing.

**How it works:**

1. **QEMU side** (host command):
   ```bash
   -fsdev local,id=shared,path=/home/linhu/repo/iminn-tools,security_model=none \
   -device virtio-9p-pci,fsdev=shared,mount_tag=shared
   ```
   - This tells QEMU: "Share the host directory `/home/linhu/repo/iminn-tools` as a virtual device named `shared`"

2. **Guest side** (inside the guest OS):
   ```bash
   mkdir -p /mnt/shared
   mount -t 9p -o trans=virtio,version=9p2000.L shared /mnt/shared
   ```
   - This tells the guest OS: "Mount the virtual device `shared` to `/mnt/shared` using the 9p filesystem"

3. **Result:**
   - `/mnt/shared` in the guest = `/home/linhu/repo/iminn-tools` on the host
   - Files are the same – no copying, no delay
   - Test results saved to `/mnt/shared/results/` appear immediately on the host

**Benefits:**
- ✅ No `sudo` required
- ✅ No file copying needed
- ✅ Real-time access to host files
- ✅ Test results automatically saved to host
- ✅ Can run binaries directly from shared directory

**Kernel requirement:** 9p filesystem support (your 6.12 kernel already has it, as confirmed by boot logs and successful 9p tests).

---

## 4. Core System-Mode Workflows

### 4.1 Canonical Multi‑Core System‑Mode Boot (Disk Image)

This is the **verified working** multi‑core system‑mode command:

```bash
QEMU_SYS_BIN="/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64"

"${QEMU_SYS_BIN}" \
  -machine virt \
  -cpu imicpu-v1 \
  -smp 4 \
  -m 4G \
  -kernel /projects2/linhu/VSI/linux-kernels/Image-6.12 \
  -append "root=/dev/vda ro console=ttyS0" \
  -drive file=/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2,format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -nographic
```

Key properties:

- Boots Linux 6.12 on a `riscv-virtio,qemu` machine.
- Rootfs is `rootfs-6.12.ext2` as `/dev/vda`.
- **4 vCPUs** (`-smp 4`) are available for multi‑threaded apps.

### 4.2 Multi‑Core Verification Inside the Guest

Because `lscpu` is not present in this rootfs, use:

```bash
cat /proc/cpuinfo | grep processor
cat /proc/cpuinfo | grep processor | wc -l        # CPU count
cat /proc/cpuinfo | grep "cpu-impl" | head -1    # Should show imicpu-v1
cat /sys/devices/system/cpu/online                # e.g. 0-3
nproc                                             # number of processing units
```

Memory, machine model, and kernel cmdline:

```bash
cat /proc/meminfo | grep MemTotal
uname -a
dmesg | grep "Machine model"
cat /proc/cmdline
```

---

## 5. Exiting and Operating QEMU System Mode

### 5.1 Clean Shutdown (Recommended)

From inside the guest:

```bash
poweroff
# or
shutdown -h now
# or
halt
```

This shuts down the guest OS cleanly; QEMU exits automatically.

### 5.2 QEMU Monitor and Hotkeys

When running with `-nographic` + `-serial mon:stdio`:

- **Enter monitor:** `Ctrl+A`, then `C`
- **Quit from monitor:** at `(qemu)` prompt:

```text
quit
```

- **Force quit:** `Ctrl+A`, then `X`

Common monitor commands:

```text
(qemu) info version
(qemu) info cpus
(qemu) info mem
(qemu) info block
(qemu) info network
```

### 5.3 Kill from Another Terminal (Last Resort)

```bash
ps aux | grep qemu-system-riscv64
kill <PID>         # SIGTERM
kill -9 <PID>      # SIGKILL (force)
```

---

## 6. Login and Rootfs Inspection

### 6.1 Logging In

Typical login prompt:

```text
Welcome to IMI CPU
IMichines login:
```

Most likely credentials:

- `root` with **no password** (just press Enter).
- If that fails, try `root/root`, `root/password`, `admin/admin`, or `imi/imi`.

If you cannot login, you can boot directly to a shell:

```bash
QEMU_SYS_BIN="/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64"

"${QEMU_SYS_BIN}" \
  -machine virt \
  -cpu imicpu-v1 \
  -smp 4 \
  -m 4G \
  -kernel /projects2/linhu/VSI/linux-kernels/Image-6.12 \
  -append "root=/dev/vda rw console=ttyS0 init=/bin/sh" \
  -drive file=/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2,format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -nographic
```

### 6.2 Inspecting Rootfs from the Host (requires sudo)

```bash
sudo mkdir -p /mnt/qemu_rootfs
sudo mount -o loop /projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2 /mnt/qemu_rootfs

sudo find /mnt/qemu_rootfs -name "llama-cli" -o -name "*.gguf"
sudo cat /mnt/qemu_rootfs/etc/passwd

sudo umount /mnt/qemu_rootfs
```

---

## 7. Multi‑Core, Topology, NUMA, and Affinity

### 7.1 `-smp` Basics

Syntax:

```bash
-smp [cpus=]n[,cores=C][,threads=T][,sockets=S][,maxcpus=M]
```

Examples:

- Simple 4‑core machine:

```bash
-smp 4
```

- 8 vCPUs, 2 sockets, 4 cores per socket:

```bash
-smp cpus=8,sockets=2,cores=4,threads=1
```

- 4 cores with 2 hardware threads each (8 vCPUs, SMT style):

```bash
-smp cpus=8,sockets=1,cores=4,threads=2
```

**Rule of thumb:** total vCPUs = `sockets × cores × threads`.  
Match `llama-cli -t N` to that total for best utilization.

### 7.2 Checking Topology and CPU Count Inside Guest

Without `lscpu`:

```bash
cat /proc/cpuinfo | grep processor
cat /proc/cpuinfo | grep processor | wc -l        # vCPU count
cat /sys/devices/system/cpu/online                # online CPU range
```

With `lscpu` (if installed in some other rootfs):

```bash
lscpu | grep -E "CPU\\(s\\)|Thread|Core|Socket"
```

### 7.3 NUMA and “Big/LITTLE” Style Experiments

RISC‑V `virt` in QEMU **does not support heterogeneous CPU models** in the same machine: all vCPUs share the same `-cpu` model (`imicpu-v1` or `rv64`).

What you *can* do:

- Use **NUMA nodes** to group vCPUs:

```bash
"${QEMU_SYS_BIN}" \
  -machine virt \
  -cpu imicpu-v1 \
  -smp cpus=6,sockets=2,cores=3 \
  -m 4G \
  -numa node,nodeid=0,cpus=0-2 \
  -numa node,nodeid=1,cpus=3-5 \
  ... (kernel / rootfs / 9p) ...
```

- Inside guest, inspect:

```bash
cat /proc/cpuinfo | grep processor
ls /sys/devices/system/node/
```

You can *conceptually* treat one NUMA node or socket as “big” vs “little”, but performance is identical—this is only logical grouping.

### 7.4 CPU Affinity (Pinning Threads to Cores)

Inside guest, to control where threads run:

```bash
# Pin whole process to CPUs 0–3
taskset -c 0-3 /path/to/llama-cli -m model.gguf -t 4

# Monitor CPU usage per thread
ps -eLo pid,tid,psr,comm | grep llama-cli
```

This lets you experiment with one‑thread‑per‑core mappings, but remember: in QEMU RISC‑V all cores are architecturally identical.

---

## 8. Running `llama.cpp` in System Mode – Four Approaches

This section consolidates all test approaches into a single, conflict‑free view. They all assume:

- System QEMU: `qemu-system-riscv64` from `install-sys-local`.
- Kernel: `/projects2/linhu/VSI/linux-kernels/Image-6.12` (disk‑based) or `/projects2/linhu/VSI/linux-kernels-v2/Image-6.12` (initrd‑capable).
- Rootfs disk: `/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2`.
- `llama-cli` and GGUF models exist in your host repo.

### 8.1 Overview of the Four Approaches

- **Approach 1 – Manual copy into rootfs (disk image)**  
  Mount rootfs on host (needs `sudo`), copy binaries/models/prompts into it, boot QEMU, run tests from inside guest.

- **Approach 2 – Automated initrd script (no sudo, new kernel)**  
  Create a **cpio initrd** with just what you need (shell, `llama-cli`, model, prompt, small init script), boot with the **v2** kernel that supports initrd. Everything happens in RAM.

- **Approach 3 – Persistent test script embedded in rootfs**  
  Like Approach 1, but add a reusable `/run_test.sh` (optionally auto‑run on boot).

- **Approach 4 – 9p shared filesystem (no sudo, very flexible)**  
  Share your existing host repo into the guest via 9p, mount it, run `llama-cli` directly from the shared tree, and save logs to host. This is your most convenient day‑to‑day method.

#### Recommended Choices (Current Status)

- **Stable & verified working today:** **Approach 4 (9p filesystem)** – this is the method you should use.
- **Conceptual / historical only:** Approaches **1–3** document alternative designs (disk‑copy and initrd‑based flows) but are **not wired into your current workflow** and may not work with the kernels/rootfs you have without additional effort.

---

### 8.2 Approach 1 – Manual Rootfs Editing (with sudo)

**Mount, copy, unmount:**

```bash
sudo mkdir -p /mnt/qemu_rootfs
sudo mount -o loop /projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2 /mnt/qemu_rootfs

# Binary
sudo mkdir -p /mnt/qemu_rootfs/bin
sudo cp /home/linhu/repo/iminn-tools/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
        /mnt/qemu_rootfs/bin/
sudo chmod +x /mnt/qemu_rootfs/bin/llama-cli

# Model
sudo mkdir -p /mnt/qemu_rootfs/models
sudo cp /home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
        /mnt/qemu_rootfs/models/

# Prompt
sudo mkdir -p /mnt/qemu_rootfs/prompts
sudo cp /home/linhu/repo/iminn-tools/src/iminnt/resources/prompts/hello-world.txt \
        /mnt/qemu_rootfs/prompts/

sudo umount /mnt/qemu_rootfs
```

**Boot QEMU (disk image):**

Use the canonical multi‑core command from §4.1.

**Run test inside guest:**

```bash
/bin/llama-cli \
  -m /models/stories15M-q4_0.gguf \
  --seed 42 -t 4 -ngl 0 -n 32 \
  -no-cnv -st --no-warmup \
  --file /prompts/hello-world.txt
```

Pros: persistent, simple mental model. Cons: requires sudo and manual recopy when you change binaries/models.

---

### 8.3 Approach 2 – Automated Initrd Script (No sudo, New Kernel)

This approach uses:

- Kernel with proper initrd support:  
  `/projects2/linhu/VSI/linux-kernels-v2/Image-6.12`
- A script (you had it as `/tmp/prepare_and_run_test_initrd.sh`) that:
  - Builds a temporary initrd tree.
  - Copies `busybox` + dynamic linker + libs (from a one‑time extracted cache).
  - Adds `llama-cli`, GGUF model, prompt file.
  - Writes an `/init` script that:
    - Sets `PATH` and `LD_LIBRARY_PATH`.
    - Ensures `/lib/ld-linux-riscv64-lp64d.so.1` is executable and reachable.
    - Runs the test.
    - Executes `poweroff` at the end.
  - Packs everything into a `newc` cpio archive.
  - Boots QEMU with `-kernel .../linux-kernels-v2/Image-6.12 -initrd /tmp/...cpio`.

Conceptually:

```bash
# Pseudocode for what prepare_and_run_test_initrd.sh does
WORKDIR=/tmp/initrd_work_$$
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# 1) Copy cached shell + libs (from ~/.cache/iminn-tools/initrd_shell/)
# 2) Copy llama-cli, model, and prompt
# 3) Create /init script that runs the test and calls poweroff

find . | cpio -o -H newc > /tmp/test_initrd.cpio

QEMU_SYS_BIN="/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64"

"${QEMU_SYS_BIN}" \
  -machine virt \
  -cpu imicpu-v1 \
  -smp 4 \
  -m 4G \
  -kernel /projects2/linhu/VSI/linux-kernels-v2/Image-6.12 \
  -initrd /tmp/test_initrd.cpio \
  -append "console=ttyS0" \
  -nographic
```

This is the **best no‑sudo automation** path *provided that* the kernel supports initrd. Debugging here centered around:

- Ensuring `/bin/sh` is present and executable.
- Copying **real `busybox`** instead of a symlink.
- Ensuring the dynamic linker (`ld-linux-riscv64-lp64d.so.1`) has execute bit set.
- Verifying the initrd’s contents with `cpio -t` and `file` on binaries.

Because this area was the most fragile, Approach 4 (9p) is still your most robust, low‑friction method for real tests, and Approach 2 remains the "systems‑programming exercise" path.

---

### 8.4 Approach 3 – Embedded Test Script in Rootfs (with sudo)

Same as Approach 1, but you also create a persistent script like `/run_test.sh` in the rootfs:

```bash
sudo mount -o loop /projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2 /mnt/qemu_rootfs

sudo tee /mnt/qemu_rootfs/run_test.sh > /dev/null << 'EOF'
#!/bin/sh
echo "=== Running llama.cpp test_q4_0_stories ==="
/bin/llama-cli \
  -m /models/stories15M-q4_0.gguf \
  --seed 42 -t 4 -ngl 0 -n 32 \
  -no-cnv -st --no-warmup \
  --file /prompts/hello-world.txt
echo "=== Test complete ==="
EOF

sudo chmod +x /mnt/qemu_rootfs/run_test.sh
sudo umount /mnt/qemu_rootfs
```

Then after booting:

```bash
/run_test.sh
```

Optional: configure init or systemd in rootfs to run `/run_test.sh` automatically on boot for fully automated system‑mode tests.

---

### 8.5 Approach 4 – 9p Filesystem (Recommended and Currently Working)

**Host QEMU command (multi‑core, disk image + 9p):**

```bash
#!/usr/bin/env bash

# QEMU system-mode test runner using 9p (virtfs) sharing.
# This script is a recreation of the previous /tmp/run_test_9p.sh
# and is designed to:
#   - NOT require sudo
#   - Share the entire /home/linhu/repo/iminn-tools directory with the guest
#   - Use the working kernel and rootfs (Image-6.12 + rootfs-6.12.ext2)
#
# After booting, you can:
#   - Mount the shared folder in the guest:
#       mkdir -p /mnt/shared
#       mount -t 9p -o trans=virtio,version=9p2000.L shared /mnt/shared
#   - Run llama.cpp tests from /mnt/shared

set -euo pipefail

QEMU_SYS_BIN="/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64"
KERNEL="/projects2/linhu/VSI/linux-kernels/Image-6.12"
ROOTFS="/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2"

# Directory on the HOST to share with the guest
SHARED_PATH="/home/linhu/repo/iminn-tools"
MOUNT_TAG="shared"

# Optional: directory on the HOST where guest test results will be saved
RESULTS_DIR="${SHARED_PATH}/dev_env/qemu_results"

echo "Using QEMU binary:      ${QEMU_SYS_BIN}"
echo "Using kernel image:     ${KERNEL}"
echo "Using rootfs image:     ${ROOTFS}"
echo "Sharing host path:      ${SHARED_PATH}"
echo "Results output (host):  ${RESULTS_DIR}"
echo

if [[ ! -x "${QEMU_SYS_BIN}" ]]; then
  echo "ERROR: QEMU binary not found or not executable: ${QEMU_SYS_BIN}" >&2
  exit 1
fi

if [[ ! -f "${KERNEL}" ]]; then
  echo "ERROR: Kernel image not found: ${KERNEL}" >&2
  exit 1
fi

if [[ ! -f "${ROOTFS}" ]]; then
  echo "ERROR: Rootfs image not found: ${ROOTFS}" >&2
  exit 1
fi

if [[ ! -d "${SHARED_PATH}" ]]; then
  echo "ERROR: Shared path does not exist: ${SHARED_PATH}" >&2
  exit 1
fi

# Ensure results directory exists on the host (will be visible in guest under /mnt/shared/...)
mkdir -p "${RESULTS_DIR}"

echo "Starting QEMU (no sudo required)..."
echo

exec "${QEMU_SYS_BIN}" \
  -machine virt \
  -cpu imicpu-v1 \
  -smp cpus=2 \
  -m 4G \
  -kernel "${KERNEL}" \
  -append "root=/dev/vda ro console=ttyS0" \
  -drive file="${ROOTFS}",format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -fsdev local,id=shared,path="${SHARED_PATH}",security_model=none \
  -device virtio-9p-pci,fsdev=shared,mount_tag="${MOUNT_TAG}" \
  -nographic
```

Save this script as:

```bash
/projects2/linhu/tmp/run_test_9p.sh
chmod +x /projects2/linhu/tmp/run_test_9p.sh
```

Then launch QEMU with:

```bash
/projects2/linhu/tmp/run_test_9p.sh
```

**Inside guest – full command sequence (the part that actually runs `llama-cli`):**

```bash
# 1) Become root if necessary
sudo -i 2>/dev/null || su - || true

# 2) Create and mount shared directory
mkdir -p /mnt/shared
mount -t 9p -o trans=virtio,version=9p2000.L shared /mnt/shared

# 3) Sanity checks
ls /mnt/shared
ls /mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli
ls /mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf
ls /mnt/shared/src/iminnt/resources/prompts/hello-world.txt

# 4) Create results directory (this is on the host!)
mkdir -p /mnt/shared/results

# 5) Match threads to vCPUs:
#   -smp cpus=2    → use -t 2
#   -smp cpus=4    → use -t 4
#   -smp cpus=6... → use -t 6, etc.

TS=$(date +%Y%m%d_%H%M%S)

/mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
  --seed 42 \
  -t 2 \
  -ngl 0 \
  -n 32 \
  -no-cnv -st --no-warmup \
  --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt \
  > "/mnt/shared/results/test_q4_0_stories_system_mode_${TS}.txt" 2>&1

ls -lh /mnt/shared/results
poweroff
```

**On the host**, results live at:

```bash
ls -lh /home/linhu/repo/iminn-tools/results/
cat /home/linhu/repo/iminn-tools/results/test_q4_0_stories_system_mode_*.txt
```

Because 9p shares `/home/linhu/repo/iminn-tools`, the guest path `/mnt/shared/results/...` is literally the same host file.

---

### 8.6 Multi-thread on Multi-core Scheduling and Verification (Approach 4)

In Approach 4, you are using **QEMU system mode + 9p** with:

- QEMU started via the script in this section, configured with **`-smp cpus=2`** (2 vCPUs).
- `llama-cli` run in the guest with **`-t 2`** (2 worker threads).

#### 8.6.1 Step-by-Step: Launch Multi-core QEMU and Run Multi-threaded Test

**Complete walkthrough for 2 cores and 2 threads:**

**Step 1: On the host, launch QEMU with 2 cores**

```bash
/projects2/linhu/tmp/run_test_9p.sh
```

This script uses `-smp cpus=2` to emulate 2 vCPUs. Wait for the guest to boot and show a login prompt.

**Step 2: Inside the guest, log in and mount the shared filesystem**

After QEMU boots, you'll see a login prompt. Log in (usually `root` with no password, or check your rootfs configuration).

**Why mount?** QEMU's 9p filesystem sharing doesn't automatically mount in the guest. You must explicitly mount it to access the host directory. Think of it like a USB drive: it's connected, but you need to mount it to use it.

Once logged in, run:

```bash
# Mount the 9p shared directory
mkdir -p /mnt/shared
mount -t 9p -o trans=virtio,version=9p2000.L shared /mnt/shared

# Verify the mount worked
ls /mnt/shared
```

**What this does:**
- Creates `/mnt/shared` directory in the guest
- Mounts the host's `/home/linhu/repo/iminn-tools` directory to `/mnt/shared` in the guest
- After mounting, files in `/mnt/shared` are the same as files on the host (no copying needed!)

**Step 3: Verify 2 cores are available**

```bash
cat /proc/cpuinfo | grep processor | wc -l   # should print: 2
cat /sys/devices/system/cpu/online           # should print: 0-1
```

**Step 4: Create results directory**

```bash
mkdir -p /mnt/shared/results
```

**Step 5: Run the multi-threaded test in the background**

```bash
TS=$(date +%Y%m%d_%H%M%S)
nohup /mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
  --seed 42 \
  -t 2 \
  -ngl 0 \
  -n 512 \
  -no-cnv -st --no-warmup \
  --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt \
  >"/mnt/shared/results/test_q4_0_stories_system_mode_${TS}.txt" 2>&1 &
```

**Important:** The `-t 2` flag tells `llama-cli` to use 2 threads, matching the 2 vCPUs configured with `-smp cpus=2`.

**Where to check test results:**

The test results are saved to a timestamped file. You can check them in two places:

**Inside the guest OS:**
```bash
# List all result files
ls -lh /mnt/shared/results/

# View the most recent result (replace TS with your actual timestamp)
tail -n 40 /mnt/shared/results/test_q4_0_stories_system_mode_${TS}.txt

# Or view the latest file
ls -t /mnt/shared/results/test_q4_0_stories_system_mode_*.txt | head -1 | xargs tail -n 40

# While test is running, view partial output in real-time
tail -f /mnt/shared/results/test_q4_0_stories_system_mode_${TS}.txt
```

**On the host (after exiting QEMU or in another terminal):**

Since `/mnt/shared` in the guest maps to `/home/linhu/repo/iminn-tools` on the host:

```bash
# List all result files
ls -lh /home/linhu/repo/iminn-tools/results/

# View the most recent result
ls -t /home/linhu/repo/iminn-tools/results/test_q4_0_stories_system_mode_*.txt | head -1 | xargs cat

# Or view all results
cat /home/linhu/repo/iminn-tools/results/test_q4_0_stories_system_mode_*.txt
```

**Note:** The `${TS}` variable was set when you ran the command. If you forgot the timestamp, use `ls -t` to find the newest file.

**Step 6: Verify both cores are active (in another terminal or after a few seconds)**

Open a new guest shell (or wait a moment), then check CPU usage:

```bash
# Check per-CPU activity
cat /proc/stat | grep "^cpu[01]"
# Wait 3-5 seconds, then run again:
cat /proc/stat | grep "^cpu[01]"
```

Both `cpu0` and `cpu1` should show increasing `user` time values.

Or use `top`:

```bash
top
```

- Look for `llama-cli` in the process list.
- The system CPU line should show high utilization (e.g., `99% usr 0% idle`) when both cores are busy.

**Step 7: Check when the test completes**

```bash
# Check if process is still running
jobs
ps | grep llama-cli | grep -v grep

# When finished, view results
tail -n 40 "/mnt/shared/results/test_q4_0_stories_system_mode_${TS}.txt"
```

**Step 8: View results on the host**

After the test completes, you can view the results on the host:

```bash
# On the host (exit QEMU or use another terminal)
ls -lh /home/linhu/repo/iminn-tools/results/
cat /home/linhu/repo/iminn-tools/results/test_q4_0_stories_system_mode_*.txt
```

**Summary:**
- **Host:** Launch QEMU with `-smp cpus=2` → 2 vCPUs emulated
- **Guest:** Run `llama-cli` with `-t 2` → 2 threads created
- **Result:** 2 threads run across 2 vCPUs in parallel

---

#### 8.6.2 Interpreting Test Results

After running a test, the output file contains detailed information about the model execution. Here's how to interpret the key sections:

**1. Initial Warnings (Lines 1-3)**
```
warning: no usable GPU found, --gpu-layers option will be ignored
```
- **Expected:** This is normal when using `-ngl 0` (CPU-only mode)
- **Meaning:** The system is running on CPU, not GPU

**2. Build Information (Line 4)**
```
build: 229 (ddd53c6) with clang version 22.0.0git ... for riscv64-unknown-linux-gnu
```
- **Meaning:** Shows the `llama.cpp` build version and target architecture (RISC-V 64-bit)

**3. Model Loading (Lines 7-92)**
- Shows model metadata, architecture details, and tensor information
- **Key metrics:**
  - `file type = Q4_0`: Quantization format (4-bit quantization)
  - `file size = 17.50 MiB`: Model file size
  - `model params = 24.41 M`: Number of model parameters (24.41 million)
  - `n_layer = 6`: Number of transformer layers
  - `n_head = 6`: Number of attention heads

**4. Thread Configuration (Line 116)**
```
main: llama threadpool init, n_threads = 2
```
- **Critical:** Confirms that 2 threads were initialized (matching your `-t 2` parameter)
- **Verification:** This confirms multi-threading is active

**5. System Info (Line 119)**
```
system_info: n_threads = 2 (n_threads_batch = 2) / 2 | CPU : RISCV_V = 1 | IMI = 1
```
- **Meaning:**
  - `n_threads = 2`: Using 2 threads
  - `RISCV_V = 1`: RISC-V vector extension support enabled
  - `IMI = 1`: IMI CPU extensions enabled

**6. Generated Output (Lines 130-145)**
- The actual text generated by the model
- **Length:** Should match your `-n` parameter (e.g., 512 tokens)

**7. Performance Metrics (Lines 146-152) - Most Important Section**

```bash
common_perf_print:    sampling time =    1081.53 ms
common_perf_print:    samplers time =     187.21 ms /   527 tokens
common_perf_print:        load time =    1575.67 ms
common_perf_print: prompt eval time =    1537.77 ms /    15 tokens (  102.52 ms per token,     9.75 tokens per second)
common_perf_print:        eval time =  131439.26 ms /   511 runs   (  257.22 ms per token,     3.89 tokens per second)
common_perf_print:       total time =  134509.55 ms /   526 tokens
```

**Key Performance Indicators:**

- **`load time`**: Time to load the model into memory (one-time cost)
  - Example: `1575.67 ms` ≈ 1.6 seconds

- **`prompt eval time`**: Time to process the input prompt
  - Example: `1537.77 ms / 15 tokens` = 102.52 ms per token, 9.75 tokens/second
  - This is typically faster than generation

- **`eval time`**: Time to generate new tokens (most important for inference speed)
  - Example: `131439.26 ms / 511 runs` = 257.22 ms per token, **3.89 tokens per second**
  - **This is your inference throughput** - how fast the model generates text

- **`total time`**: End-to-end time from start to finish
  - Example: `134509.55 ms` ≈ 134.5 seconds (2.24 minutes) for 526 tokens

- **`sampling time`**: Time spent in token sampling/selection logic
  - Usually a small fraction of total time

**8. Memory Breakdown (Lines 154-157)**

```
llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
llama_memory_breakdown_print: |   - Host               |                  108 =    17 +      27 +      63                |
llama_memory_breakdown_print: |   - CPU_IMI            |                   12 =    12 +       0 +       0                |
```

- **Host memory**: Total RAM used (108 MiB)
  - Model: 17 MiB
  - Context (KV cache): 27 MiB
  - Compute buffers: 63 MiB
- **CPU_IMI memory**: IMI-specific memory (12 MiB)

**What to Look For:**

✅ **Success indicators:**
- `n_threads = 2` appears in the output (confirms multi-threading)
- No error messages (only expected GPU warnings)
- Performance metrics show reasonable token generation speed
- Generated text appears at the end

⚠️ **Potential issues:**
- If `n_threads = 1` appears instead of 2, multi-threading didn't work
- Very slow token generation (< 1 token/second) might indicate single-core execution
- Memory errors would appear as explicit error messages

**Comparing Performance:**

- **Single-core baseline**: Run with `-t 1` and compare `eval time` and tokens/second
- **Multi-core speedup**: With `-t 2`, you should see:
  - Similar or better tokens/second (ideally 1.5-2x faster)
  - Lower `eval time` per token
  - Both CPU cores showing activity (verified via `/proc/stat`)

---

#### 8.6.3 Performance Comparison: Single-core vs Multi-core

To verify that multi-core execution provides a speedup, you can run both single-core and multi-core tests and compare the results.

**Step 1: Run Single-core Baseline Test**

Inside the guest OS, run the test with `-t 1` (single thread):

```bash
TS_SINGLE=$(date +%Y%m%d_%H%M%S)
nohup /mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
  --seed 42 \
  -t 1 \
  -ngl 0 \
  -n 512 \
  -no-cnv -st --no-warmup \
  --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt \
  >"/mnt/shared/results/test_q4_0_stories_system_mode_single_${TS_SINGLE}.txt" 2>&1 &
```

**Note:** The only difference is `-t 1` instead of `-t 2`.

Wait for the test to complete, then check the results:

```bash
# Check if process finished
ps | grep llama-cli | grep -v grep

# View results
tail -n 20 "/mnt/shared/results/test_q4_0_stories_system_mode_single_${TS_SINGLE}.txt"
```

**Step 2: Run Multi-core Test**

Run the same test with `-t 2` (as shown in Step 5 of section 8.6.1):

```bash
TS_MULTI=$(date +%Y%m%d_%H%M%S)
nohup /mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
  --seed 42 \
  -t 2 \
  -ngl 0 \
  -n 512 \
  -no-cnv -st --no-warmup \
  --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt \
  >"/mnt/shared/results/test_q4_0_stories_system_mode_multi_${TS_MULTI}.txt" 2>&1 &
```

**Step 3: Extract Key Metrics from Both Tests**

On the host (or in guest), extract the performance metrics:

**For single-core test:**
```bash
# Extract eval time and tokens/second from single-core test
grep "eval time" /home/linhu/repo/iminn-tools/results/test_q4_0_stories_system_mode_single_*.txt | tail -1
```

**For multi-core test:**
```bash
# Extract eval time and tokens/second from multi-core test
grep "eval time" /home/linhu/repo/iminn-tools/results/test_q4_0_stories_system_mode_multi_*.txt | tail -1
```

**Example output format:**
```
common_perf_print:        eval time =  131439.26 ms /   511 runs   (  257.22 ms per token,     3.89 tokens per second)
```

**Step 4: Compare the Metrics**

**Option A: Use the automated comparison script (recommended)**

On the host, use the provided comparison script:

```bash
# Automatically finds and compares the most recent single-core and multi-core tests
/home/linhu/repo/iminn-tools/scripts/compare_qemu_performance.sh
```

Or specify files explicitly:

```bash
/home/linhu/repo/iminn-tools/scripts/compare_qemu_performance.sh \
  /home/linhu/repo/iminn-tools/results/test_q4_0_stories_system_mode_single_20260105_184739.txt \
  /home/linhu/repo/iminn-tools/results/test_q4_0_stories_system_mode_multi_20260105_185000.txt
```

The script will show:
- Thread configuration for both tests
- Eval time metrics
- Tokens per second comparison with speedup calculation
- Eval time per token comparison
- Total eval time comparison
- Summary with speedup assessment

**Option B: Manual comparison**

**Key metrics to compare:**

1. **Tokens per second (throughput)**
   - Single-core: Look for the number after "tokens per second"
   - Multi-core: Should be **higher** (ideally 1.5-2x faster)

2. **Eval time per token**
   - Single-core: Look for "ms per token" in the eval time line
   - Multi-core: Should be **lower** (faster)

3. **Total eval time**
   - Single-core: Total time for all token generation
   - Multi-core: Should be **lower** (faster overall)

**Manual comparison example:**

```bash
# On host, create a comparison script
cat > /tmp/compare_perf.sh << 'EOF'
#!/bin/bash
SINGLE_FILE=$(ls -t /home/linhu/repo/iminn-tools/results/test_q4_0_stories_system_mode_single_*.txt | head -1)
MULTI_FILE=$(ls -t /home/linhu/repo/iminn-tools/results/test_q4_0_stories_system_mode_multi_*.txt | head -1)

echo "=== Single-core Performance ==="
grep "eval time" "$SINGLE_FILE" | tail -1
grep "n_threads = " "$SINGLE_FILE" | head -1

echo ""
echo "=== Multi-core Performance ==="
grep "eval time" "$MULTI_FILE" | tail -1
grep "n_threads = " "$MULTI_FILE" | head -1

echo ""
echo "=== Speedup Calculation ==="
SINGLE_TPS=$(grep "eval time" "$SINGLE_FILE" | tail -1 | grep -oP '\d+\.\d+ tokens per second')
MULTI_TPS=$(grep "eval time" "$MULTI_FILE" | tail -1 | grep -oP '\d+\.\d+ tokens per second')

echo "Single-core: $SINGLE_TPS"
echo "Multi-core:  $MULTI_TPS"
EOF

chmod +x /tmp/compare_perf.sh
/tmp/compare_perf.sh
```

**Step 5: Verify Multi-core CPU Usage**

While the multi-core test is running, verify both cores are active:

```bash
# In guest, while test is running
cat /proc/stat | grep "^cpu[01]"
# Wait 5 seconds
sleep 5
cat /proc/stat | grep "^cpu[01]"
```

**What to look for:**
- Both `cpu0` and `cpu1` should show increasing `user` time values
- The difference between the two readings should show both CPUs accumulating user time

**Expected Results:**

✅ **Good multi-core speedup:**
- Multi-core tokens/second is **1.5-2x** higher than single-core
- Multi-core eval time per token is **lower** than single-core
- Both CPU cores show activity in `/proc/stat`

⚠️ **Limited or no speedup:**
- Multi-core performance is similar to single-core (may indicate single-core execution)
- Only one CPU shows activity in `/proc/stat`
- Check that `n_threads = 2` appears in the multi-core output

**Real-world Example Results:**

Here's an actual comparison from a 2-core QEMU setup:

```
=== Thread Configuration ===
Single-core: 1 threads
Multi-core:  2 threads

=== Throughput Comparison ===
Single-core: 2.55 tokens/second
Multi-core:  3.88 tokens/second
Speedup:     1.52x

=== Eval Time Per Token ===
Single-core: 392.15 ms per token
Multi-core:  257.95 ms per token
Speedup:     1.52x (lower is better)

=== Total Eval Time ===
Single-core: 200.39 seconds (200390.62 ms)
Multi-core:  131.81 seconds (131814.18 ms)
Speedup:     1.52x
```

**Interpretation:**
- ✅ **1.52x speedup** confirms multi-core execution is working
- ✅ Both cores are being utilized (verified by `/proc/stat` monitoring)
- ✅ Consistent speedup across all metrics (throughput, latency, total time)
- ℹ️ **Not perfect 2x scaling** is normal due to:
  - Memory bandwidth limitations
  - Thread synchronization overhead
  - Some sequential portions of computation
  - Cache effects and memory access patterns

This level of speedup (1.5-1.6x) is typical for CPU-bound inference workloads on 2 cores.

**Factors affecting speedup:**
- **Model size**: Larger models may show better multi-core scaling
- **Token count**: Longer sequences may show more consistent speedup
- **Memory bandwidth**: If memory-bound, speedup may be limited
- **Thread overhead**: Very small workloads may not benefit from multi-threading

**Example Comparison Table:**

| Metric | Single-core (`-t 1`) | Multi-core (`-t 2`) | Speedup |
|--------|----------------------|---------------------|---------|
| Eval time per token | 500 ms | 257 ms | 1.95x |
| Tokens per second | 2.0 | 3.89 | 1.95x |
| Total eval time | 255.5 sec | 131.4 sec | 1.95x |
| Threads used | 1 | 2 | - |

---

#### 8.6.4 How scheduling works in the guest

- Guest Linux sees:
  - 2 CPUs (vCPUs): **CPU 0** and **CPU 1**.
  - One process (`llama-cli`) that spawns 2 threads (because of `-t 2`).
- If you **don’t** restrict affinity:
  - The guest scheduler is free to move both threads across CPUs 0 and 1.
  - Under load, it will normally **spread** the 2 threads across the 2 CPUs, so they can run in parallel.
- If you wrap the command with `taskset`:

  ```bash
  taskset -c 0,1 \
    /mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
      -m /mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
      --seed 42 -t 2 -ngl 0 -n 32 \
      -no-cnv -st --no-warmup \
      --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt
  ```

  - You tell Linux: *all `llama-cli` threads may only run on CPUs 0 and 1*.
  - The scheduler will then typically keep one thread on each core while both are busy.

You can also use `taskset -c 0` to force both threads onto CPU 0 (for comparison), which will make them time-slice on a single emulated core.

#### 8.6.5 Verifying that 2 cores and 2 threads are used

All of these checks happen **inside the guest**.

1. **Confirm QEMU exposed 2 vCPUs**

```bash
cat /proc/cpuinfo | grep processor | wc -l   # should print 2
cat /sys/devices/system/cpu/online           # typically "0-1"
```

If this is not 2, the `-smp` setting in the host script is not what you expect.

2. **Run `llama-cli` with 2 threads (and optional pinning)**

With affinity (recommended for clear behavior):

```bash
taskset -c 0,1 \
  /mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
    -m /mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
    --seed 42 -t 2 -ngl 0 -n 32 \
    -no-cnv -st --no-warmup \
    --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt
```

Without `taskset`:

```bash
/mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
  --seed 42 -t 2 -ngl 0 -n 32 \
  -no-cnv -st --no-warmup \
  --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt
```

For a **single manual test run in the background** (so you can monitor with `top`, etc.), use `nohup` to avoid terminal-related interruptions:

```bash
TS=$(date +%Y%m%d_%H%M%S)
nohup /mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
  --seed 42 \
  -t 2 \
  -ngl 0 \
  -n 512 \
  -no-cnv -st --no-warmup \
  --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt \
  >"/mnt/shared/results/test_q4_0_stories_system_mode_${TS}.txt" 2>&1 &
```

This saves output to a timestamped file in `/mnt/shared/results/` on the host (accessible from guest as `/mnt/shared/results/`).

3. **While it runs, verify both cores are active**

Check per-CPU usage with `/proc/stat` (works with BusyBox):

```bash
cat /proc/stat | grep "^cpu[01]"
# Wait a few seconds, then run again:
cat /proc/stat | grep "^cpu[01]"
```

- Both `cpu0` and `cpu1` should show increasing `user` time values, indicating both cores are working.

Or use `top`:

```bash
top
```

- Look for `llama-cli` in the process list and check its `%CPU` column.
- Note: BusyBox `top` may show `%CPU` as 100% (representing aggregate view), but `/proc/stat` confirms per-CPU activity.
- Overall system CPU line should show high utilization (e.g., `99% usr 0% idle`) when both cores are busy.

4. **Check when the process completes**

```bash
jobs          # if empty, process has finished
ps | grep llama-cli | grep -v grep   # if nothing, process finished
tail -n 40 "/mnt/shared/results/test_q4_0_stories_system_mode_${TS}.txt"  # view results
```

This gives you a practical, repeatable way in Approach 4 to:

- Configure **multi-thread on multi-core** (2 threads on 2 QEMU-emulated cores).
- Verify that the threads are indeed running across the 2 vCPUs.
- Monitor and confirm completion of the test.

---

## 9. Standard Testing and Debugging Flow

### 9.1 Layered Test Strategy

1. **Level 0 – Verify tools & components**
   - QEMU system binary exists and runs.
   - Kernel is RISC‑V 64‑bit.
   - Rootfs disk image exists (for disk‑based boot).
   - Optional: kernel supports initrd (for initrd‑based tests).

2. **Level 1 – Minimal boot**
   - For disk image: boot with `root=/dev/vda ro console=ttyS0` and ensure login prompt appears.
   - For initrd: boot trivial initrd with `/init` that prints a message and `poweroff`.

3. **Level 2 – Shell in initrd / rootfs**
   - Provide `/bin/sh` and launch it from `/init` to debug interactively.

4. **Level 3 – Binary smoke test**
   - Add `llama-cli` but run `--help` or a tiny prompt.

5. **Level 4 – Full test (`test_q4_0_stories`)**
   - Add model + prompt, run full invocation, capture logs.

### 9.2 Common Failure Modes and Fixes

- **Kernel panic: unable to mount root fs (error -6) with `-initrd`**
  - Kernel lacks initrd support – use disk image instead, or switch to an initrd‑capable kernel.

- **Kernel panic: Failed to execute /init (error -2 / -13 / -80)**
  - `/init` missing, not executable, or interpreter (`/bin/sh` + dynamic linker) missing/invalid.
  - Fix by:
    - Ensuring `/init` exists and is `chmod +x`.
    - Including a real shell binary (e.g. `busybox`) instead of symlinks.
    - Including the dynamic linker and making it executable.

- **`error: failed to open model or prompt` in guest**
  - Path mistakes (most often with 9p).
  - From guest, **always** use absolute guest paths:
    - Model: `/mnt/shared/dev_env/llama.cpp/models/...`
    - Prompt: `/mnt/shared/src/iminnt/resources/prompts/...`

- **9p mount errors**
  - `"unknown filesystem type '9p'"` → kernel lacks 9p support → use disk‑copy approaches.
  - `"No such device"` → `virtio-9p-pci` or `fsdev` not set correctly; check QEMU command for `-fsdev` and `-device virtio-9p-pci,...`.

- **Only 1 CPU visible despite `-smp N`**
  - Check from guest:

    ```bash
    cat /proc/cpuinfo | grep processor | wc -l
    dmesg | grep "Brought up .* CPUs"
    ```

  - Some kernels must be built with `CONFIG_SMP`; your 6.12 kernel already has SMP and is verified with 4 CPUs.

---

## 10. Quick Reference

- **User vs System mode**
  - User: `qemu-riscv64`, one emulated CPU, host syscalls; used by `iminnt` today.
  - System: `qemu-system-riscv64`, full OS, multi‑core via `-smp`.

- **Core system‑mode boot (disk + 4 cores)**

```bash
qemu-system-riscv64 -machine virt -cpu imicpu-v1 -smp 4 -m 4G \
  -kernel /projects2/linhu/VSI/linux-kernels/Image-6.12 \
  -append "root=/dev/vda ro console=ttyS0" \
  -drive file=/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2,format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -nographic
```

- **Mount 9p in guest**

```bash
mkdir -p /mnt/shared
mount -t 9p -o trans=virtio,version=9p2000.L shared /mnt/shared
```

- **Check CPU count (no `lscpu`)**

```bash
cat /proc/cpuinfo | grep processor | wc -l
cat /sys/devices/system/cpu/online
```

- **Exit QEMU cleanly**

```bash
poweroff       # inside guest
Ctrl+A, C; quit   # QEMU monitor
Ctrl+A, X         # force quit
```

- **Run `llama.cpp` test via 9p & save log**

```bash
mkdir -p /mnt/shared/results
TS=$(date +%Y%m%d_%H%M%S)
/mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli \
  -m /mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf \
  --seed 42 -t 4 -ngl 0 -n 32 \
  -no-cnv -st --no-warmup \
  --file /mnt/shared/src/iminnt/resources/prompts/hello-world.txt \
  > "/mnt/shared/results/test_q4_0_stories_system_mode_${TS}.txt" 2>&1
```

This file is your **single source of truth** for QEMU fundamentals, configuration, multi‑core behavior, and practical system‑mode workflows (disk, initrd, and 9p) in the `iminn-tools` environment.


