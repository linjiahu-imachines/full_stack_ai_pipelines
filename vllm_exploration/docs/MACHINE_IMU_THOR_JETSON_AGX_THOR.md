# Machine profile: imu-thor (Jetson AGX Thor)

**Purpose:** Record hardware and software identity for this **new machine** so benchmarks, vLLM builds, and CUDA/PyTorch choices can be interpreted correctly.

**Recorded:** 2026-04-03  
**Hostname:** `imu-thor`

---

## 1. System summary

| Item | Value |
|------|--------|
| **Board / product** | NVIDIA Jetson AGX Thor Developer Kit (from `/proc/device-tree/model`) |
| **OS** | Ubuntu 24.04.3 LTS (Noble Numbat) |
| **Kernel** | Linux 6.8.12-tegra, aarch64, SMP PREEMPT |
| **Architecture** | **aarch64** (64-bit ARM little-endian) |

This is an **embedded SoC** platform (Tegra/Jetson family), not a desktop x86 server. Software must use **ARM64** binaries and, for GPU/CUDA, stacks intended for **Jetson / L4T** where applicable.

---

## 2. CPU

| Item | Value |
|------|--------|
| **Vendor** | ARM |
| **Logical CPUs** | 14 (1 thread per core) |
| **Topology** | 1 cluster, 14 cores per cluster |
| **CPU max clock** | 2601 MHz (min ~54 MHz; cpufreq scaling active) |
| **Caches (reported)** | L1d 896 KiB (14×), L1i 896 KiB (14×), L2 14 MiB (14×) |
| **NUMA** | 1 node (CPUs 0–13) |

**ISA highlights (from `lscpu` flags):** ASIMD, SVE/SVE2, AES, SHA2/SHA3, BF16, I8MM, and related ARMv9-era features.

NVIDIA positions Jetson Thor’s CPU complex as **Arm Neoverse–class** cores suitable for robotics and edge AI workloads (see [NVIDIA Jetson Thor announcement](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)).

---

## 3. Memory and storage

| Resource | Value (at inventory) |
|----------|----------------------|
| **RAM** | ~122 GiB total; ~118 GiB available under light load |
| **Swap** | None configured (0 B) |
| **Root disk** | ~937 GB total, ~873 GB free on `/` (NVMe `nvme0n1p1`) |

---

## 4. GPU — overview

| Item | Value |
|------|--------|
| **Marketing name** | NVIDIA Thor (integrated GPU on Jetson AGX Thor) |
| **Reported architecture** | **Blackwell** (`Product Architecture` in `nvidia-smi -q`) |
| **CUDA compute capability** | **11.0** (`nvidia-smi --query-gpu=compute_cap`) |
| **PCI device** | `00000000:01:00.0`, NVIDIA device `0x2B00` (rev a1), GPU part number **2B00--A1** |
| **Driver** | 580.00 |
| **CUDA version (driver reports)** | 13.0 |
| **UUID** | `GPU-a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600` |

**Brand line in NVML:** `Product Brand` is reported as GeForce for this SKU; the important part for software is **architecture = Blackwell** and **compute capability 11.0**.

---

## 5. GPU architecture (what “Blackwell” means here)

**Short answer:** On this machine the GPU is **NVIDIA Blackwell**-generation silicon, exposed to CUDA as **compute capability 11.0**.

**Architecture (microarchitecture):**  
**Blackwell** is NVIDIA’s GPU microarchitecture family succeeding Hopper. It brings updated SMs, memory subsystem, and features aimed at large-scale AI (e.g. emphasis on low-precision tensor work and datacenter-scale designs). The **Jetson AGX Thor** integrates a **Blackwell-based GPU** in a single SoC with unified memory and a power envelope suitable for robotics and edge deployment—not a discrete PCIe datacenter card.

**Compute capability:**  
`11.0` is the **CUDA hardware capability** identifier your toolchain uses to select kernels and features. PyTorch, TensorRT, and vLLM must be built (or shipped as wheels) for **CUDA 12.x/13.x + sm_110** (or Jetson-specific builds) to use this GPU fully.

**Relation to older lab docs in this repo:**  
Prior `vllm_exploration` write-ups assumed **x86_64**, **discrete GPUs** (e.g. TITAN RTX, compute capability 7.5), and different wheel lines. **This machine is aarch64 + integrated Blackwell (sm_110).** Expect different install paths, performance, and sometimes **Jetson/L4T** packages instead of generic Linux x86 CUDA wheels.

**Official / high-level references:**

- [Introducing NVIDIA Jetson Thor (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)  
- [Blackwell architecture compatibility (CUDA docs)](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)  
- Press/product: [Blackwell-powered Jetson Thor availability](https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Blackwell-Powered-Jetson-Thor-Now-Available-Accelerating-the-Age-of-General-Robotics/default.aspx)

---

## 6. GPU — extra details from `nvidia-smi -q` (sample snapshot)

Values below were captured during inventory; they can change with load, driver, or power mode.

| Topic | Observation |
|--------|-------------|
| **Temperature** | GPU ~31 °C (idle/light load) |
| **Power** | Instantaneous draw ~2.38 W (idle sample) |
| **Clocks (sample)** | Graphics / Video ~315 MHz at idle |
| **PCIe link (reported)** | Current gen 1, current width 1×; max device gen 5, max width 16× (link training / SoC wiring may differ from discrete GPU servers) |
| **MIG** | Disabled |
| **Processes** | Example: Xorg (~48 MiB), gnome-shell (~25 MiB) GPU memory |

Some fields (fan, ECC, full memory totals) may show **N/A** on integrated Jetson GPUs compared to datacenter cards.

---

## 7. Software notes (inventory)

| Component | Status (at inventory) |
|-----------|------------------------|
| **`nvcc`** | Not in default `PATH` (full CUDA toolkit may be optional or installed elsewhere) |
| **PyTorch (`python3`)** | Not verified in system Python; use project `venv` and Jetson-compatible installs |

**Driver / userspace warnings:**  
In some environments `nvidia-smi` may print **NvRm / nvmap** messages (e.g. permission or memory manager) while still listing the GPU. If CUDA apps fail, check permissions, Jetson power mode, and that you use **aarch64 + Jetson**-appropriate CUDA stacks.

---

## 8. How to refresh this document

Run on the machine:

```bash
uname -a
cat /etc/os-release
cat /proc/device-tree/model 2>/dev/null | tr -d '\0'
lscpu
free -h
df -h /
nvidia-smi -q
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
```

Update the tables and the date at the top when hardware or OS changes.

---

**Document status:** Living inventory for **imu-thor** (Jetson AGX Thor, Blackwell GPU, sm_110).
