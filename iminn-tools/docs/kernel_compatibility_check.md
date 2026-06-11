# RISC-V Kernel Compatibility Check

**Date:** December 9, 2025  
**Location:** `/projects2/linhu/VSI/linux-kernels`

## Available Kernels

### Kernel 6.12
- **File:** `Image-6.12`
- **Size:** 24 MB
- **Type:** RISC-V 64-bit EFI application
- **Version:** Linux 6.12.0 (compiled Nov 27, 2025)
- **Compiler:** riscv64-unknown-linux-gnu-gcc 14.2.0
- **SMP Support:** ✅ Yes (SMP in version string)
- **Rootfs:** `rootfs-6.12.ext2` (512 MB)
- **Status:** ✅ **READY TO USE**

### Kernel 6.6.37
- **File:** `Image-6.6.37`
- **Size:** 22 MB
- **Type:** RISC-V 64-bit EFI application
- **Version:** Linux 6.6.37 (compiled Nov 30, 2025)
- **Compiler:** riscv64-unknown-linux-gnu-gcc 13.2.0
- **SMP Support:** ✅ Yes (SMP in version string)
- **Rootfs:** `rootfs-6.6.37.ext2` (512 MB)
- **Status:** ✅ **READY TO USE**

## Compatibility Check

### ✅ Both Kernels Are Compatible

**Architecture:** Both are RISC-V 64-bit ✅  
**Format:** EFI application format (correct for QEMU virt machine) ✅  
**SMP Support:** Both support multi-core ✅  
**Rootfs:** Both have matching ext2 rootfs files ✅

### QEMU CPU Support

Your QEMU system mode binary supports:
- ✅ `imicpu-v1` (I-Machines CPU)
- ✅ `rv64` (Standard RISC-V 64-bit)

**Both kernels should work with either CPU model.**

## Recommended Usage

### Option 1: Use Kernel 6.12 (Newer)

```bash
QEMU_SYS_BIN="/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64"
KERNEL="/projects2/linhu/VSI/linux-kernels/Image-6.12"
ROOTFS="/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2"

${QEMU_SYS_BIN} \
  -machine virt \
  -cpu imicpu-v1 \
  -smp 4 \
  -m 4G \
  -kernel ${KERNEL} \
  -append "root=/dev/vda ro console=ttyS0" \
  -drive file=${ROOTFS},format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -nographic
```

### Option 2: Use Kernel 6.6.37 (Older but Stable)

```bash
QEMU_SYS_BIN="/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64"
KERNEL="/projects2/linhu/VSI/linux-kernels/Image-6.6.37"
ROOTFS="/projects2/linhu/VSI/linux-kernels/rootfs-6.6.37.ext2"

${QEMU_SYS_BIN} \
  -machine virt \
  -cpu imicpu-v1 \
  -smp 4 \
  -m 4G \
  -kernel ${KERNEL} \
  -append "root=/dev/vda ro console=ttyS0" \
  -drive file=${ROOTFS},format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -nographic
```

## Next Steps

1. **Test boot with single core:**
   ```bash
   # Try kernel 6.12 first
   qemu-system-riscv64 -machine virt -cpu imicpu-v1 -smp 1 -m 2G \
     -kernel /projects2/linhu/VSI/linux-kernels/Image-6.12 \
     -append "root=/dev/vda ro console=ttyS0" \
     -drive file=/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2,format=raw,id=hd0 \
     -device virtio-blk-device,drive=hd0 \
     -nographic
   ```

2. **Check rootfs contents:**
   - Verify llama-cli binary exists
   - Check if GGUF models are present
   - Verify required libraries are available

3. **Test multi-core:**
   - Use `-smp 4` for 4 cores
   - Verify with `cat /proc/cpuinfo` inside guest OS

## Rootfs Contents Check

**To check what's in the rootfs (requires sudo):**

```bash
# Mount rootfs
sudo mkdir -p /mnt/rootfs
sudo mount -o loop /projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2 /mnt/rootfs

# Check for llama-cli
ls -la /mnt/rootfs/usr/bin/llama-cli
ls -la /mnt/rootfs/bin/llama-cli

# Check for models
find /mnt/rootfs -name "*.gguf" 2>/dev/null

# Unmount
sudo umount /mnt/rootfs
```

## Summary

| Kernel | Version | Size | Rootfs | Status |
|--------|---------|------|--------|--------|
| **6.12** | 6.12.0 | 24 MB | rootfs-6.12.ext2 (512 MB) | ✅ Ready |
| **6.6.37** | 6.6.37 | 22 MB | rootfs-6.6.37.ext2 (512 MB) | ✅ Ready |

**Recommendation:** Start with **Kernel 6.12** (newer version, more recent).

Both kernels are RISC-V 64-bit, support SMP (multi-core), and have matching rootfs files. They should work with your QEMU system mode binary!

