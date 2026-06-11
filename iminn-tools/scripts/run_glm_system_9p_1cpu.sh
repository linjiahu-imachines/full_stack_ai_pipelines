#!/usr/bin/env bash
# QEMU system-mode (disk rootfs) + 9p share, single vCPU.
# After boot, run the GLM command inside the guest (see below).

set -euo pipefail

QEMU_SYS_BIN="/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64"
KERNEL="/projects2/linhu/VSI/linux-kernels/Image-6.12"
ROOTFS="/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2"

SHARED_PATH="/home/linhu/repo/iminn-tools"
MOUNT_TAG="shared"

if [[ ! -x "${QEMU_SYS_BIN}" ]]; then
  echo "ERROR: QEMU system binary not found/executable: ${QEMU_SYS_BIN}" >&2
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
  echo "ERROR: Shared path not found: ${SHARED_PATH}" >&2
  exit 1
fi

echo "Starting QEMU system-mode (1 vCPU) with 9p share..."
echo "  QEMU:    ${QEMU_SYS_BIN}"
echo "  Kernel:  ${KERNEL}"
echo "  Rootfs:  ${ROOTFS}"
echo "  Share:   ${SHARED_PATH} (mount tag: ${MOUNT_TAG})"
echo
echo "Inside guest, run:"
echo "  mkdir -p /mnt/shared && mount -t 9p -o trans=virtio,version=9p2000.L ${MOUNT_TAG} /mnt/shared"
echo "  (then run llama-cli from /mnt/shared/...)"
echo
echo "To exit QEMU:"
echo "  - Ctrl+C (may take a moment), OR"
echo "  - Ctrl+A then C, type: quit"
echo

exec "${QEMU_SYS_BIN}" \
  -machine virt \
  -cpu imicpu-v1 \
  -smp cpus=1 \
  -m 4G \
  -kernel "${KERNEL}" \
  -append "root=/dev/vda ro console=ttyS0" \
  -drive file="${ROOTFS}",format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -fsdev local,id=shared,path="${SHARED_PATH}",security_model=none \
  -device virtio-9p-pci,fsdev=shared,mount_tag="${MOUNT_TAG}" \
  -nographic
