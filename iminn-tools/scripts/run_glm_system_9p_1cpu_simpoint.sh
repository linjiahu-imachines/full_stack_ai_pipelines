#!/usr/bin/env bash
# QEMU system-mode (disk rootfs) + 9p share + SimPoint basic-block profiling (host-side plugin output).
#
# This script:
#  - boots qemu-system-riscv64 with libsimpbbprof.so enabled
#  - writes BB.log and qemu_simpoint.bb to a HOST output directory (not inside guest)
#  - you then run GLM inside the guest as usual
#  - after QEMU exits, run dev_env/simpoint/bin/simpoint on qemu_simpoint.bb
#
# NOTE:
#  - This is NOT Permafrost/Pilos "psim simpoint". It's QEMU system-mode basic-block profiling + SimPoint clustering.
#  - The BB vector will include kernel/boot activity unless you make your rootfs auto-run workload immediately.

set -euo pipefail

QEMU_SYS_BIN="/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/bin/qemu-system-riscv64"
KERNEL="/projects2/linhu/VSI/linux-kernels/Image-6.12"
ROOTFS="/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2"

SHARED_PATH="/home/linhu/repo/iminn-tools"
MOUNT_TAG="shared"

PLUGIN="/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/plugins/libsimpbbprof.so"
ROI_PLUGIN="/home/linhu/repo/iminn-tools/dev_env/csqemu-v9/install-sys-local/plugins/libroi-always-on.so"

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="/home/linhu/repo/iminn-tools/results/qemu_sys_simpoint_${TS}"
BBLOG="${OUT_DIR}/BB.log"
BBVEC="${OUT_DIR}/qemu_simpoint.bb"

mkdir -p "${OUT_DIR}"

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
if [[ ! -f "${PLUGIN}" ]]; then
  echo "ERROR: SimPoint BB profiler plugin not found: ${PLUGIN}" >&2
  exit 1
fi
if [[ ! -f "${ROI_PLUGIN}" ]]; then
  echo "ERROR: ROI enable plugin not found: ${ROI_PLUGIN}" >&2
  echo "Build it on host with:" >&2
  echo "  gcc -shared -fPIC -O2 \\" >&2
  echo "    -I /home/linhu/repo/iminn-tools/dev_env/csqemu-v9/include \\" >&2
  echo "    \$(pkg-config --cflags glib-2.0) \\" >&2
  echo "    /home/linhu/repo/iminn-tools/dev_env/csqemu-v9/contrib/plugins/roi-always-on.c \\" >&2
  echo "    -o ${ROI_PLUGIN} \\" >&2
  echo "    \$(pkg-config --libs glib-2.0)" >&2
  exit 1
fi

echo "Starting QEMU system-mode (1 vCPU) WITH simpoint BB profiling..."
echo "  OUT_DIR: ${OUT_DIR}"
echo "  BB.log : ${BBLOG}"
echo "  BB vec : ${BBVEC}"
echo
echo "Inside guest, run:"
echo "  mkdir -p /mnt/shared && mount -t 9p -o trans=virtio,version=9p2000.L ${MOUNT_TAG} /mnt/shared"
echo "  (then run llama-cli from /mnt/shared/...)"
echo
echo "After QEMU exits, run on host:"
echo "  /home/linhu/repo/iminn-tools/dev_env/simpoint/bin/simpoint -maxK 30 \\"
echo "    -saveSimpoints ${OUT_DIR}/simpoints \\"
echo "    -saveSimpointWeights ${OUT_DIR}/weights \\"
echo "    -loadFVFile ${BBVEC}"
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
  -plugin "${ROI_PLUGIN}" \
  -plugin "${PLUGIN}",interval=10000000,bbfile="${BBVEC}",bblogfile="${BBLOG}" \
  -nographic

