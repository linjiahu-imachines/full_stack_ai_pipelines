from pathlib import Path
import os

## Default logging config
BUILD_DEBUG = False
# Binaries and sources are stored in a bunch of different places, so these help organize them

### General purpose paths
IMINNT_ROOT = Path(__file__).parent.parent.parent
DEV_ENV_ROOT = IMINNT_ROOT / "dev_env"
PYSRC_ROOT = IMINNT_ROOT / "src" / "iminnt"
CROSS_TOOLCHAIN_PATH = IMINNT_ROOT / "toolchain.cmake"

# Uncomment this to use the pre-built riscv, which has fewer features, but more stable
# IMI_SDK_ROOT = DEV_ENV_ROOT / "riscv-sdk" / "llvm-project-install"
IMI_SDK_ROOT = DEV_ENV_ROOT / "riscv-env" / "riscv"
QEMU_BASE_DIR = DEV_ENV_ROOT / "csqemu-v9"
DEBUG_QEMU = False
if DEBUG_QEMU:
    QEMU_USER_DIR = QEMU_BASE_DIR / "install-debug"
else:
    QEMU_USER_DIR = QEMU_BASE_DIR / "install-local"

QEMU_SYS_DIR = QEMU_BASE_DIR / "install-sys-local"

QEMU_SYS_BIN = QEMU_SYS_DIR / "bin" / "qemu-system-riscv64"

# System mode configuration
QEMU_SYS_KERNEL = Path("/projects2/linhu/VSI/linux-kernels/Image-6.12")
QEMU_SYS_ROOTFS = Path("/projects2/linhu/VSI/linux-kernels/rootfs-6.12.ext2")
QEMU_SYS_MEMORY = "4G"  # Default memory for system mode
QEMU_SYS_SMP_DEFAULT = 4  # Default CPU cores for system mode
QEMU_USER_BIN = QEMU_USER_DIR / "bin" / "qemu-riscv64"
if DEBUG_QEMU:
    QEMU_USER_BIN = f"{QEMU_USER_BIN} -g 1234"
PERMAFROST_BIN = DEV_ENV_ROOT / "Permafrost" / "build" / "permafrost"
RESOURCE_DIR = PYSRC_ROOT / "resources"
XNNPACK_RESOURCES = RESOURCE_DIR / "xnnpack"
PROMPTS_DIR = RESOURCE_DIR / "prompts"
SIM_RESOURCES = RESOURCE_DIR / "simcfg"
SPIKE_DIR = DEV_ENV_ROOT / "IMachines_Spike"
BENCH_FILE = XNNPACK_RESOURCES / "benchmarks.txt"
RISCV_ENV_ROOT = DEV_ENV_ROOT / "riscv-env"
RISCV_ENV_INSTALL = RISCV_ENV_ROOT / "riscv"

NPU_ENV_ROOT = DEV_ENV_ROOT / "npu-env"
NPU_SDK = NPU_ENV_ROOT / "npu_driver" / "src" / "build" / "sdk"
NPU_ENV_INSTALL = NPU_ENV_ROOT / "npu"

RESULTS_DIR = IMINNT_ROOT / "results"
DEV_SCRIPT_PATHS = IMINNT_ROOT / "scripts"
APP_CXX_PATH = "clang"
APP_CC_PATH = "clang"
CPU_FREQ = 3.0e9

BASE_ENV = os.environ.copy()
IMI_CPU_ALIAS = "imicpu-v1"
# Old, simple architecture spec in case debugging is needed
# IMI_RISCV_ARCH = "RV64GCV_zfh_zvfh_zvl128b_ximimce"
IMI_RISCV_ARCH = "RV64GCV_ximimce_zba_zbb_zbs_zicntr_zihpm_zihintpause_zicbom_zicbop_zicboz_zihintntl_zicond_zcb_zfa_zawrs_zfh_zvfh_zfhmin_zvbb_zimop_zcmop_zfbfmin_zvfbfwma"
RVV_RISCV_ARCH = "RV64GCV_zba_zbb_zbs_zicntr_zihpm_zihintpause_zicbom_zicbop_zicboz_zihintntl_zicond_zcb_zfa_zawrs_zfh_zvfh_zvbb_zimop_zcmop_zfbfmin_zvfbfwma"
# This should always be set to True to enable debugging/troubleshooting, but in the rare scenario where debug symbols should not be included, you can set it to False here
USE_DEBUG_SYMS=True

AZURE_KEY_PATH = DEV_ENV_ROOT / "server_keys"
AZURE_BENCH_PATH = "/home/azureuser/benchmarking"

IMI_ENV = { 
    "CROSS_ARCH" :"riscv64",
    "CROSS_COMP" :"clang",
    "CROSS_TRIPLE" :"riscv64-unknown-linux-gnu",
    "CROSS_CPU" : IMI_RISCV_ARCH.lower(),
    "CROSS_TOOLCHAIN" : f"{IMI_SDK_ROOT}",
    "CROSS_SYSROOT" : f"{IMI_SDK_ROOT / 'sysroot'}",
    "IMI_LLVM_PATH" : f"{IMI_SDK_ROOT}",
    # "QEMU_LD_PREFIX": f"{IMI_SDK_ROOT / 'sysroot'}",
    "CROSS_QEMU_PATH" : f"{DEV_ENV_ROOT}/csqemu-v9/install-local/bin/qemu-riscv64",
    "RVV_VLEN": "128"
}

RVV_ENV = { 
    "CROSS_ARCH" :"riscv64",
    "CROSS_COMP" :"clang",
    "CROSS_TRIPLE" :"riscv64-unknown-linux-gnu",
    "CROSS_CPU" : RVV_RISCV_ARCH.lower(),
    "CROSS_TOOLCHAIN" : f"{IMI_SDK_ROOT}",
    "CROSS_SYSROOT" : f"{IMI_SDK_ROOT / 'sysroot'}",
    "IMI_LLVM_PATH" : f"{IMI_SDK_ROOT}",
    # "QEMU_LD_PREFIX": f"{IMI_SDK_ROOT / 'sysroot'}",
    "CROSS_QEMU_PATH" : f"{DEV_ENV_ROOT}/csqemu-v9/install-local/bin/qemu-riscv64",
    "RVV_VLEN": "128"
}

NEOVERSE_REMOTE = {
    "name": "neoverse_xnnpack",
    "uname": "azureuser",
    "host": "128.85.32.248",
    "port": 22,
    "bench_dir": AZURE_BENCH_PATH
}
NEOVERSE_ENV = {
    "CROSS_ARCH" :"aarch64",
    "CROSS_COMP" :"clang",
    "CROSS_TRIPLE" :"aarch64-linux-gnu",
    "CROSS_CPU" :"neoverse-n2",
    "CROSS_SYSROOT" : f"/",
    "IMI_LLVM_PATH" : f"/usr",
    "CROSS_QEMU_PATH" : f"/usr/bin/qemu-aarch64-static",
}

AMX_REMOTE = {
    "name": "amx_xnnpack",
    "uname": "azureuser",
    # "host": "20.163.72.67",
    "host": "98.71.34.100",
    "port": 22,
    "bench_dir": AZURE_BENCH_PATH,
    "use_agent": True
}
AMX_ENV = {
    "CROSS_ARCH" :"x86_64",
    "CROSS_COMP" :"clang",
    "CROSS_TRIPLE" :"x86_64-linux-gnu",
    "CROSS_CPU" :"sapphirerapids",
    "CROSS_SYSROOT" : f"/",
    "IMI_LLVM_PATH" : f"/usr",
}

ORYON_REMOTE = {
    "name": "amx_xnnpack",
    "uname": "azureuser",
    # "host": "20.163.72.67",
    "host": "98.71.34.100",
    "port": 22,
    "bench_dir": AZURE_BENCH_PATH,
    "use_agent": True
}

ORYON_ENV = {
    "CROSS_ARCH" :"aarch64",
    "CROSS_COMP" :"clang",
    "CROSS_TRIPLE" :"aarch64-unknown-linux-gnu",
    "CROSS_CPU" :"oryon-1",
    "CROSS_ARCH_VARIANT" :"armv8.6-a+dotprod+i8mm",
    "CROSS_SYSROOT" : f"/",
    "IMI_LLVM_PATH" : f"/usr",
    "CROSS_QEMU_PATH" : f"/usr/bin/qemu-aarch64-static"
}

NATIVE_ENV = {
    "CROSS_ARCH" :"x86_64",
    "CROSS_COMP" :"clang",
    "CROSS_TRIPLE" :"x86_64-linux-gnu",
    "CROSS_CPU" :"native",
    "CROSS_SYSROOT" : f"/",
    "IMI_LLVM_PATH" : f"/usr",
}