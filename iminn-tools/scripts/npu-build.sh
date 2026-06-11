#!/usr/bin/env bash
set -e
PROG=${0##*/}

usage() {
    echo "Usage: $0 [options]"
    echo "Commands:"
    echo "Options:"
    echo "--toolchain     Absolute path to the RISCV toolchain."
    echo "--install-dir   Specify the name of the install directory. Defaults to 'install-local'"
}

abort() {
  echo "Error: $1" >&2
  cat << EOF >&2
Usage: $PROG --toolchain <toolchain-dir> --build-dir <build-dir> --linux-kernel <linux-kernel-dir> [--clean]
Build npu drivers.
EOF
  exit 1
}

TOOLCHAIN=
CLEAN=false
KERNEL_PATH=

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--toolchain)
            TOOLCHAIN=$2
            shift 2
        ;;
        -b|--build-dir)
            BUILD_DIR=$2
            shift 2
        ;;
        -l|--linux-kernel)
            KERNEL_PATH=$2
            shift 1
        ;;
        -c|--clean)
            CLEAN=true
            shift 1
        ;;
        -*)
            usage
            abort "Unknown option: $1"
            ;;
        *)
            break
            ;;
    esac
done

if [ -z "${TOOLCHAIN}" ]; then
    abort "Toolchain path is required"
fi

if [[ $TOOLCHAIN != /* ]]; then
    abort "Toolchain path must be an absolute path"
fi

if [ -z "${KERNEL_PATH}" ]; then
    abort "Linux kernel path is required"
fi

if [[ $KERNEL_PATH != /* ]]; then
    abort "Linux kernel path must be an absolute path"
fi


########################################################
# establish build environment and build options value
# Please modify the following items according your build environment

ARCH=riscv


export AQROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/src
export AQARCH=$AQROOT/arch/XAQ2

export SDK_DIR=$AQROOT/build/sdk
export OVXLIB_DIR=$AQROOT/Vivante_ML_Toolkit_OVXLIB_dev
export FIXED_ARCH_TYPE=riscv64-unknown-linux-gnu
export ARCH_TYPE=$ARCH
export CPU_TYPE=0
export KERNEL_DIR=$KERNEL_PATH
export CROSS_COMPILE=riscv64-unknown-linux-gnu-
export TOOLCHAIN=$TOOLCHAIN
export LIB_DIR=$TOOLCHAIN/sysroot/usr/lib
export ENABLE_40BIT_VA=1
export MMU_40VA_40PA=1
export GBM_LIBS=
export USE_OVXLIB=1
export SOC_PLATFORM=vdk
export OPT_VIVANTE_NO_GL4=1


########################################################
# Driver Build Options
#
BUILD_OPTION_ABI=0
BUILD_OPTION_LINUX_OABI=0
BUILD_OPTION_NO_DMA_COHERENT=0
BUILD_OPTION_USE_VDK=1
BUILD_OPTION_gcdSTATIC_LINK=0
BUILD_OPTION_CUSTOM_PIXMAP=0
BUILD_OPTION_USE_FB_DOUBLE_BUFFER=0
BUILD_OPTION_USE_PLATFORM_DRIVER=1
BUILD_OPTION_FPGA_BUILD=1
BUILD_OPTION_LINUX_EMULATOR=0
BUILD_OPTION_VIVANTE_ENABLE_40BITS_VA=1
BUILD_OPTION_MMU_40VA_40PA=1

BUILD_OPTION_USE_OPENCL=1
BUILD_OPTION_USE_OPENVX=1
BUILD_OPTION_USE_OVXLIB=0


BUILD_OPTION_DEBUG=0
if [ -z $BUILD_OPTION_EGL_API_FB ]; then
    BUILD_OPTION_EGL_API_FB=1
fi
if [ -z $BUILD_OPTION_EGL_API_DFB ]; then
    BUILD_OPTION_EGL_API_DFB=0
fi
if [ -z $BUILD_OPTION_EGL_API_DRI ]; then
    BUILD_OPTION_EGL_API_DRI=0
fi
if [ -z $BUILD_OPTION_X11_DRI3 ]; then
    BUILD_OPTION_X11_DRI3=0
fi
if [ -z $BUILD_OPTION_EGL_API_WL ]; then
    BUILD_OPTION_EGL_API_WL=0
fi
if [ -z $BUILD_OPTION_EGL_API_X ]; then
    BUILD_OPTION_EGL_API_X=0
fi
if [ -z $BUILD_OPTION_EGL_API_GBM ]; then
    BUILD_OPTION_EGL_API_GBM=0
fi
if [ -z $BUILD_OPTION_EGL_API_NULLWS ]; then
    BUILD_OPTION_EGL_API_NULLWS=0
fi

if [ -z $BUILD_OPTION_USE_OPENCL ]; then
    if $(ls ${AQROOT}/driver/khronos/ | grep -qE 'libCL');then
        BUILD_OPTION_USE_OPENCL=1
    else
        BUILD_OPTION_USE_OPENCL=0
    fi
fi

if [ -z $BUILD_OPTION_USE_OPENVX ]; then
    if $(ls ${AQROOT}/driver/khronos/ | grep -qE 'libOpenVX');then
        BUILD_OPTION_USE_OPENVX=1
    else
        BUILD_OPTION_USE_OPENVX=0
    fi
fi

if [ -z $BUILD_OPTION_USE_VULKAN ]; then
    if $(ls ${AQROOT}/driver/khronos/ | grep -qE 'libVulkan');then
        BUILD_OPTION_USE_VULKAN=1
    else
        BUILD_OPTION_USE_VULKAN=0
    fi
fi

if [ -z $BUILD_OPTION_ENABLE_GPU_CLOCK_BY_DRIVER ]; then
    BUILD_OPTION_ENABLE_GPU_CLOCK_BY_DRIVER=0
fi
if [ -z $BUILD_OPTION_GL4_DRI_BUILD ]; then
    BUILD_OPTION_GL4_DRI_BUILD=0
fi

if [ -z $BUILD_OPTION_VIVANTE_NO_GL4 ]; then
    if $(ls ${AQROOT}/driver/khronos/ | grep -qE 'libGL[0-9]+');then
        BUILD_OPTION_VIVANTE_NO_GL4=0
    else
        BUILD_OPTION_VIVANTE_NO_GL4=1
    fi
fi

#if [ -z $BUILD_OPTION_VIVANTE_NO_VG ]; then
#    BUILD_OPTION_VIVANTE_NO_VG=0
#fi
if [ -z $BUILD_OPTION_VIVANTE_NO_VG ]; then
    if $(ls ${AQROOT}/driver/khronos/ | grep -qE 'libOpenVG');then
        BUILD_OPTION_VIVANTE_NO_VG=0
    else
        BUILD_OPTION_VIVANTE_NO_VG=1
    fi
fi

if [ -z $BUILD_OPTION_VIVANTE_ENABLE_DRM ]; then
    BUILD_OPTION_VIVANTE_ENABLE_DRM=0
fi
if [ -z $BUILD_OPTION_VIVANTE_ENABLE_3D ]; then
    BUILD_OPTION_VIVANTE_ENABLE_3D=1
fi
if [ -z $BUILD_OPTION_VIVANTE_ENABLE_2D ]; then
    BUILD_OPTION_VIVANTE_ENABLE_2D=0
fi

if [ "$ARCH" = "X86_PCIE" ]; then
    BUILD_OPTION_NO_DMA_COHERENT=1
    BUILD_OPTION_ENABLE_ARM_L2_CACHE=0
    BUILD_OPTION_gcdNO_POWER_MANAGEMENT=1
    BUILD_OPTION_USE_POWER_MANAGEMENT=0
    BUILD_OPTION_LINUX_EMULATOR=0
    BUILD_OPTION_USE_PLATFORM_DRIVER=0
    BUILD_OPTION_FPGA_BUILD=1
fi

if [ "$ARCH" = "X86_CMODEL" ]; then
    BUILD_OPTION_NO_DMA_COHERENT=1
    BUILD_OPTION_ENABLE_ARM_L2_CACHE=0
    BUILD_OPTION_gcdNO_POWER_MANAGEMENT=0
    BUILD_OPTION_USE_POWER_MANAGEMENT=1
    BUILD_OPTION_LINUX_EMULATOR=1
fi

BUILD_OPTIONS="NO_DMA_COHERENT=$BUILD_OPTION_NO_DMA_COHERENT"
BUILD_OPTIONS="$BUILD_OPTIONS USE_VDK=$BUILD_OPTION_USE_VDK"
BUILD_OPTIONS="$BUILD_OPTIONS EGL_API_WL=$BUILD_OPTION_EGL_API_WL"
BUILD_OPTIONS="$BUILD_OPTIONS EGL_API_FB=$BUILD_OPTION_EGL_API_FB"
BUILD_OPTIONS="$BUILD_OPTIONS EGL_API_DFB=$BUILD_OPTION_EGL_API_DFB"
BUILD_OPTIONS="$BUILD_OPTIONS EGL_API_DRI=$BUILD_OPTION_EGL_API_DRI"
BUILD_OPTIONS="$BUILD_OPTIONS EGL_API_X=$BUILD_OPTION_EGL_API_X"
BUILD_OPTIONS="$BUILD_OPTIONS EGL_API_GBM=$BUILD_OPTION_EGL_API_GBM"
BUILD_OPTIONS="$BUILD_OPTIONS X11_DRI3=$BUILD_OPTION_X11_DRI3"
BUILD_OPTIONS="$BUILD_OPTIONS EGL_API_NULLWS=$BUILD_OPTION_EGL_API_NULLWS"
BUILD_OPTIONS="$BUILD_OPTIONS gcdSTATIC_LINK=$BUILD_OPTION_gcdSTATIC_LINK"
BUILD_OPTIONS="$BUILD_OPTIONS ABI=$BUILD_OPTION_ABI"
BUILD_OPTIONS="$BUILD_OPTIONS LINUX_OABI=$BUILD_OPTION_LINUX_OABI"
BUILD_OPTIONS="$BUILD_OPTIONS DEBUG=$BUILD_OPTION_DEBUG"
BUILD_OPTIONS="$BUILD_OPTIONS CUSTOM_PIXMAP=$BUILD_OPTION_CUSTOM_PIXMAP"
BUILD_OPTIONS="$BUILD_OPTIONS USE_OPENCL=$BUILD_OPTION_USE_OPENCL"
BUILD_OPTIONS="$BUILD_OPTIONS USE_OPENVX=$BUILD_OPTION_USE_OPENVX"
BUILD_OPTIONS="$BUILD_OPTIONS USE_VULKAN=$BUILD_OPTION_USE_VULKAN"
BUILD_OPTIONS="$BUILD_OPTIONS USE_FB_DOUBLE_BUFFER=$BUILD_OPTION_USE_FB_DOUBLE_BUFFER"
BUILD_OPTIONS="$BUILD_OPTIONS USE_PLATFORM_DRIVER=$BUILD_OPTION_USE_PLATFORM_DRIVER"
BUILD_OPTIONS="$BUILD_OPTIONS ENABLE_GPU_CLOCK_BY_DRIVER=$BUILD_OPTION_ENABLE_GPU_CLOCK_BY_DRIVER"
BUILD_OPTIONS="$BUILD_OPTIONS FPGA_BUILD=$BUILD_OPTION_FPGA_BUILD"
BUILD_OPTIONS="$BUILD_OPTIONS GL4_DRI_BUILD=$BUILD_OPTION_GL4_DRI_BUILD"
BUILD_OPTIONS="$BUILD_OPTIONS VIVANTE_NO_GL4=$BUILD_OPTION_VIVANTE_NO_GL4"
BUILD_OPTIONS="$BUILD_OPTIONS VIVANTE_NO_VG=$BUILD_OPTION_VIVANTE_NO_VG"
BUILD_OPTIONS="$BUILD_OPTIONS VIVANTE_ENABLE_DRM=$BUILD_OPTION_VIVANTE_ENABLE_DRM"
BUILD_OPTIONS="$BUILD_OPTIONS VIVANTE_ENABLE_3D=$BUILD_OPTION_VIVANTE_ENABLE_3D"
BUILD_OPTIONS="$BUILD_OPTIONS VIVANTE_ENABLE_2D=$BUILD_OPTION_VIVANTE_ENABLE_2D"
BUILD_OPTIONS="$BUILD_OPTIONS LINUX_EMULATOR=$BUILD_OPTION_LINUX_EMULATOR"
BUILD_OPTIONS="$BUILD_OPTIONS VIVANTE_ENABLE_40BITS_VA=$BUILD_OPTION_VIVANTE_ENABLE_40BITS_VA"
BUILD_OPTIONS="$BUILD_OPTIONS MMU_40VA_40PA=$BUILD_OPTION_MMU_40VA_40PA"

BUILD_OPTIONS="$BUILD_OPTIONS USE_VIP_ONLY=1"

MULTIJOBS=4
BUILD_OPTIONS="$BUILD_OPTIONS -j${MULTIJOBS}"

export PATH=$TOOLCHAIN/bin:$PATH
########################################################
# clean/build driver and samples
# build results will save to $SDK_DIR/
#

if [[ "$CLEAN" = true ]]; then
    cd $AQROOT; make -j1 -f makefile.linux $BUILD_OPTIONS clean
    cd $OVXLIB_DIR; make -j1 -f makefile.linux $BUILD_OPTIONS clean

fi
cd $AQROOT; make -j1 -f makefile.linux $BUILD_OPTIONS install
cd $OVXLIB_DIR; make -j1 -f makefile.linux $BUILD_OPTIONS install


